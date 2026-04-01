#!/usr/bin/env python3
import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import math
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _extract_action_array(action_value) -> np.ndarray:
    if isinstance(action_value, np.ndarray):
        arr = action_value
    else:
        arr = np.asarray(action_value)
    return arr.astype(np.float32)


def load_task_texts(dataset_root: Path) -> dict[int, str]:
    tasks_file = dataset_root / "meta" / "tasks.parquet"
    if not tasks_file.exists():
        return {}

    df = pd.read_parquet(tasks_file)
    if len(df) == 0:
        return {}

    idx_col = None
    for c in ["task_index", "index", "id"]:
        if c in df.columns:
            idx_col = c
            break
    if idx_col is None:
        for c in df.columns:
            if pd.api.types.is_integer_dtype(df[c]):
                idx_col = c
                break
    if idx_col is None:
        return {}

    text_col = None
    for c in ["task", "task_description", "instruction", "language", "text"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        for c in df.columns:
            if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
                text_col = c
                break
    if text_col is None:
        return {}

    out = {}
    for _, row in df.iterrows():
        try:
            out[int(row[idx_col])] = str(row[text_col])
        except Exception:
            continue
    return out


@dataclass
class EpisodeData:
    actions: np.ndarray
    task_index: int


class LiberoActionChunkDataset(Dataset):
    def __init__(self, dataset_root: Path, chunk_size: int = 16, stride: int = 4):
        self.chunk_size = chunk_size
        self.stride = stride

        parquet_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

        frames = []
        cols = ["episode_index", "frame_index", "task_index", "action"]
        for f in parquet_files:
            frames.append(pd.read_parquet(f, columns=cols))
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

        self.episodes: dict[int, EpisodeData] = {}
        for ep, g in df.groupby("episode_index", sort=True):
            actions = np.stack([_extract_action_array(v) for v in g["action"].tolist()], axis=0)
            task_idx = int(g["task_index"].iloc[0])
            self.episodes[int(ep)] = EpisodeData(actions=actions, task_index=task_idx)

        self.samples: list[tuple[int, int]] = []
        for ep, ep_data in self.episodes.items():
            T = ep_data.actions.shape[0]
            if T < self.chunk_size:
                continue
            for start in range(0, T - self.chunk_size + 1, self.stride):
                self.samples.append((ep, start))

        if not self.samples:
            raise RuntimeError("No valid action chunks were generated. Check chunk_size/stride.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep, start = self.samples[idx]
        ep_data = self.episodes[ep]
        act = ep_data.actions[start : start + self.chunk_size]
        return {
            "actions": torch.from_numpy(act),
            "task_index": torch.tensor(ep_data.task_index, dtype=torch.long),
        }


def build_model(args, device):
    from lerobot_policy_mint.mint_utils import MultiScaleVQVAE

    spectral_scale_weights = None
    if args.spectral_scale_weights.strip():
        spectral_scale_weights = [
            float(x.strip()) for x in args.spectral_scale_weights.split(",") if x.strip()
        ]

    model = MultiScaleVQVAE(
        seq_dim=7,
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
        ch=args.ch,
        patch_nums=(1, 2, 4),
        ch_mult=(2, 4, 8),
        beta=args.vq_beta,
        patchwise={"enable": True, "d_embed": 8, "grouped_depth": 2, "norm": "layer"},
        tokenizer_align_enable=args.align,
        tokenizer_align_model_name=args.align_model,
        tokenizer_align_proj_dim=args.align_dim,
        tokenizer_align_temperature=args.align_temp,
        tokenizer_align_weight=args.align_weight,
        tokenizer_align_warmup_steps=args.align_warmup,
        tokenizer_align_max_length=args.align_max_len,
        tokenizer_aux_l1_weight=args.aux_l1_weight,
        tokenizer_vq_weight=args.vq_weight,
        tokenizer_spectral_weight=args.spectral_weight,
        tokenizer_spectral_exclude_last_dim=not args.include_gripper_in_spectral,
        tokenizer_spectral_scale_weights=spectral_scale_weights,
        tokenizer_spectral_normalize_by_scale_sum=not args.disable_spectral_scale_normalization,
    )
    if args.tokenizer_ckpt:
        model.load_vqvae_weights(args.tokenizer_ckpt)
    return model.to(device)


def save_checkpoint(model, optimizer, step: int, out_dir: Path):
    ckpt_dir = out_dir / "checkpoints" / f"{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_dir / "tokenizer.pt",
    )
    last = out_dir / "checkpoints" / "last"
    if last.exists() or last.is_symlink():
        last.unlink()
    last.symlink_to(ckpt_dir.name)


def parse_args():
    p = argparse.ArgumentParser(description="Train MINT tokenizer on LIBERO with optional BGE align loss.")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--chunk_size", type=int, default=16)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--lr_peak_scale", type=float, default=0.5)
    p.add_argument("--lr_warmup_steps", type=int, default=100)
    p.add_argument("--lr_warmup_init_ratio", type=float, default=0.1)
    p.add_argument("--lr_min_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log_freq", type=int, default=10)
    p.add_argument("--save_freq", type=int, default=200)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--tokenizer_ckpt", type=str, default="")

    p.add_argument("--align", action="store_true")
    p.add_argument("--align_model", type=str, default="BAAI/bge-large-en-v1.5")
    p.add_argument("--align_dim", type=int, default=256)
    p.add_argument("--align_temp", type=float, default=0.07)
    p.add_argument("--align_weight", type=float, default=0.1)
    p.add_argument("--align_warmup", type=int, default=1000)
    p.add_argument("--align_max_len", type=int, default=64)

    p.add_argument("--aux_l1_weight", type=float, default=1.0)
    p.add_argument("--vq_weight", type=float, default=1.0)
    p.add_argument("--vq_beta", type=float, default=0.25)
    p.add_argument("--spectral_weight", type=float, default=1.0)
    p.add_argument(
        "--spectral_scale_weights",
        type=str,
        default="",
        help="Comma-separated per-scale weights for spectral loss (e.g., '1,1,1').",
    )
    p.add_argument(
        "--disable_spectral_scale_normalization",
        action="store_true",
        help="If set, do not normalize spectral loss by sum of scale weights.",
    )
    p.add_argument(
        "--include_gripper_in_spectral",
        action="store_true",
        help="If set, include the last action dim in DCT spectral loss. By default it is excluded.",
    )

    p.add_argument("--codebook_size", type=int, default=512)
    p.add_argument("--codebook_dim", type=int, default=32)
    p.add_argument("--ch", type=int, default=48)
    return p.parse_args()


def build_lr_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    if args.scheduler == "none":
        return None

    total_steps = max(1, int(args.steps))
    warmup_steps = min(max(0, int(args.lr_warmup_steps)), total_steps)
    init_ratio = float(args.lr_warmup_init_ratio)
    min_ratio = float(args.lr_min_ratio)

    def lr_lambda(step_idx: int) -> float:
        step = step_idx + 1
        if warmup_steps > 0 and step <= warmup_steps:
            progress = step / float(warmup_steps)
            return init_ratio + (1.0 - init_ratio) * progress

        if total_steps <= warmup_steps:
            return 1.0

        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_texts = load_task_texts(dataset_root)
    ds = LiberoActionChunkDataset(dataset_root=dataset_root, chunk_size=args.chunk_size, stride=args.stride)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    it = iter(loader)

    model = build_model(args, device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.lr_peak_scale <= 0:
        raise ValueError("lr_peak_scale must be > 0")
    if not (0 < args.lr_warmup_init_ratio <= 1.0):
        raise ValueError("lr_warmup_init_ratio must be in (0, 1]")
    if not (0 < args.lr_min_ratio <= 1.0):
        raise ValueError("lr_min_ratio must be in (0, 1]")

    peak_lr = args.lr * args.lr_peak_scale
    optimizer = torch.optim.AdamW(trainable, lr=peak_lr, weight_decay=args.weight_decay)
    scheduler = build_lr_scheduler(optimizer, args)

    history = []
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        actions = batch["actions"].to(device=device, dtype=torch.float32)
        task_idx = batch["task_index"].tolist()
        texts = [task_texts.get(int(i), f"task_{int(i)}") for i in task_idx]

        losses = model.compute_tokenizer_losses(
            actions,
            texts=texts if args.align else None,
            global_step=step,
        )
        loss = losses["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        current_lr = float(optimizer.param_groups[0]["lr"])

        item = {
            "step": step,
            "loss": float(losses["loss"].detach().cpu().item()),
            "recon_loss": float(losses["recon_loss"].detach().cpu().item()),
            "freq_loss": float(losses["freq_loss"].detach().cpu().item()),
            "aux_l1_loss": float(losses["aux_l1_loss"].detach().cpu().item()),
            "aux_l1_weight": float(losses["aux_l1_weight"].detach().cpu().item()),
            "vq_loss": float(losses["vq_loss"].detach().cpu().item()),
            "vq_loss_raw": float(losses["vq_loss_raw"].detach().cpu().item()),
            "vq_weight": float(losses["vq_weight"].detach().cpu().item()),
            "align_loss": float(losses["align_loss"].detach().cpu().item()),
            "align_loss_raw": float(losses["align_loss_raw"].detach().cpu().item()),
            "align_weight": float(losses["align_weight"].detach().cpu().item()),
            "lr": current_lr,
        }
        history.append(item)

        if step == 1 or step % args.log_freq == 0 or step == args.steps:
            print(
                f"step:{step} loss:{item['loss']:.4f} recon:{item['recon_loss']:.4f} "
                f"freq:{item['freq_loss']:.4f} aux_l1:{item['aux_l1_loss']:.4f} "
                f"aux_w:{item['aux_l1_weight']:.4f} "
                f"vq:{item['vq_loss']:.4f} vq_raw:{item['vq_loss_raw']:.4f} "
                f"vq_w:{item['vq_weight']:.4f} align:{item['align_loss']:.4f} "
                f"align_raw:{item['align_loss_raw']:.4f} w:{item['align_weight']:.4f} "
                f"lr:{item['lr']:.6e}",
                flush=True,
            )

        if step % args.save_freq == 0 or step == args.steps:
            save_checkpoint(model, optimizer, step, out_dir)

    report = {
        "args": vars(args),
        "device": str(device),
        "dataset_chunks": len(ds),
        "first_loss": history[0]["loss"],
        "last_loss": history[-1]["loss"],
        "history_tail": history[-min(50, len(history)):],
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to: {out_dir / 'train_report.json'}")


if __name__ == "__main__":
    main()