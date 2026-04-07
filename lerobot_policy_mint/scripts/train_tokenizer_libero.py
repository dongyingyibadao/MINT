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

    idx_is_text = False
    if len(df.index) > 0:
        idx_is_text = isinstance(df.index[0], str)
    if idx_is_text:
        out = {}
        for _, row in df.iterrows():
            try:
                out[int(row[idx_col])] = str(row.name)
            except Exception:
                continue
        return out

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


class LiberoEpisodeDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        temporal_stride: int = 1,
        min_episode_len: int = 16,
        max_episode_len: int = 0,
    ):
        self.temporal_stride = max(1, int(temporal_stride))
        self.min_episode_len = max(1, int(min_episode_len))
        self.max_episode_len = max(0, int(max_episode_len))

        parquet_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

        frames = []
        cols = ["episode_index", "frame_index", "task_index", "action"]
        for f in parquet_files:
            frames.append(pd.read_parquet(f, columns=cols))
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

        self.episodes: list[EpisodeData] = []
        for _, g in df.groupby("episode_index", sort=True):
            actions = np.stack([_extract_action_array(v) for v in g["action"].tolist()], axis=0)
            actions = actions[:: self.temporal_stride]
            if self.max_episode_len > 0 and actions.shape[0] > self.max_episode_len:
                start = random.randint(0, actions.shape[0] - self.max_episode_len)
                actions = actions[start : start + self.max_episode_len]
            if actions.shape[0] < self.min_episode_len:
                continue
            task_idx = int(g["task_index"].iloc[0])
            self.episodes.append(EpisodeData(actions=actions, task_index=task_idx))

        if not self.episodes:
            raise RuntimeError("No valid episodes found. Check min/max length and temporal_stride.")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        return {
            "actions": torch.from_numpy(ep.actions),
            "task_index": torch.tensor(ep.task_index, dtype=torch.long),
            "episode_len": torch.tensor(ep.actions.shape[0], dtype=torch.long),
        }


def collate_episode_batch(batch):
    B = len(batch)
    max_len = max(int(item["actions"].shape[0]) for item in batch)
    action_dim = int(batch[0]["actions"].shape[1])

    actions = torch.zeros((B, max_len, action_dim), dtype=torch.float32)
    action_mask = torch.zeros((B, max_len), dtype=torch.bool)
    task_index = torch.zeros((B,), dtype=torch.long)

    for i, item in enumerate(batch):
        t = int(item["actions"].shape[0])
        actions[i, :t] = item["actions"].float()
        action_mask[i, :t] = True
        task_index[i] = item["task_index"]

    return {
        "actions": actions,
        "action_mask": action_mask,
        "task_index": task_index,
    }


def build_episode_chunks(
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    chunk_size: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunks = []
    owners = []
    B = actions.shape[0]

    for b in range(B):
        valid_len = int(action_mask[b].sum().item())
        if valid_len < chunk_size:
            continue
        starts = list(range(0, valid_len - chunk_size + 1, stride))
        last = valid_len - chunk_size
        if len(starts) == 0 or starts[-1] != last:
            starts.append(last)

        for s in starts:
            chunks.append(actions[b, s : s + chunk_size])
            owners.append(b)

    if not chunks:
        raise RuntimeError("No valid chunks generated from episode batch. Check chunk_size/stride.")

    return torch.stack(chunks, dim=0), torch.tensor(owners, dtype=torch.long, device=actions.device)


def concat_chunk_tokens_by_episode(
    chunk_tokens: torch.Tensor,
    owners: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # chunk_tokens: [N_chunk, T_tok, C]
    token_lists = [[] for _ in range(batch_size)]
    for i in range(chunk_tokens.shape[0]):
        owner = int(owners[i].item())
        token_lists[owner].append(chunk_tokens[i])

    token_dim = int(chunk_tokens.shape[-1])
    max_tokens = max(sum(int(t.shape[0]) for t in ls) for ls in token_lists)
    all_tokens = chunk_tokens.new_zeros((batch_size, max_tokens, token_dim))
    all_masks = torch.zeros((batch_size, max_tokens), dtype=torch.bool, device=chunk_tokens.device)

    for b, seq_list in enumerate(token_lists):
        if not seq_list:
            continue
        seq = torch.cat(seq_list, dim=0)
        t = int(seq.shape[0])
        all_tokens[b, :t] = seq
        all_masks[b, :t] = True

    return all_tokens, all_masks


def build_model(args, device):
    from lerobot_policy_mint.mint_utils import MultiScaleVQVAE

    def _parse_int_csv(raw: str) -> list[int]:
        vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
        if len(vals) == 0:
            raise ValueError("Expected at least one integer in comma-separated list")
        return vals

    spectral_scale_weights = None
    if args.spectral_scale_weights.strip():
        spectral_scale_weights = [
            float(x.strip()) for x in args.spectral_scale_weights.split(",") if x.strip()
        ]

    local_action_window_sizes = _parse_int_csv(args.align_local_action_window_sizes)
    local_text_window_sizes = _parse_int_csv(args.align_local_text_window_sizes)

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
        tokenizer_align_global_weight=args.align_global_weight,
        tokenizer_align_local_weight=args.align_local_weight,
        tokenizer_align_local_mode=args.align_local_mode,
        tokenizer_align_local_action_window_sizes=local_action_window_sizes,
        tokenizer_align_local_text_window_sizes=local_text_window_sizes,
        tokenizer_align_local_topk=args.align_local_topk,
        tokenizer_align_local_window_temperature=args.align_local_window_temperature,
        tokenizer_align_order_weight=args.align_order_weight,
        tokenizer_align_order_mode=args.align_order_mode,
        tokenizer_align_order_window_temperature=args.align_order_window_temperature,
        tokenizer_align_quant_weight=args.align_quant_weight,
        tokenizer_align_encoder_weight=args.align_encoder_weight,
        tokenizer_align_order_margin=args.align_order_margin,
        tokenizer_align_warmup_steps=args.align_warmup,
        tokenizer_align_order_warmup_steps=args.align_order_warmup,
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
    p.add_argument(
        "--episode_level_align",
        action="store_true",
        help="If set, sample full demonstrations (episode-level) instead of fixed action chunks.",
    )
    p.add_argument("--temporal_stride", type=int, default=1)
    p.add_argument("--min_episode_len", type=int, default=16)
    p.add_argument(
        "--max_episode_len",
        type=int,
        default=0,
        help="Optional cap on episode length after temporal stride (0 means no cap).",
    )
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
    p.add_argument(
        "--align_global_weight",
        type=float,
        default=None,
        help="Global demo-language align weight. Defaults to --align_weight when omitted.",
    )
    p.add_argument("--align_local_weight", type=float, default=0.0)
    p.add_argument(
        "--align_local_mode",
        type=str,
        default="window",
        choices=["token", "window"],
        help="Local align matching unit.",
    )
    p.add_argument(
        "--align_local_action_window_sizes",
        type=str,
        default="1,2",
        help="Comma-separated causal window sizes for action local matching.",
    )
    p.add_argument(
        "--align_local_text_window_sizes",
        type=str,
        default="1,2,3",
        help="Comma-separated window sizes for text local matching.",
    )
    p.add_argument(
        "--align_local_topk",
        type=int,
        default=2,
        help="Top-k text windows used in soft aggregation.",
    )
    p.add_argument(
        "--align_local_window_temperature",
        type=float,
        default=0.07,
        help="Temperature for soft top-k window aggregation.",
    )
    p.add_argument("--align_order_weight", type=float, default=0.0)
    p.add_argument(
        "--align_order_mode",
        type=str,
        default="window",
        choices=["token", "window"],
        help="Order align matching unit.",
    )
    p.add_argument(
        "--align_order_window_temperature",
        type=float,
        default=0.07,
        help="Temperature for order align when using window mode.",
    )
    p.add_argument(
        "--align_quant_weight",
        type=float,
        default=1.0,
        help="Weight of quantized-token align branch (directly updates codebook rows).",
    )
    p.add_argument(
        "--align_encoder_weight",
        type=float,
        default=1.0,
        help="Weight of encoder-ST align branch (primarily updates encoder).",
    )
    p.add_argument("--align_order_margin", type=float, default=0.02)
    p.add_argument("--align_warmup", type=int, default=1000)
    p.add_argument("--align_order_warmup", type=int, default=3000)
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
    if args.episode_level_align:
        ds = LiberoEpisodeDataset(
            dataset_root=dataset_root,
            temporal_stride=args.temporal_stride,
            min_episode_len=args.min_episode_len,
            max_episode_len=args.max_episode_len,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            collate_fn=collate_episode_batch,
        )
    else:
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
        action_mask = batch.get("action_mask")
        if action_mask is not None:
            action_mask = action_mask.to(device=device)

        if args.episode_level_align:
            if action_mask is None:
                raise RuntimeError("episode_level_align requires action_mask in batch")

            chunk_actions, chunk_owners = build_episode_chunks(
                actions,
                action_mask,
                chunk_size=args.chunk_size,
                stride=args.stride,
            )
            valid_owner_ids = sorted(set(int(x) for x in chunk_owners.detach().cpu().tolist()))

            # Demo-balanced chunk loss: average within each demo first, then average across demos.
            per_demo_losses = []
            for owner in valid_owner_ids:
                owner_mask = chunk_owners == owner
                owner_chunks = chunk_actions[owner_mask]
                per_demo_losses.append(
                    model.compute_tokenizer_losses(
                        owner_chunks,
                        texts=None,
                        global_step=step,
                    )
                )

            chunk_losses = {}
            for k in per_demo_losses[0].keys():
                chunk_losses[k] = torch.stack([d[k] for d in per_demo_losses], dim=0).mean(dim=0)

            align_aux = {
                "align_loss": torch.zeros_like(chunk_losses["loss"]),
                "align_loss_raw": torch.zeros_like(chunk_losses["loss"]),
                "align_weight": torch.zeros_like(chunk_losses["loss"]),
                "align_global_loss": torch.zeros_like(chunk_losses["loss"]),
                "align_global_loss_raw": torch.zeros_like(chunk_losses["loss"]),
                "align_global_weight": torch.zeros_like(chunk_losses["loss"]),
                "align_local_loss": torch.zeros_like(chunk_losses["loss"]),
                "align_local_loss_raw": torch.zeros_like(chunk_losses["loss"]),
                "align_local_weight": torch.zeros_like(chunk_losses["loss"]),
                "align_order_loss": torch.zeros_like(chunk_losses["loss"]),
                "align_order_loss_raw": torch.zeros_like(chunk_losses["loss"]),
                "align_order_weight": torch.zeros_like(chunk_losses["loss"]),
            }

            if args.align:
                task_idx = batch["task_index"].tolist()
                texts_full = [task_texts.get(int(i), f"task_{int(i)}") for i in task_idx]
                texts = [texts_full[i] for i in valid_owner_ids]
                text_feat = model.encode_text_features(texts, device=device)

                chunk_s1, chunk_s1_st = model.inp_to_s1_token_pair(chunk_actions)
                remapped_owners = torch.empty_like(chunk_owners)
                for new_owner, old_owner in enumerate(valid_owner_ids):
                    remapped_owners[chunk_owners == old_owner] = new_owner

                demo_tokens, demo_token_mask = concat_chunk_tokens_by_episode(
                    chunk_s1,
                    remapped_owners,
                    batch_size=len(valid_owner_ids),
                )
                demo_tokens_st, _ = concat_chunk_tokens_by_episode(
                    chunk_s1_st,
                    remapped_owners,
                    batch_size=len(valid_owner_ids),
                )
                align_aux = model.compute_align_loss(
                    action_tokens=demo_tokens,
                    text_global=text_feat["text_global"],
                    text_tokens=text_feat["text_tokens"],
                    text_mask=text_feat["text_mask"],
                    action_mask=demo_token_mask,
                    global_step=step,
                    action_tokens_encoder_st=demo_tokens_st,
                )

            losses = dict(chunk_losses)
            losses["loss"] = chunk_losses["loss"] + align_aux["align_loss"]
            losses["align_loss"] = align_aux["align_loss"]
            losses["align_loss_raw"] = align_aux["align_loss_raw"]
            losses["align_weight"] = align_aux["align_weight"]
            losses["align_global_loss"] = align_aux["align_global_loss"]
            losses["align_global_loss_raw"] = align_aux["align_global_loss_raw"]
            losses["align_global_weight"] = align_aux["align_global_weight"]
            losses["align_local_loss"] = align_aux["align_local_loss"]
            losses["align_local_loss_raw"] = align_aux["align_local_loss_raw"]
            losses["align_local_weight"] = align_aux["align_local_weight"]
            losses["align_order_loss"] = align_aux["align_order_loss"]
            losses["align_order_loss_raw"] = align_aux["align_order_loss_raw"]
            losses["align_order_weight"] = align_aux["align_order_weight"]
        else:
            task_idx = batch["task_index"].tolist()
            texts = [task_texts.get(int(i), f"task_{int(i)}") for i in task_idx]

            losses = model.compute_tokenizer_losses(
                actions,
                action_mask=action_mask,
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
            "align_global_loss": float(losses["align_global_loss"].detach().cpu().item()),
            "align_global_loss_raw": float(losses["align_global_loss_raw"].detach().cpu().item()),
            "align_global_weight": float(losses["align_global_weight"].detach().cpu().item()),
            "align_local_loss": float(losses["align_local_loss"].detach().cpu().item()),
            "align_local_loss_raw": float(losses["align_local_loss_raw"].detach().cpu().item()),
            "align_local_weight": float(losses["align_local_weight"].detach().cpu().item()),
            "align_order_loss": float(losses["align_order_loss"].detach().cpu().item()),
            "align_order_loss_raw": float(losses["align_order_loss_raw"].detach().cpu().item()),
            "align_order_weight": float(losses["align_order_weight"].detach().cpu().item()),
            "lr": current_lr,
        }
        if action_mask is not None:
            item["avg_episode_len"] = float(action_mask.sum(dim=1).float().mean().detach().cpu().item())
            item["action_valid_ratio"] = float(action_mask.float().mean().detach().cpu().item())
        else:
            item["avg_episode_len"] = float(actions.shape[1])
            item["action_valid_ratio"] = 1.0
        history.append(item)

        if step == 1 or step % args.log_freq == 0 or step == args.steps:
            print(
                f"step:{step} loss:{item['loss']:.4f} recon:{item['recon_loss']:.4f} "
                f"freq:{item['freq_loss']:.4f} aux_l1:{item['aux_l1_loss']:.4f} "
                f"aux_w:{item['aux_l1_weight']:.4f} "
                f"vq:{item['vq_loss']:.4f} vq_raw:{item['vq_loss_raw']:.4f} "
                f"vq_w:{item['vq_weight']:.4f} align:{item['align_loss']:.4f} "
                f"align_raw:{item['align_loss_raw']:.4f} w:{item['align_weight']:.4f} "
                f"g:{item['align_global_loss']:.4f} l:{item['align_local_loss']:.4f} "
                f"o:{item['align_order_loss']:.4f} "
                f"avg_len:{item['avg_episode_len']:.1f} valid:{item['action_valid_ratio']:.3f} "
                f"lr:{item['lr']:.6e}",
                flush=True,
            )

        if step % args.save_freq == 0 or step == args.steps:
            save_checkpoint(model, optimizer, step, out_dir)

    report = {
        "args": vars(args),
        "device": str(device),
        "dataset_units": len(ds),
        "first_loss": history[0]["loss"],
        "last_loss": history[-1]["loss"],
        "history_tail": history[-min(50, len(history)):],
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to: {out_dir / 'train_report.json'}")


if __name__ == "__main__":
    main()