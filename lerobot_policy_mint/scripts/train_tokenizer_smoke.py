#!/usr/bin/env python3
import argparse
import importlib.util
import json
import random
from pathlib import Path

import torch


def load_multiscale_vqvae_class(repo_root: Path):
    mint_utils_path = repo_root / "src" / "lerobot_policy_mint" / "mint_utils.py"
    spec = importlib.util.spec_from_file_location("mint_utils_local", str(mint_utils_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {mint_utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MultiScaleVQVAE


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_fake_batch(batch_size: int, seq_len: int, device: torch.device):
    actions = torch.randn(batch_size, seq_len, 7, device=device)
    # Make gripper channel near {-1, 1} to match downstream discretization.
    actions[..., 6] = torch.where(actions[..., 6] > 0, torch.ones_like(actions[..., 6]), -torch.ones_like(actions[..., 6]))
    text_pool = [
        "pick up the mug",
        "open the drawer",
        "close the cabinet",
        "place the bowl on table",
        "turn on the lamp",
        "stack blocks",
    ]
    texts = [text_pool[i % len(text_pool)] for i in range(batch_size)]
    return actions, texts


def build_model(args, device: torch.device, MultiScaleVQVAE):
    model = MultiScaleVQVAE(
        seq_dim=7,
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
        ch=args.ch,
        patch_nums=(1, 2, 4),
        ch_mult=(2, 4, 8),
        patchwise={"enable": True, "d_embed": 8, "grouped_depth": 2, "norm": "layer"},
        tokenizer_align_enable=args.align,
        tokenizer_align_model_name=args.align_model,
        tokenizer_align_proj_dim=args.align_dim,
        tokenizer_align_temperature=args.align_temp,
        tokenizer_align_weight=args.align_weight,
        tokenizer_align_warmup_steps=args.align_warmup,
        tokenizer_align_max_length=args.align_max_len,
    )
    return model.to(device)


def save_action_encoder_checkpoint(model, path: Path):
    state = model.state_dict()
    drop_prefixes = (
        "text_encoder.",
        "align_action_proj.",
        "align_text_proj.",
    )
    filtered = {k: v for k, v in state.items() if not any(k.startswith(p) for p in drop_prefixes)}
    torch.save({"model_state_dict": filtered}, path)


def validate_action_encoder_load(args, ckpt_path: Path, device: torch.device, MultiScaleVQVAE):
    # Validate encoder usability independent of language encoder availability.
    model_eval = MultiScaleVQVAE(
        seq_dim=7,
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
        ch=args.ch,
        patch_nums=(1, 2, 4),
        ch_mult=(2, 4, 8),
        patchwise={"enable": True, "d_embed": 8, "grouped_depth": 2, "norm": "layer"},
        tokenizer_align_enable=False,
    ).to(device)

    payload = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model_eval.load_state_dict(payload["model_state_dict"], strict=False)

    x = torch.randn(2, args.seq_len, 7, device=device)
    with torch.no_grad():
        latent = model_eval.encode_to_latent(x)
        f_hat, _, _ = model_eval.quantizer(latent, ret_usages=False, ret_fhat_scales=False)
        rec = model_eval.decode_from_latent(f_hat)

    return {
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "latent_shape": list(latent.shape),
        "recon_shape": list(rec.shape),
    }


def main():
    parser = argparse.ArgumentParser(description="Smoke training for MINT tokenizer baseline/align.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--align_model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--align_dim", type=int, default=256)
    parser.add_argument("--align_temp", type=float, default=0.07)
    parser.add_argument("--align_weight", type=float, default=0.1)
    parser.add_argument("--align_warmup", type=int, default=1000)
    parser.add_argument("--align_max_len", type=int, default=64)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--codebook_dim", type=int, default=32)
    parser.add_argument("--ch", type=int, default=48)
    parser.add_argument("--output_dir", type=str, default="outputs/tokenizer_smoke")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MultiScaleVQVAE = load_multiscale_vqvae_class(repo_root)
    model = build_model(args, device, MultiScaleVQVAE)

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    for step in range(1, args.steps + 1):
        actions, texts = make_fake_batch(args.batch_size, args.seq_len, device)

        losses = model.compute_tokenizer_losses(
            actions,
            texts=texts if args.align else None,
            global_step=step,
        )
        loss = losses["loss"]

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        item = {
            "step": step,
            "loss": float(losses["loss"].detach().cpu().item()),
            "recon_loss": float(losses["recon_loss"].detach().cpu().item()),
            "vq_loss": float(losses["vq_loss"].detach().cpu().item()),
            "align_loss": float(losses["align_loss"].detach().cpu().item()),
            "align_weight": float(losses["align_weight"].detach().cpu().item()),
        }
        history.append(item)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                f"step={step:04d} "
                f"loss={item['loss']:.4f} recon={item['recon_loss']:.4f} "
                f"vq={item['vq_loss']:.4f} align={item['align_loss']:.4f} "
                f"w={item['align_weight']:.4f}"
            )

    ckpt_name = "tokenizer_align_smoke.pt" if args.align else "tokenizer_baseline_smoke.pt"
    ckpt_path = output_dir / ckpt_name
    save_action_encoder_checkpoint(model, ckpt_path)

    validation = validate_action_encoder_load(args, ckpt_path, device, MultiScaleVQVAE)

    report = {
        "args": vars(args),
        "device": str(device),
        "first_loss": history[0]["loss"],
        "last_loss": history[-1]["loss"],
        "history_tail": history[-min(10, len(history)):],
        "checkpoint": str(ckpt_path),
        "validation": validation,
    }

    report_name = "report_align.json" if args.align else "report_baseline.json"
    report_path = output_dir / report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to: {report_path}")
    print(f"Saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
