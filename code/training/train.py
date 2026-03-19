"""
train both models and save the runs to runs/v1 and runs/v2

uv run python training/train.py \
    --model transformer \
    --train_tokens data/train_tokens_10b.bin \
    --train_starts data/train_starts_10b.npy \
    --val_tokens data/val_tokens_10b.bin \
    --val_starts data/val_starts_10b.npy \
    --out_dir runs/v1 \
    --batch_size 32 \
    --total_tokens 10000000000 \
    --warmup_frac 0.005 \
    --eval_interval 100   && \
uv run python training/train.py \
    --model grassmann \
    --train_tokens data/train_tokens_10b.bin \
    --train_starts data/train_starts_10b.npy \
    --val_tokens data/val_tokens_10b.bin \
    --val_starts data/val_starts_10b.npy \
    --out_dir runs/v2 \
    --batch_size 32 \
    --total_tokens 10000000000 \
    --warmup_frac 0.005 \
    --eval_interval 100

"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing.data import FixedOrderTokenDataset
from training.checkpoints import serialize_config
from .grassman import (
    GrassmannConfig,
    GrassmannLM,
    TransformerConfig,
    TransformerLM,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr_frac: float = 0.1,
) -> float:
    """warmup then cosine decay to min_lr_frac * base_lr."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_frac + (1.0 - min_lr_frac) * cosine)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    max_batches: int = 100,
) -> float:
    model.eval()
    losses: list[float] = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
        ):
            out = model(input_ids=input_ids, labels=labels)
        losses.append(out["loss"].item())
    model.train()
    return sum(losses) / max(1, len(losses))



def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    total_steps: int,
    grad_accum_steps: int,
    base_lr: float,
    warmup_steps: int,
    device: str,
    out_dir: Path,
    eval_interval: int = 200,
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"

    step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # accumulators reset after each log
    running_loss = 0.0
    n_accum = 0
    running_tokens = 0
    t0 = time.perf_counter()

    pbar = tqdm(total=total_steps, desc="train", unit="step", dynamic_ncols=True)

    with open(log_path, "w") as log_f:
        log_f.write("step,lr,train_loss,val_loss,val_ppl,tok_per_sec\n")

        while step < total_steps:
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                lr = cosine_lr(step, total_steps, warmup_steps, base_lr)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=(device == "cuda"),
                ):
                    out = model(input_ids=input_ids, labels=labels)
                    loss = out["loss"] / grad_accum_steps

                loss.backward()
                running_loss += out["loss"].item() 
                n_accum += 1
                running_tokens += input_ids.numel()

                if (step + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if step % eval_interval == 0:
                    val_loss = evaluate(model, val_loader, device)
                    val_ppl = math.exp(val_loss)
                    elapsed = time.perf_counter() - t0
                    tps = running_tokens / max(elapsed, 1e-9)
                    train_loss = running_loss / max(1, n_accum) 

                    pbar.write(
                        f"step={step:5d} | lr={lr:.2e} | "
                        f"train_loss={train_loss:.4f} | "
                        f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f} | "
                        f"tok/s={tps:,.0f}"
                    )
                    log_f.write(
                        f"{step},{lr:.6e},{train_loss:.6f},"
                        f"{val_loss:.6f},{val_ppl:.4f},{tps:.1f}\n"
                    )
                    log_f.flush()
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": step,
                            "config": serialize_config(model.cfg),
                            "model_type": model.MODEL_TYPE,
                            "checkpoint_version": 2,
                        },
                        out_dir / "checkpoint.pt",
                    )
                    running_loss = 0.0
                    n_accum = 0
                    running_tokens = 0
                    t0 = time.perf_counter()

                pbar.set_postfix(loss=f"{out['loss'].item():.4f}", lr=f"{lr:.2e}")
                pbar.update(1)
                step += 1
                if step >= total_steps:
                    break

    pbar.close()


## cli helpers

def build_model(args: argparse.Namespace) -> torch.nn.Module:
    offsets = tuple(int(x) for x in args.offsets.split(",") if x.strip())

    if args.model == "grassmann":
        cfg = GrassmannConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            r=args.r,
            offsets=offsets,
            dropout=args.dropout,
        )
        return GrassmannLM(cfg)

    cfg = TransformerConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    return TransformerLM(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GrassmannLM or TransformerLM on FineWeb"
    )

    parser.add_argument("--train_tokens")
    parser.add_argument("--train_starts")
    parser.add_argument("--val_tokens")
    parser.add_argument("--val_starts")

    parser.add_argument(
        "--model", choices=["grassmann", "transformer"], default="grassmann"
    )
    parser.add_argument("--d_model", type=int, default=640)
    parser.add_argument("--n_layers", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=2560)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Grassmann-specific
    parser.add_argument("--r", type=int, default=24)
    parser.add_argument("--offsets", default="1,2,4,8,16,32,64")
    # Transformer-specific
    parser.add_argument("--n_heads", type=int, default=10)

    # Training
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--total_tokens",
        type=float,
        default=1e9,
    )
    parser.add_argument("--base_lr", type=float, default=6e-4)
    parser.add_argument(
        "--warmup_frac",
        type=float,
        default=0.02
    )
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--out_dir", default="runs/v1")

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    train_ds = FixedOrderTokenDataset(
        args.train_tokens, args.train_starts, args.seq_len
    )
    val_ds = FixedOrderTokenDataset(args.val_tokens, args.val_starts, args.seq_len)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False, # not needed, as we already have a pre-tokenized fixed order.
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs)


    model = build_model(args)
    n_params = model.num_params()
    eff_batch_tokens = args.batch_size * args.seq_len * args.grad_accum
    total_steps = max(1, int(args.total_tokens / eff_batch_tokens))
    warmup_steps = max(1, int(args.warmup_frac * total_steps))

    print(f"Model            : {args.model}")
    print(f"Params           : {n_params:,}")
    print(f"Eff batch tokens : {eff_batch_tokens:,}")
    print(f"Total tokens     : {int(args.total_tokens):,}")
    print(f"Total steps      : {total_steps:,}")
    print(f"Warmup steps     : {warmup_steps}")
    print(f"Train examples   : {len(train_ds):,}")

    out_dir = Path(args.out_dir) / args.model
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        total_steps=total_steps,
        grad_accum_steps=args.grad_accum,
        base_lr=args.base_lr,
        warmup_steps=warmup_steps,
        device=device,
        out_dir=out_dir,
        eval_interval=args.eval_interval,
    )
    print(f"\nDone, written to {out_dir}")


if __name__ == "__main__":
    main()
