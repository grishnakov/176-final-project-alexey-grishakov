from __future__ import annotations

import csv
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_processing.data import FixedOrderTokenDataset
from training.checkpoints import infer_model_type, load_model_from_checkpoint, serialize_config
from training.grassman import (
    GrassmannConfig,
    TransformerConfig,
)


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cuda_supports_bfloat16() -> bool:
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(checker and checker())


def resolve_amp_dtype(device: str, requested: str = "auto") -> torch.dtype | None:
    if device != "cuda":
        return None
    if requested == "auto":
        return torch.bfloat16 if _cuda_supports_bfloat16() else torch.float16
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "none": None,
    }
    if requested not in mapping:
        raise ValueError(f"Unsupported dtype {requested!r}")
    return mapping[requested]


def autocast_context(device: str, amp_dtype: torch.dtype | None):
    if device != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def checkpoint_label(path: str | Path) -> str:
    path = Path(path)
    stem = path.stem
    if stem == "checkpoint" and path.parent.name:
        return path.parent.name
    return stem


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt_path = Path(checkpoint_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "config" not in ckpt:
        model_type = infer_model_type(ckpt)
        default_cfg = GrassmannConfig() if model_type == "grassmann" else TransformerConfig()
        ckpt["config"] = serialize_config(default_cfg)
    model, model_type = load_model_from_checkpoint(ckpt)
    model.to(device)
    model.eval()

    metadata = {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_name": checkpoint_label(ckpt_path),
        "model_type": model_type,
        "step": ckpt.get("step"),
        "param_count": int(model.num_params()),
        "config": serialize_config(model.cfg),
    }
    return model, metadata


def get_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.model_max_length = int(1e30)
    return tok


def build_eval_loader(
    token_file: str,
    starts_file: str,
    seq_len: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> DataLoader:
    dataset = FixedOrderTokenDataset(token_file, starts_file, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )


def token_losses(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    )
    return losses.view_as(labels)


def encode_prompt(
    tokenizer,
    prompt: str,
    max_seq_len: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    original_len = len(prompt_ids)
    if original_len > max_seq_len:
        prompt_ids = prompt_ids[-max_seq_len:]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    return input_ids, {
        "prompt_tokens": len(prompt_ids),
        "original_prompt_tokens": original_len,
        "truncated_prompt_tokens": max(0, original_len - len(prompt_ids)),
    }


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def generate_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: str,
    amp_dtype: torch.dtype | None = None,
    decode_mode: str = "auto",
) -> dict[str, Any]:
    if decode_mode not in {"auto", "full", "stateful"}:
        raise ValueError(f"Unsupported decode_mode {decode_mode!r}")

    ids = input_ids.to(device)
    if ids.size(1) == 0:
        raise ValueError("generate_tokens requires at least one prompt token")
    max_seq_len = model.cfg.max_seq_len

    can_stateful_decode = bool(
        getattr(model, "supports_stateful_decode", lambda: False)()
    )
    use_stateful = decode_mode != "full" and can_stateful_decode
    if decode_mode == "stateful" and not use_stateful:
        raise ValueError("Requested stateful decoding for a model without cache support")

    state: dict[str, Any] | None = None
    next_logits: torch.Tensor | None = None

    synchronize(device)
    start = time.perf_counter()

    if use_stateful:
        prompt_context = ids[:, -max_seq_len:]
        with autocast_context(device, amp_dtype):
            prefill = model.prefill(prompt_context)
        state = prefill["state"]
        next_logits = prefill["logits"][:, -1, :]

    for _ in range(max_new_tokens):
        if use_stateful:
            assert state is not None
            if next_logits is None:
                if int(state["next_position"]) < max_seq_len:
                    with autocast_context(device, amp_dtype):
                        step = model.decode_step(ids[:, -1:], state)
                    state = step["state"]
                    logits = step["logits"][:, -1, :]
                else:
                    context = ids[:, -max_seq_len:]
                    with autocast_context(device, amp_dtype):
                        prefill = model.prefill(context)
                    state = prefill["state"]
                    logits = prefill["logits"][:, -1, :]
            else:
                logits = next_logits
                next_logits = None
        else:
            context = ids[:, -max_seq_len:]
            with autocast_context(device, amp_dtype):
                logits = model(context)["logits"][:, -1, :]

        if temperature <= 0.0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            if temperature != 1.0:
                logits = logits / temperature

            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                kth = logits.topk(top_k_val, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                sorted_probs = sorted_logits.softmax(dim=-1)
                remove = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs = logits.softmax(dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        ids = torch.cat([ids, next_id], dim=1)

    synchronize(device)
    elapsed = time.perf_counter() - start
    new_ids = ids[:, input_ids.size(1):]
    generated_tokens = int(new_ids.numel())
    return {
        "new_ids": new_ids.cpu(),
        "elapsed_sec": elapsed,
        "generated_tokens": generated_tokens,
        "tokens_per_sec": (
            generated_tokens / elapsed if elapsed > 0.0 and generated_tokens > 0 else 0.0
        ),
    }


def load_prompt_suite(path: str | Path) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    with open(path) as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "id" not in item or "prompt" not in item or "category" not in item:
                raise ValueError(
                    f"Prompt file {path} line {line_number} must contain id/category/prompt"
                )
            prompts.append(item)
    return prompts


def bootstrap_confidence_interval(
    samples: list[float],
    seed: int,
    num_bootstrap: int,
    alpha: float = 0.05,
) -> dict[str, float] | None:
    if len(samples) < 2 or num_bootstrap <= 0:
        return None
    values = np.asarray(samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(num_bootstrap, dtype=np.float64)
    for i in range(num_bootstrap):
        draw = rng.choice(values, size=len(values), replace=True)
        boot_means[i] = float(draw.mean())
    lower = float(np.quantile(boot_means, alpha / 2))
    upper = float(np.quantile(boot_means, 1.0 - alpha / 2))
    return {
        "mean": float(values.mean()),
        "lower": lower,
        "upper": upper,
    }


def ensure_dir(path: str | Path) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_json(path: str | Path, payload: Any) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def release_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
