
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_PATHS = (
    ROOT_DIR / "grassmann.pt",
    ROOT_DIR / "transformer.pt",
)

sys.path.insert(0, str(ROOT_DIR))

from eval_helpers import (
    autocast_context,
    bootstrap_confidence_interval,
    build_eval_loader,
    encode_prompt,
    ensure_dir,
    generate_tokens,
    get_tokenizer,
    load_model_checkpoint,
    load_prompt_suite,
    parse_int_list,
    release_cuda_memory,
    resolve_amp_dtype,
    resolve_device,
    seed_everything,
    token_losses,
    write_csv,
    write_json,
    write_jsonl,
)


def resolve_checkpoint_paths(requested_checkpoints: list[str] | None) -> list[str]:
    if requested_checkpoints:
        return requested_checkpoints

    missing = [str(path) for path in DEFAULT_CHECKPOINT_PATHS if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Default checkpoints were not found. Missing: "
            + ", ".join(missing)
            + ". Pass --checkpoints explicitly to override."
        )
    return [str(path) for path in DEFAULT_CHECKPOINT_PATHS]


def evaluate_heldout(
    model: torch.nn.Module,
    loader,
    device: str,
    amp_dtype: torch.dtype | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    total_loss = 0.0
    total_tokens = 0
    total_windows = 0
    batch_means: list[float] = []

    start = time.perf_counter()
    for batch in tqdm(loader, desc="heldout", unit="batch", dynamic_ncols=True):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast_context(device, amp_dtype):
            logits = model(input_ids)["logits"]
        losses = token_losses(logits.float(), labels)
        total_loss += float(losses.sum().item())
        total_tokens += int(losses.numel())
        total_windows += int(input_ids.size(0))
        batch_means.append(float(losses.mean().item()))

    avg_loss = total_loss / max(1, total_tokens)
    metrics: dict[str, Any] = {
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "num_tokens_scored": total_tokens,
        "num_windows_scored": total_windows,
        "num_batches": len(batch_means),
        "elapsed_sec": time.perf_counter() - start,
    }

    ci = bootstrap_confidence_interval(batch_means, bootstrap_seed, bootstrap_samples)
    if ci is not None:
        metrics["loss_ci95"] = {
            "lower": ci["lower"],
            "upper": ci["upper"],
        }
        metrics["ppl_ci95"] = {
            "lower": math.exp(ci["lower"]),
            "upper": math.exp(ci["upper"]),
        }
    return metrics


def evaluate_context_buckets(
    model: torch.nn.Module,
    loader,
    context_lengths: list[int],
    target_suffix_len: int,
    device: str,
    amp_dtype: torch.dtype | None,
    max_batches: int,
) -> list[dict[str, Any]]:
    stats = {
        context_len: {
            "loss_sum": 0.0,
            "token_count": 0,
            "window_count": 0,
            "batch_means": [],
        }
        for context_len in context_lengths
    }

    iterator = loader
    if max_batches > 0:
        total = min(len(loader), max_batches)
    else:
        total = len(loader)

    progress = tqdm(total=total, desc="context", unit="batch", dynamic_ncols=True)
    for batch_idx, batch in enumerate(iterator):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        seq_len = int(input_ids.size(1))

        for context_len in context_lengths:
            start_idx = max(0, seq_len - target_suffix_len - context_len)
            local_input = input_ids[:, start_idx:]
            local_labels = labels[:, start_idx:]
            with autocast_context(device, amp_dtype):
                logits = model(local_input)["logits"]
            losses = token_losses(logits.float(), local_labels)
            scored = losses[:, -min(target_suffix_len, losses.size(1)) :]

            bucket = stats[context_len]
            bucket["loss_sum"] += float(scored.sum().item())
            bucket["token_count"] += int(scored.numel())
            bucket["window_count"] += int(scored.size(0))
            bucket["batch_means"].append(float(scored.mean().item()))

        progress.update(1)
    progress.close()

    results = []
    for context_len in context_lengths:
        bucket = stats[context_len]
        avg_loss = bucket["loss_sum"] / max(1, bucket["token_count"])
        results.append(
            {
                "context_len": context_len,
                "target_suffix_len": target_suffix_len,
                "loss": avg_loss,
                "perplexity": math.exp(avg_loss),
                "num_tokens_scored": bucket["token_count"],
                "num_windows_scored": bucket["window_count"],
            }
        )
    return results


def summarize_generation_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[(sample["checkpoint_name"], sample["mode"])].append(sample)

    rows: list[dict[str, Any]] = []
    for (checkpoint_name, mode), group in sorted(grouped.items()):
        rows.append(
            {
                "checkpoint_name": checkpoint_name,
                "mode": mode,
                "samples": len(group),
                "avg_prompt_tokens": sum(x["prompt_tokens"] for x in group) / len(group),
                "avg_generated_tokens": sum(x["generated_tokens"] for x in group) / len(group),
                "avg_tokens_per_sec": sum(x["tokens_per_sec"] for x in group) / len(group),
                "avg_elapsed_sec": sum(x["elapsed_sec"] for x in group) / len(group),
            }
        )
    return rows


def _diff_or_none(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def build_polished_metrics_rows(
    heldout_results: list[dict[str, Any]],
    context_results: list[dict[str, Any]],
    generation_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    context_by_checkpoint = {
        item["checkpoint_name"]: item["results"]
        for item in context_results
    }
    generation_by_checkpoint_mode = {
        (item["checkpoint_name"], item["mode"]): item
        for item in generation_summary
    }
    heldout_rank = {
        item["checkpoint_name"]: idx
        for idx, item in enumerate(
            sorted(heldout_results, key=lambda result: result["loss"]),
            start=1,
        )
    }

    comparison_lookup: dict[str, dict[str, Any]] = {}
    if len(heldout_results) == 2:
        first, second = heldout_results
        comparison_lookup[first["checkpoint_name"]] = second
        comparison_lookup[second["checkpoint_name"]] = first

    rows: list[dict[str, Any]] = []
    for item in heldout_results:
        checkpoint_name = item["checkpoint_name"]
        context_rows = context_by_checkpoint.get(checkpoint_name, [])
        best_context = min(context_rows, key=lambda row: row["loss"], default=None)
        worst_context = max(context_rows, key=lambda row: row["loss"], default=None)
        greedy_summary = generation_by_checkpoint_mode.get((checkpoint_name, "greedy"), {})
        stochastic_summary = generation_by_checkpoint_mode.get((checkpoint_name, "stochastic"), {})

        row = {
            "checkpoint_name": checkpoint_name,
            "model_type": item["model_type"],
            "step": item["step"],
            "param_count": item["param_count"],
            "heldout_rank": heldout_rank[checkpoint_name],
            "heldout_loss": item["loss"],
            "heldout_perplexity": item["perplexity"],
            "heldout_loss_ci95_lower": item.get("loss_ci95", {}).get("lower"),
            "heldout_loss_ci95_upper": item.get("loss_ci95", {}).get("upper"),
            "heldout_eval_sec": item["elapsed_sec"],
            "num_tokens_scored": item["num_tokens_scored"],
            "best_context_len": None if best_context is None else best_context["context_len"],
            "best_context_loss": None if best_context is None else best_context["loss"],
            "best_context_perplexity": (
                None if best_context is None else best_context["perplexity"]
            ),
            "worst_context_len": None if worst_context is None else worst_context["context_len"],
            "worst_context_loss": None if worst_context is None else worst_context["loss"],
            "worst_context_perplexity": (
                None if worst_context is None else worst_context["perplexity"]
            ),
            "avg_greedy_generated_tokens": greedy_summary.get("avg_generated_tokens"),
            "avg_greedy_tokens_per_sec": greedy_summary.get("avg_tokens_per_sec"),
            "avg_stochastic_generated_tokens": stochastic_summary.get("avg_generated_tokens"),
            "avg_stochastic_tokens_per_sec": stochastic_summary.get("avg_tokens_per_sec"),
        }

        other = comparison_lookup.get(checkpoint_name)
        if other is not None:
            other_name = other["checkpoint_name"]
            other_greedy = generation_by_checkpoint_mode.get((other_name, "greedy"), {})
            other_stochastic = generation_by_checkpoint_mode.get((other_name, "stochastic"), {})
            row.update(
                {
                    "compared_against": other_name,
                    "loss_delta_vs_other": item["loss"] - other["loss"],
                    "perplexity_delta_vs_other": item["perplexity"] - other["perplexity"],
                    "eval_sec_delta_vs_other": item["elapsed_sec"] - other["elapsed_sec"],
                    "best_context_loss_delta_vs_other": _diff_or_none(
                        row["best_context_loss"],
                        min(
                            context_by_checkpoint.get(other_name, []),
                            key=lambda context_row: context_row["loss"],
                            default={},
                        ).get("loss"),
                    ),
                    "greedy_tokens_per_sec_delta_vs_other": _diff_or_none(
                        row["avg_greedy_tokens_per_sec"],
                        other_greedy.get("avg_tokens_per_sec"),
                    ),
                    "stochastic_tokens_per_sec_delta_vs_other": _diff_or_none(
                        row["avg_stochastic_tokens_per_sec"],
                        other_stochastic.get("avg_tokens_per_sec"),
                    ),
                    "wins_heldout_loss": item["loss"] < other["loss"],
                    "wins_heldout_perplexity": item["perplexity"] < other["perplexity"],
                    "wins_eval_speed": item["elapsed_sec"] < other["elapsed_sec"],
                }
            )

        rows.append(row)
    return rows


def run_generation_suite(
    model: torch.nn.Module,
    checkpoint_meta: dict[str, Any],
    tokenizer,
    prompts: list[dict[str, Any]],
    device: str,
    amp_dtype: torch.dtype | None,
    max_new_tokens: int,
    top_k: int,
    top_p: float,
    temperature: float,
    stochastic_seeds: list[int],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    checkpoint_name = checkpoint_meta["checkpoint_name"]

    progress_total = len(prompts) * (1 + len(stochastic_seeds))
    progress = tqdm(total=progress_total, desc="generation", unit="sample", dynamic_ncols=True)
    for prompt_item in prompts:
        prompt = prompt_item["prompt"]
        encoded_prompt, prompt_meta = encode_prompt(tokenizer, prompt, model.cfg.max_seq_len)

        # Greedy serves as the reproducible baseline.
        greedy = generate_tokens(
            model=model,
            input_ids=encoded_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=top_k,
            top_p=1.0,
            device=device,
            amp_dtype=amp_dtype,
        )
        samples.append(
            {
                "checkpoint_name": checkpoint_name,
                "model_type": checkpoint_meta["model_type"],
                "prompt_id": prompt_item["id"],
                "category": prompt_item["category"],
                "mode": "greedy",
                "seed": None,
                "prompt_tokens": prompt_meta["prompt_tokens"],
                "original_prompt_tokens": prompt_meta["original_prompt_tokens"],
                "truncated_prompt_tokens": prompt_meta["truncated_prompt_tokens"],
                "generated_tokens": greedy["generated_tokens"],
                "elapsed_sec": greedy["elapsed_sec"],
                "tokens_per_sec": greedy["tokens_per_sec"],
                "text": tokenizer.decode(greedy["new_ids"][0].tolist(), skip_special_tokens=True),
            }
        )
        progress.update(1)

        for seed in stochastic_seeds:
            seed_everything(seed)
            generated = generate_tokens(
                model=model,
                input_ids=encoded_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device,
                amp_dtype=amp_dtype,
            )
            samples.append(
                {
                    "checkpoint_name": checkpoint_name,
                    "model_type": checkpoint_meta["model_type"],
                    "prompt_id": prompt_item["id"],
                    "category": prompt_item["category"],
                    "mode": "stochastic",
                    "seed": seed,
                    "prompt_tokens": prompt_meta["prompt_tokens"],
                    "original_prompt_tokens": prompt_meta["original_prompt_tokens"],
                    "truncated_prompt_tokens": prompt_meta["truncated_prompt_tokens"],
                    "generated_tokens": generated["generated_tokens"],
                    "elapsed_sec": generated["elapsed_sec"],
                    "tokens_per_sec": generated["tokens_per_sec"],
                    "text": tokenizer.decode(
                        generated["new_ids"][0].tolist(),
                        skip_special_tokens=True,
                    ),
                }
            )
            progress.update(1)
    progress.close()
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair evaluation suite for v1 checkpoints")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
    )
    parser.add_argument("--token_file", default="data/test_tokens.bin")
    parser.add_argument("--starts_file", default="data/test_starts.npy")
    parser.add_argument(
        "--seq_len",
        type=int,
        default=0,
        help="0 means infer the minimum max_seq_len across checkpoints",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32", "none"],
        default="auto",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=200)
    parser.add_argument("--bootstrap_seed", type=int, default=1234)
    parser.add_argument("--context_lengths", default="32,64,128,256,512,1024")
    parser.add_argument("--target_suffix_len", type=int, default=128)
    parser.add_argument(
        "--context_max_batches",
        type=int,
        default=0,
        help="0 evaluates all batches; use a small positive value for a faster sweep",
    )
    parser.add_argument("--prompt_suite", default="v1/eval_prompts.jsonl")
    parser.add_argument("--generation_max_prompts", type=int, default=0)
    parser.add_argument("--generation_max_new_tokens", type=int, default=64)
    parser.add_argument("--generation_temperature", type=float, default=0.8)
    parser.add_argument("--generation_top_k", type=int, default=50)
    parser.add_argument("--generation_top_p", type=float, default=0.95)
    parser.add_argument("--generation_seeds", default="11,23,37")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--output_dir", default="v1/eval_runs/latest")
    args = parser.parse_args()

    checkpoint_paths = resolve_checkpoint_paths(args.checkpoints)
    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(device, args.dtype)
    tokenizer = get_tokenizer()
    out_dir = ensure_dir(args.output_dir)

    checkpoint_metas: list[dict[str, Any]] = []
    max_seq_lens: list[int] = []
    for checkpoint_path in checkpoint_paths:
        model, meta = load_model_checkpoint(checkpoint_path, device)
        checkpoint_metas.append(meta)
        max_seq_lens.append(int(model.cfg.max_seq_len))
        del model
        release_cuda_memory()

    inferred_seq_len = min(max_seq_lens)
    seq_len = inferred_seq_len if args.seq_len <= 0 else min(args.seq_len, inferred_seq_len)
    loader = build_eval_loader(
        token_file=args.token_file,
        starts_file=args.starts_file,
        seq_len=seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    prompt_suite = []
    if not args.skip_generation:
        prompt_suite = load_prompt_suite(args.prompt_suite)
        if args.generation_max_prompts > 0:
            prompt_suite = prompt_suite[: args.generation_max_prompts]

    heldout_results: list[dict[str, Any]] = []
    context_results: list[dict[str, Any]] = []
    generation_samples: list[dict[str, Any]] = []

    context_lengths = [length for length in parse_int_list(args.context_lengths) if length > 0]
    stochastic_seeds = parse_int_list(args.generation_seeds)

    for checkpoint_path, meta in zip(checkpoint_paths, checkpoint_metas, strict=True):
        model, _ = load_model_checkpoint(checkpoint_path, device)
        print(
            f"Evaluating {meta['checkpoint_name']} ({meta['model_type']}) "
            f"at step={meta['step']}"
        )
        heldout = evaluate_heldout(
            model=model,
            loader=loader,
            device=device,
            amp_dtype=amp_dtype,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        heldout_results.append({**meta, **heldout})

        context = evaluate_context_buckets(
            model=model,
            loader=loader,
            context_lengths=context_lengths,
            target_suffix_len=args.target_suffix_len,
            device=device,
            amp_dtype=amp_dtype,
            max_batches=args.context_max_batches,
        )
        context_results.append(
            {
                "checkpoint_name": meta["checkpoint_name"],
                "model_type": meta["model_type"],
                "results": context,
            }
        )

        if prompt_suite:
            generation_samples.extend(
                run_generation_suite(
                    model=model,
                    checkpoint_meta=meta,
                    tokenizer=tokenizer,
                    prompts=prompt_suite,
                    device=device,
                    amp_dtype=amp_dtype,
                    max_new_tokens=args.generation_max_new_tokens,
                    top_k=args.generation_top_k,
                    top_p=args.generation_top_p,
                    temperature=args.generation_temperature,
                    stochastic_seeds=stochastic_seeds,
                )
            )

        del model
        release_cuda_memory()

    generation_summary = summarize_generation_samples(generation_samples)
    metrics = {
        "comparison_contract": {
            "tokenizer": "gpt2",
            "token_file": args.token_file,
            "starts_file": args.starts_file,
            "shared_seq_len": seq_len,
            "device": device,
            "dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "none",
            "generation": {
                "max_new_tokens": args.generation_max_new_tokens,
                "temperature": args.generation_temperature,
                "top_k": args.generation_top_k,
                "top_p": args.generation_top_p,
                "stochastic_seeds": stochastic_seeds,
            },
        },
        "checkpoints": checkpoint_metas,
        "heldout": heldout_results,
        "context": context_results,
        "generation_summary": generation_summary,
        "caveats": [
            "Held-out loss/perplexity is the primary quality metric; prompt generation is secondary.",
        ],
    }

    metrics_json = out_dir / "metrics.json"
    metrics_csv = out_dir / "metrics.csv"
    polished_metrics_csv = out_dir / "polished_metrics.csv"
    samples_jsonl = out_dir / "samples.jsonl"
    context_json = out_dir / "context_metrics.json"
    context_csv = out_dir / "context_metrics.csv"
    report_path = out_dir / "report.md"

    write_json(metrics_json, metrics)
    metrics_rows = []
    for item in heldout_results:
        metrics_rows.append(
            {
                "checkpoint_name": item["checkpoint_name"],
                "model_type": item["model_type"],
                "step": item["step"],
                "param_count": item["param_count"],
                "loss": item["loss"],
                "perplexity": item["perplexity"],
                "num_tokens_scored": item["num_tokens_scored"],
                "elapsed_sec": item["elapsed_sec"],
                "loss_ci95_lower": item.get("loss_ci95", {}).get("lower"),
                "loss_ci95_upper": item.get("loss_ci95", {}).get("upper"),
            }
        )
    write_csv(metrics_csv, metrics_rows)
    polished_metrics_rows = build_polished_metrics_rows(
        heldout_results=heldout_results,
        context_results=context_results,
        generation_summary=generation_summary,
    )
    write_csv(polished_metrics_csv, polished_metrics_rows)
    write_jsonl(samples_jsonl, generation_samples)

    flattened_context_rows = []
    for checkpoint in context_results:
        for row in checkpoint["results"]:
            flattened_context_rows.append(
                {
                    "checkpoint_name": checkpoint["checkpoint_name"],
                    "model_type": checkpoint["model_type"],
                    **row,
                }
            )
    write_json(context_json, context_results)
    write_csv(context_csv, flattened_context_rows)

    report_text = build_report(metrics)
    report_path.write_text(report_text)

    output_bundle = {
        "metrics_json": str(metrics_json),
        "metrics_csv": str(metrics_csv),
        "polished_metrics_csv": str(polished_metrics_csv),
        "context_json": str(context_json),
        "context_csv": str(context_csv),
        "samples_jsonl": str(samples_jsonl),
    }
    print(json.dumps(output_bundle, indent=2))


if __name__ == "__main__":
    main()
