r"""
uv run python benchmarks/bench.py \ 
    --checkpoints grassmann.pt transformer.pt \
    --prompt_lengths 8,16,32,64,128,256,512,1024 \
    --decode_steps 64 \
    --output_dir eval_runs/default_benchmark
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
from tqdm import tqdm

import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.eval_helpers import (
    autocast_context,
    ensure_dir,
    generate_tokens,
    load_model_checkpoint,
    parse_int_list,
    release_cuda_memory,
    resolve_amp_dtype,
    resolve_device,
    seed_everything,
    synchronize,
    write_csv,
    write_json,
)


def synthetic_input(
    batch_size: int,
    prompt_len: int,
    vocab_size: int,
    device: str,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device if device == "cuda" else "cpu")
    generator.manual_seed(seed)
    return torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, prompt_len),
        generator=generator,
        dtype=torch.long,
        device=device,
    )


def reset_peak_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mib(device: str) -> float | None:
    if device != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def measure_prefill(
    model: torch.nn.Module,
    prompt_len: int,
    batch_size: int,
    warmup_iters: int,
    measure_iters: int,
    device: str,
    amp_dtype: torch.dtype | None,
    base_seed: int,
) -> dict[str, Any]:
    input_ids = synthetic_input(
        batch_size=batch_size,
        prompt_len=prompt_len,
        vocab_size=model.cfg.vocab_size,
        device=device,
        seed=base_seed + prompt_len,
    )

    for _ in range(warmup_iters):
        with torch.no_grad(), autocast_context(device, amp_dtype):
            _ = model(input_ids)["logits"]
        synchronize(device)

    timings_ms: list[float] = []
    peak_values: list[float] = []
    for _ in range(measure_iters):
        reset_peak_memory(device)
        synchronize(device)
        start = time.perf_counter()
        with torch.no_grad(), autocast_context(device, amp_dtype):
            _ = model(input_ids)["logits"]
        synchronize(device)
        timings_ms.append((time.perf_counter() - start) * 1000.0)
        peak = peak_memory_mib(device)
        if peak is not None:
            peak_values.append(peak)

    return {
        "benchmark": "prefill",
        "prompt_len": prompt_len,
        "batch_size": batch_size,
        "avg_elapsed_ms": statistics.mean(timings_ms),
        "std_elapsed_ms": statistics.pstdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        "peak_memory_mib": max(peak_values) if peak_values else None,
    }


def measure_uncached_decode(
    model: torch.nn.Module,
    prompt_len: int,
    decode_steps: int,
    batch_size: int,
    warmup_iters: int,
    measure_iters: int,
    device: str,
    amp_dtype: torch.dtype | None,
    base_seed: int,
) -> dict[str, Any]:
    input_ids = synthetic_input(
        batch_size=batch_size,
        prompt_len=prompt_len,
        vocab_size=model.cfg.vocab_size,
        device=device,
        seed=base_seed + 10_000 + prompt_len,
    )

    for _ in range(warmup_iters):
        _ = generate_tokens(
            model=model,
            input_ids=input_ids,
            max_new_tokens=decode_steps,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            device=device,
            amp_dtype=amp_dtype,
            decode_mode="full",
        )

    elapsed_ms: list[float] = []
    tokens_per_sec: list[float] = []
    peak_values: list[float] = []
    for _ in range(measure_iters):
        reset_peak_memory(device)
        result = generate_tokens(
            model=model,
            input_ids=input_ids,
            max_new_tokens=decode_steps,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            device=device,
            amp_dtype=amp_dtype,
            decode_mode="full",
        )
        elapsed_ms.append(result["elapsed_sec"] * 1000.0)
        tokens_per_sec.append(result["tokens_per_sec"])
        peak = peak_memory_mib(device)
        if peak is not None:
            peak_values.append(peak)

    avg_elapsed_ms = statistics.mean(elapsed_ms)
    avg_tps = statistics.mean(tokens_per_sec)
    return {
        "benchmark": "uncached_decode",
        "prompt_len": prompt_len,
        "decode_steps": decode_steps,
        "batch_size": batch_size,
        "avg_elapsed_ms": avg_elapsed_ms,
        "std_elapsed_ms": statistics.pstdev(elapsed_ms) if len(elapsed_ms) > 1 else 0.0,
        "avg_tokens_per_sec": avg_tps,
        "avg_ms_per_token": avg_elapsed_ms / max(1, decode_steps * batch_size),
        "peak_memory_mib": max(peak_values) if peak_values else None,
    }


def plot_runtime_scaling(rows: list[dict[str, Any]], output_path: Path) -> Path | None: 
    if not rows:
        return None

    benchmarks = ["prefill", "uncached_decode"]
    titles = {
        "prefill": "Prefill Runtime",
        "uncached_decode": "Uncached Decode Throughput",
    }
    ylabels = {
        "prefill": "Elapsed time (ms, log scale)",
        "uncached_decode": "Tokens / sec (log scale)",
    }
    value_keys = {
        "prefill": "avg_elapsed_ms",
        "uncached_decode": "avg_tokens_per_sec",
    }
    colors = {
        "grassmann": "#0f766e",
        "transformer": "#b45309",
    }
    markers = {
        "grassmann": "o",
        "transformer": "s",
    }
    labels = {
        "grassmann": "Grassmann",
        "transformer": "Transformer",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    axes = list(axes.ravel())

    for ax, benchmark in zip(axes, benchmarks, strict=True):
        benchmark_rows = [row for row in rows if row["benchmark"] == benchmark]
        model_types = sorted({row["model_type"] for row in benchmark_rows})
        xticks = sorted({int(row["prompt_len"]) for row in benchmark_rows})

        for model_type in model_types:
            model_rows = sorted(
                (row for row in benchmark_rows if row["model_type"] == model_type),
                key=lambda row: row["prompt_len"],
            )
            ax.plot(
                [row["prompt_len"] for row in model_rows],
                [row[value_keys[benchmark]] for row in model_rows],
                marker=markers.get(model_type, "o"),
                linewidth=2.5,
                color=colors.get(model_type, "#334155"),
                label=labels.get(model_type, model_type.title()),
            )

        if xticks:
            ax.set_xscale("log", base=2)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(tick) for tick in xticks])
        ax.set_yscale("log")
        ax.set_xlabel("Prompt tokens")
        ax.set_ylabel(ylabels[benchmark])
        ax.set_title(titles[benchmark], fontsize=11)
        ax.grid(True, which="major", axis="both", alpha=0.25)
        ax.legend(loc="best")

    fig.suptitle("Runtime Benchmark Scaling", fontsize=14, fontweight="bold")
    fig.tight_layout()

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--prompt_lengths", default="32,128,512,1024")
    parser.add_argument("--decode_steps", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup_iters", type=int, default=3)
    parser.add_argument("--measure_iters", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32", "none"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", default="eval_runs/default_benchmark")
    args = parser.parse_args()

    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(device, args.dtype)
    prompt_lengths = [length for length in parse_int_list(args.prompt_lengths) if length > 0]
    out_dir = ensure_dir(args.output_dir)

    rows: list[dict[str, Any]] = []
    progress = tqdm(
        total=len(args.checkpoints) * len(prompt_lengths) * 2,
        desc="bench",
        unit="run",
        dynamic_ncols=True,
    )

    for checkpoint_path in args.checkpoints:
        model, meta = load_model_checkpoint(checkpoint_path, device)
        seed_everything(args.seed)

        for prompt_len in prompt_lengths:
            capped_prompt_len = min(prompt_len, int(model.cfg.max_seq_len))

            prefill = measure_prefill(
                model=model,
                prompt_len=capped_prompt_len,
                batch_size=args.batch_size,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                device=device,
                amp_dtype=amp_dtype,
                base_seed=args.seed,
            )
            rows.append({**meta, **prefill, "requested_prompt_len": prompt_len})
            progress.update(1)

            decode = measure_uncached_decode(
                model=model,
                prompt_len=capped_prompt_len,
                decode_steps=args.decode_steps,
                batch_size=args.batch_size,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                device=device,
                amp_dtype=amp_dtype,
                base_seed=args.seed,
            )
            rows.append({**meta, **decode, "requested_prompt_len": prompt_len})
            progress.update(1)

        del model
        release_cuda_memory()

    progress.close()

    runtime = {
        "comparison_contract": {
            "device": device,
            "dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "none",
            "prompt_lengths": prompt_lengths,
            "decode_steps": args.decode_steps,
            "warmup_iters": args.warmup_iters,
            "measure_iters": args.measure_iters,
            "batch_size": args.batch_size,
        },
        "rows": rows,
        "caveats": [
            "created random token prompts",
        ],
    }

    runtime_json = out_dir / "runtime.json"
    runtime_csv = out_dir / "runtime.csv"
    plot_path = out_dir / "runtime_scaling.png"
    write_json(runtime_json, runtime)
    write_csv(runtime_csv, rows)
    saved_plot = plot_runtime_scaling(rows, plot_path)
    print(
        json.dumps(
            {
                "runtime_json": str(runtime_json),
                "runtime_csv": str(runtime_csv),
                "plot_png": str(saved_plot) if saved_plot is not None else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
