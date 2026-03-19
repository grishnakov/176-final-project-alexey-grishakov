"""
uv run python benchmarks/compute_profile.py \
    --prompt_lengths 8,16,32,64,128,256,512,1024 \
    --warmup_iters 2 \
    --measure_iters 3 \
    --output_dir eval_runs/profiled_compute
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib
import torch
from torch.profiler import ProfilerActivity, profile
from tqdm import tqdm

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


def profiler_activities(device: str) -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def profiler_context(device: str):
    return profile(
        activities=profiler_activities(device),
        record_shapes=False,
        with_flops=True,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        acc_events=False,
    )


def summarize_flops(prof) -> tuple[float, list[dict[str, Any]]]:
    events = prof.key_averages()
    total_flops = 0.0
    op_rows: list[dict[str, Any]] = []
    for event in events:
        flops = float(getattr(event, "flops", 0.0) or 0.0)
        total_flops += flops
        if flops <= 0.0:
            continue
        op_rows.append(
            {
                "op_name": event.key,
                "flops": flops,
                "gflops": flops / 1e9,
                "cpu_time_total_us": float(getattr(event, "cpu_time_total", 0.0) or 0.0),
                "self_cpu_time_total_us": float(getattr(event, "self_cpu_time_total", 0.0) or 0.0),
            }
        )

    op_rows.sort(key=lambda row: row["flops"], reverse=True)
    return total_flops, op_rows


def measure_profiled_prefill(
    model: torch.nn.Module,
    checkpoint_meta: dict[str, Any],
    prompt_len: int,
    batch_size: int,
    warmup_iters: int,
    measure_iters: int,
    device: str,
    amp_dtype: torch.dtype | None,
    base_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_ids = synthetic_input(
        batch_size=batch_size,
        prompt_len=prompt_len,
        vocab_size=model.cfg.vocab_size,
        device=device,
        seed=base_seed + prompt_len,
    )

    for _ in range(warmup_iters):
        with torch.inference_mode(), autocast_context(device, amp_dtype):
            _ = model(input_ids)["logits"]
        synchronize(device)

    iter_flops: list[float] = []
    iter_elapsed_ms: list[float] = []
    combined_top_ops: dict[str, dict[str, Any]] = {}

    for iter_idx in range(measure_iters):
        synchronize(device)
        with profiler_context(device) as prof:
            start_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
            end_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

            if start_event is not None:
                start_event.record()
            with torch.inference_mode(), autocast_context(device, amp_dtype):
                _ = model(input_ids)["logits"]
            if end_event is not None:
                end_event.record()

        synchronize(device)

        if start_event is not None and end_event is not None:
            elapsed_ms = float(start_event.elapsed_time(end_event))
        else:
            elapsed_ms = sum(
                float(getattr(event, "cpu_time_total", 0.0) or 0.0) for event in prof.key_averages()
            ) / 1000.0

        total_flops, op_rows = summarize_flops(prof)
        iter_flops.append(total_flops)
        iter_elapsed_ms.append(elapsed_ms)

        for rank, row in enumerate(op_rows[:10], start=1):
            key = row["op_name"]
            slot = combined_top_ops.setdefault(
                key,
                {"op_name": key, "flops_values": [], "best_rank": rank},
            )
            slot["flops_values"].append(row["flops"])
            slot["best_rank"] = min(slot["best_rank"], rank)

    top_ops: list[dict[str, Any]] = []
    for slot in combined_top_ops.values():
        mean_flops = statistics.mean(slot["flops_values"])
        top_ops.append(
            {
                "op_name": slot["op_name"],
                "avg_flops": mean_flops,
                "avg_gflops": mean_flops / 1e9,
                "best_rank": slot["best_rank"],
            }
        )
    top_ops.sort(key=lambda row: row["avg_flops"], reverse=True)

    avg_flops = statistics.mean(iter_flops)
    std_flops = statistics.pstdev(iter_flops) if len(iter_flops) > 1 else 0.0
    avg_elapsed_ms = statistics.mean(iter_elapsed_ms)

    row = {
        **checkpoint_meta,
        "benchmark": "profiled_prefill",
        "prompt_len": prompt_len,
        "batch_size": batch_size,
        "measure_iters": measure_iters,
        "avg_profiled_flops": avg_flops,
        "std_profiled_flops": std_flops,
        "avg_profiled_gflops": avg_flops / 1e9,
        "avg_elapsed_ms": avg_elapsed_ms,
        "profiled_flops_per_token": avg_flops / max(1, batch_size * prompt_len),
    }
    return row, top_ops


def measure_profiled_decode(
    model: torch.nn.Module,
    checkpoint_meta: dict[str, Any],
    prompt_len: int,
    decode_steps: int,
    batch_size: int,
    warmup_iters: int,
    measure_iters: int,
    device: str,
    amp_dtype: torch.dtype | None,
    base_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
            decode_mode="auto",
        )
        synchronize(device)

    iter_flops: list[float] = []
    iter_elapsed_ms: list[float] = []
    combined_top_ops: dict[str, dict[str, Any]] = {}

    for iter_idx in range(measure_iters):
        synchronize(device)
        with profiler_context(device) as prof:
            start_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
            end_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

            if start_event is not None:
                start_event.record()
            _ = generate_tokens(
                model=model,
                input_ids=input_ids,
                max_new_tokens=decode_steps,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                device=device,
                amp_dtype=amp_dtype,
                decode_mode="auto",
            )
            if end_event is not None:
                end_event.record()

        synchronize(device)

        if start_event is not None and end_event is not None:
            elapsed_ms = float(start_event.elapsed_time(end_event))
        else:
            elapsed_ms = sum(
                float(getattr(event, "cpu_time_total", 0.0) or 0.0) for event in prof.key_averages()
            ) / 1000.0

        total_flops, op_rows = summarize_flops(prof)
        iter_flops.append(total_flops)
        iter_elapsed_ms.append(elapsed_ms)

        for rank, row in enumerate(op_rows[:10], start=1):
            key = row["op_name"]
            slot = combined_top_ops.setdefault(
                key,
                {"op_name": key, "flops_values": [], "best_rank": rank},
            )
            slot["flops_values"].append(row["flops"])
            slot["best_rank"] = min(slot["best_rank"], rank)

    top_ops: list[dict[str, Any]] = []
    for slot in combined_top_ops.values():
        mean_flops = statistics.mean(slot["flops_values"])
        top_ops.append(
            {
                "op_name": slot["op_name"],
                "avg_flops": mean_flops,
                "avg_gflops": mean_flops / 1e9,
                "best_rank": slot["best_rank"],
            }
        )
    top_ops.sort(key=lambda row: row["avg_flops"], reverse=True)

    avg_flops = statistics.mean(iter_flops)
    std_flops = statistics.pstdev(iter_flops) if len(iter_flops) > 1 else 0.0
    avg_elapsed_ms = statistics.mean(iter_elapsed_ms)

    row = {
        **checkpoint_meta,
        "benchmark": "profiled_decode",
        "decode_mode": "auto",
        "prompt_len": prompt_len,
        "decode_steps": decode_steps,
        "batch_size": batch_size,
        "measure_iters": measure_iters,
        "avg_profiled_flops": avg_flops,
        "std_profiled_flops": std_flops,
        "avg_profiled_gflops": avg_flops / 1e9,
        "avg_elapsed_ms": avg_elapsed_ms,
        "profiled_flops_per_generated_token": avg_flops / max(1, batch_size * decode_steps),
    }
    return row, top_ops


def plot_profiled_scaling(rows: list[dict[str, Any]], output_path: Path) -> None:
    benchmarks = ["profiled_prefill", "profiled_decode"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    axes = list(axes.ravel())

    titles = {
        "profiled_prefill": "Prefill",
        "profiled_decode": "Decode (`decode_mode=auto`)",
    }
    ylabels = {
        "profiled_prefill": "Profiled prefill compute (GFLOPs)",
        "profiled_decode": "Profiled decode compute (GFLOPs)",
    }

    for ax, benchmark in zip(axes, benchmarks, strict=True):
        benchmark_rows = [row for row in rows if row["benchmark"] == benchmark]
        grassmann_rows = [row for row in benchmark_rows if row["model_type"] == "grassmann"]
        transformer_rows = [row for row in benchmark_rows if row["model_type"] == "transformer"]

        ax.plot(
            [row["prompt_len"] for row in grassmann_rows],
            [row["avg_profiled_gflops"] for row in grassmann_rows],
            marker="o",
            linewidth=2.5,
            color="#0f766e",
            label="Grassmann",
        )
        ax.plot(
            [row["prompt_len"] for row in transformer_rows],
            [row["avg_profiled_gflops"] for row in transformer_rows],
            marker="s",
            linewidth=2.5,
            color="#b45309",
            label="Transformer",
        )

        xticks = sorted({row["prompt_len"] for row in benchmark_rows})
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(tick) for tick in xticks])
        ax.set_xlabel("Prompt tokens")
        ax.set_ylabel(ylabels[benchmark])
        ax.grid(True, which="major", axis="both", alpha=0.25)
        ax.set_title(titles[benchmark], fontsize=11)
        ax.legend(loc="upper left")

    fig.suptitle("Profiled Compute Scaling", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=["grassmann.pt", "transformer.pt"])
    parser.add_argument("--prompt_lengths", default="8,16,32,64,128,256,512,1024")
    parser.add_argument(
        "--decode_prompt_lengths",
        default="8,16,32,64,128,256",
    )
    parser.add_argument("--decode_steps", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup_iters", type=int, default=2)
    parser.add_argument("--measure_iters", type=int, default=3)
    parser.add_argument("--decode_warmup_iters", type=int, default=1)
    parser.add_argument("--decode_measure_iters", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32", "none"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", default="eval_runs/profiled_compute")
    args = parser.parse_args()

    prompt_lengths = [length for length in parse_int_list(args.prompt_lengths) if length > 0]
    
    decode_prompt_lengths = [
        length for length in parse_int_list(args.decode_prompt_lengths) if length > 0
    ]
    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(device, args.dtype)
    out_dir = ensure_dir(args.output_dir)

    rows: list[dict[str, Any]] = []
    op_rows: list[dict[str, Any]] = []
    total_runs = len(args.checkpoints) * (
        len(prompt_lengths) + len(decode_prompt_lengths)
    )
    progress = tqdm(total=total_runs, desc="profile", unit="run", dynamic_ncols=True)

    for checkpoint_path in args.checkpoints:
        model, meta = load_model_checkpoint(checkpoint_path, device)
        seed_everything(args.seed)

        for prompt_len in prompt_lengths:
            capped_prompt_len = min(prompt_len, int(model.cfg.max_seq_len))
            prefill_row, prefill_top_ops = measure_profiled_prefill(
                model=model,
                checkpoint_meta=meta,
                prompt_len=capped_prompt_len,
                batch_size=args.batch_size,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                device=device,
                amp_dtype=amp_dtype,
                base_seed=args.seed,
            )
            prefill_row["requested_prompt_len"] = prompt_len
            rows.append(prefill_row)

            for op_row in prefill_top_ops:
                op_rows.append(
                    {
                        "checkpoint_name": meta["checkpoint_name"],
                        "model_type": meta["model_type"],
                        "benchmark": "profiled_prefill",
                        "prompt_len": capped_prompt_len,
                        **op_row,
                    }
                )
            progress.update(1)

        for prompt_len in decode_prompt_lengths:
            capped_prompt_len = min(prompt_len, int(model.cfg.max_seq_len))
            decode_row, decode_top_ops = measure_profiled_decode(
                model=model,
                checkpoint_meta=meta,
                prompt_len=capped_prompt_len,
                decode_steps=args.decode_steps,
                batch_size=args.batch_size,
                warmup_iters=args.decode_warmup_iters,
                measure_iters=args.decode_measure_iters,
                device=device,
                amp_dtype=amp_dtype,
                base_seed=args.seed,
            )
            decode_row["requested_prompt_len"] = prompt_len
            rows.append(decode_row)

            for op_row in decode_top_ops:
                op_rows.append(
                    {
                        "checkpoint_name": meta["checkpoint_name"],
                        "model_type": meta["model_type"],
                        "benchmark": "profiled_decode",
                        "prompt_len": capped_prompt_len,
                        **op_row,
                    }
                )
            progress.update(1)

        del model
        release_cuda_memory()

    progress.close()

    grouped_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        "profiled_prefill": {"grassmann": [], "transformer": []},
        "profiled_decode": {"grassmann": [], "transformer": []},
    }
    for row in rows:
        grouped_rows[row["benchmark"]][row["model_type"]].append(row)
    for benchmark_rows in grouped_rows.values():
        for model_rows in benchmark_rows.values():
            model_rows.sort(key=lambda row: row["prompt_len"])
            if not model_rows:
                continue
            baseline = model_rows[0]["avg_profiled_flops"]
            for row in model_rows:
                row["relative_scale_vs_first"] = (
                    row["avg_profiled_flops"] / baseline if baseline > 0.0 else 0.0
                )

    payload = {
        "comparison_contract": {
            "checkpoints": args.checkpoints,
            "device": device,
            "dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "none",
            "prefill_prompt_lengths": prompt_lengths,
            "decode_prompt_lengths": decode_prompt_lengths,
            "decode_steps": args.decode_steps,
            "batch_size": args.batch_size,
            "warmup_iters": args.warmup_iters,
            "measure_iters": args.measure_iters,
            "decode_warmup_iters": args.decode_warmup_iters,
            "decode_measure_iters": args.decode_measure_iters,
            "seed": args.seed,
        },
        "rows": rows,
        "grouped_rows": grouped_rows,
        "top_ops": op_rows,

    }

    json_path = out_dir / "profiled_compute.json"
    csv_path = out_dir / "profiled_compute.csv"
    ops_csv_path = out_dir / "profiled_top_ops.csv"
    plot_path = out_dir / "profiled_compute.png"
    write_json(json_path, payload)
    write_csv(csv_path, rows)
    write_csv(ops_csv_path, op_rows)
    plot_profiled_scaling(rows, plot_path)

    print(
        json.dumps(
            {
                "profiled_compute_json": str(json_path),
                "profiled_compute_csv": str(csv_path),
                "profiled_top_ops_csv": str(ops_csv_path),
                "plot_png": str(plot_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
