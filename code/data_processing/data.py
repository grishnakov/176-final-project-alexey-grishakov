"""Data pipeline.

two steps for pre-processing.

1. pre-tokenize fineweb once into a compact binary file:
       uv run python data_processing/data.py tokenize --split <train/validation/test> --max_tokens 1000000000 --out data/<train/validation/test>_tokens.bin # for test use --test_frac 0.01

2. build a fixed, shuffled list of example start indices:
       uv run python data_processing/data.py index --token_file data/<train/validation/test>_tokens.bin --out data/<train/validation/test>_starts.npy --num_tokens 1000000000 
"""
from __future__ import annotations

import argparse
import contextlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


DEFAULT_FINEWEB_DIR = "/home/user/Documents/datasets/fineweb/data/sample/100BT"
DEFAULT_NUM_WORKERS = 1
TEXT_BATCH_SIZE = 1
GPT2_EOS_TOKEN_ID = 50256
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Tokenise



def _split_parquet_files(
    data_dir: str,
    split: str,
    validation_frac: float,
    test_frac: float = 0.0,
) -> list[Path]:
    """
    Split parquets (downloaded from huggingface, about 100GB (100BT)) into necessary shards.
    """
    split = split.lower()
    if split == "val":
        split = "validation"
    if split in {"test", "testing"}:
        split = "test"

    parquet_files = sorted(Path(data_dir).glob("*.parquet"))
    

    num_val_files = max(1, int(round(len(parquet_files) * validation_frac)))
    num_test_files = (
        max(1, int(round(len(parquet_files) * test_frac))) if test_frac > 0.0 else 0
    )

    train_end = len(parquet_files) - num_val_files - num_test_files
    val_end = len(parquet_files) - num_test_files
    if split == "train":
        selected = parquet_files[:train_end]
    elif split == "validation":
        selected = parquet_files[train_end:val_end]
    else:
        selected = parquet_files[val_end:]
    return selected


_WORKER_TOKENIZER = None


def _iter_local_fineweb_text_batches(
    data_dir: str,
    split: str,
    validation_frac: float,
    test_frac: float,
    num_workers: int,
):
    """get batches of text from local parquet shards."""
    parquet_files = _split_parquet_files(
        data_dir=data_dir,
        split=split,
        validation_frac=validation_frac,
        test_frac=test_frac,
    )
    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for row_group_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(
                row_group_idx,
                columns=["text"],
                use_threads=(num_workers > 1),
            )
            texts = [text for text in table.column("text").to_pylist() if text]
            for start in range(0, len(texts), TEXT_BATCH_SIZE):
                batch = texts[start : start + TEXT_BATCH_SIZE]
                if batch:
                    yield batch



def _tokenize_text_batch(texts: list[str]) -> list[list[int]]:
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e30) # no limit on the length of the text
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    eos_token_id = tokenizer.eos_token_id
    return [ids + [eos_token_id] for ids in encoded]


def _write_token_batch(
    token_batch: list[list[int]],
    f,
    progress: tqdm,
    total_tokens: int,
    n_docs: int,
    max_docs: int | None,
    max_tokens: int | None,
) -> tuple[int, int, bool]:
    """Write one tokenized batch and chgeck if we reached token limit."""
    for ids in token_batch:
        original_len = len(ids)

        if max_tokens is not None:
            remaining = max_tokens - total_tokens
            if remaining <= 0:
                return total_tokens, n_docs, True
            if original_len > remaining:
                ids = ids[:remaining]

        if not ids:
            return total_tokens, n_docs, True

        np.asarray(ids, dtype=np.uint16).tofile(f)  # GPT-2 vocab < 65536
        total_tokens += len(ids)
        progress.update(len(ids))

        if len(ids) == original_len:
            n_docs += 1
            progress.set_postfix(docs=f"{n_docs:,}")
        else:
            progress.set_postfix(docs=f"{n_docs:,}", truncated="yes")
            return total_tokens, n_docs, True

        if max_docs is not None and n_docs >= max_docs:
            return total_tokens, n_docs, True
        if max_tokens is not None and total_tokens >= max_tokens:
            return total_tokens, n_docs, True

    return total_tokens, n_docs, False


def build_token_stream(
    out_path: str,
    split: str = "train",
    max_docs: int | None = None,
    max_tokens: int | None = None,
    data_dir: str = DEFAULT_FINEWEB_DIR,
    validation_frac: float = 0.01,
    test_frac: float = 0.0,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> None:
    """actually using all the helper functions from above"""
    parquet_files = _split_parquet_files(
        data_dir=data_dir,
        split=split,
        validation_frac=validation_frac,
        test_frac=test_frac,
    )
    print(
        f"Using {len(parquet_files):,} parquet shards, split={split} "
        f"from {data_dir} with {num_workers} workers"
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    n_docs = 0

    text_batches = _iter_local_fineweb_text_batches(
        data_dir=data_dir,
        split=split,
        validation_frac=validation_frac,
        test_frac=test_frac,
        num_workers=num_workers,
    )

    progress_total = max_tokens if max_tokens is not None else None
    stop_early = False
    truncated_tail = False
    with (
        open(out_path, "wb") as f,
        tqdm(
            total=progress_total,
            desc=f"tokenize:{split}",
            unit="tok",
            unit_scale=True,
            dynamic_ncols=True,
        ) as progress,
    ):
        if num_workers == 1:
            token_batches_iter = map(_tokenize_text_batch, text_batches)
        else:
            executor = ProcessPoolExecutor(max_workers=num_workers)
            token_batches_iter = executor.map(_tokenize_text_batch, text_batches)

        try:
            for token_batch in token_batches_iter:
                total_tokens, n_docs, stop_early = _write_token_batch(
                    token_batch=token_batch,
                    f=f,
                    progress=progress,
                    total_tokens=total_tokens,
                    n_docs=n_docs,
                    max_docs=max_docs,
                    max_tokens=max_tokens,
                )
                if stop_early:
                    truncated_tail = max_tokens is not None and total_tokens >= max_tokens
                    break
        finally:
            if num_workers > 1:
                executor.shutdown(wait=not stop_early, cancel_futures=stop_early)

    if hasattr(text_batches, "close"):
        with contextlib.suppress(Exception):
            text_batches.close()

    truncated_msg = " + truncated tail" if truncated_tail else ""
    print(
        f"Saved {total_tokens:,} tokens ({n_docs:,} full docs{truncated_msg}) → {out_path}",
        flush=True,
    )

# building index files for tokenized data

def build_start_indices(
    token_file: str,
    seq_len: int,
    num_tokens_to_use: int,
    seed: int,
    out_path: str,
    stride: int | None = None,
    eos_token_id: int = GPT2_EOS_TOKEN_ID,
) -> None:
    """
    Compute and shuffle example start positions; save to .npy file. Each window is within a single EOS-delimited document.
    """
    tokens = np.memmap(token_file, dtype=np.uint16, mode="r")
    usable = min(int(len(tokens)), num_tokens_to_use)

    if stride is None:
        stride = seq_len  

    chunk_size = 10_000_000
    doc_start = 0
    n_docs = 0
    skipped_docs = 0
    starts_parts: list[np.ndarray] = []

    with tqdm(total=usable, desc="index", unit="tok", dynamic_ncols=True) as progress:
        for offset in range(0, usable, chunk_size):
            chunk_end = min(offset + chunk_size, usable)
            chunk = np.asarray(tokens[offset:chunk_end])
            eos_positions = np.flatnonzero(chunk == eos_token_id) + offset

            for eos_pos in eos_positions:
                doc_len = int(eos_pos - doc_start + 1)
                if doc_len >= seq_len + 1:
                    starts_parts.append(
                        np.arange(
                            doc_start,
                            eos_pos - seq_len + 1,
                            stride,
                            dtype=np.int64,
                        )
                    )
                else:
                    skipped_docs += 1
                doc_start = int(eos_pos + 1)
                n_docs += 1

            progress.update(chunk_end - offset)
            progress.set_postfix(docs=f"{n_docs:,}", skipped=f"{skipped_docs:,}")

    starts = (
        np.concatenate(starts_parts)
        if starts_parts
        else np.empty(0, dtype=np.int64)
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(starts)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, starts)

    print(f"Saved {len(starts):,} example starts → {out_path}")



class FixedOrderTokenDataset(Dataset):
    """
    Memory-mapped, fixed-order dataset.
    """

    def __init__(self, token_file: str, starts_file: str, seq_len: int) -> None:
        self.tokens = np.memmap(token_file, dtype=np.uint16, mode="r")
        self.starts = np.load(starts_file)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = int(self.starts[idx])
        chunk = self.tokens[s : s + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return {"input_ids": x, "labels": y}


# cli stuff, was much easier to run this way, much of this would take hours, so I would chain commands using && in terminal.



def _cmd_tokenize(args: argparse.Namespace) -> None:
    build_token_stream(
        out_path=args.out,
        split=args.split,
        max_docs=args.max_docs,
        max_tokens=args.max_tokens,
        data_dir=args.data_dir,
        validation_frac=args.validation_frac,
        test_frac=args.test_frac,
        num_workers=args.num_workers,
    )


def _cmd_index(args: argparse.Namespace) -> None:
    build_start_indices(
        token_file=args.token_file,
        seq_len=args.seq_len,
        num_tokens_to_use=int(args.num_tokens),
        seed=args.seed,
        out_path=args.out,
        stride=args.stride,
        eos_token_id=args.eos_token_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="data prep")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tok = sub.add_parser("tokenize")
    p_tok.add_argument("--split", default="train")
    p_tok.add_argument("--out", required=True)
    p_tok.add_argument("--max_docs", type=int, default=None)
    p_tok.add_argument("--max_tokens", type=int, default=None)
    p_tok.add_argument("--data_dir", default=DEFAULT_FINEWEB_DIR)
    p_tok.add_argument("--validation_frac", type=float, default=0.01)
    p_tok.add_argument("--test_frac", type=float, default=0.0)
    p_tok.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)

    p_idx = sub.add_parser("index")
    p_idx.add_argument("--token_file", required=True)
    p_idx.add_argument("--out", required=True)
    p_idx.add_argument("--num_tokens", type=float, default=1e9)
    p_idx.add_argument("--seq_len", type=int, default=1024)
    p_idx.add_argument("--stride", type=int, default=None)
    p_idx.add_argument("--seed", type=int, default=1234)
    p_idx.add_argument("--eos_token_id", type=int, default=GPT2_EOS_TOKEN_ID)

    args = parser.parse_args()
    {"tokenize": _cmd_tokenize, "index": _cmd_index}[args.cmd](args)


if __name__ == "__main__":
    main()
