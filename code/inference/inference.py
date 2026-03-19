"""
uv run python inference/inference.py --checkpoint grassmann.pt --prompt "To be or not to be"


uv run python inference/inference.py --checkpoint grassmann.pt --chat

flags for generation settings:
  --max_new_tokens 200
  --temperature 0.8
  --top_k 50
  --top_p 0.95
"""
from __future__ import annotations # helps with errors/type hints

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.eval_helpers import generate_tokens
from training.checkpoints import load_model_from_checkpoint


def load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, model_type = load_model_from_checkpoint(ckpt)
    model.to(device)
    model.eval()

    print(f"Loaded {model_type}LM from {checkpoint_path}  (step={ckpt.get('step', '?')})")
    return model




def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.model_max_length = int(1e30)
    return tok




@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: str,
    decode_mode: str,
) -> torch.Tensor:
    result = generate_tokens(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
        amp_dtype=None,
        decode_mode=decode_mode,
    )
    return result["new_ids"]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def run_prompt(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: str,
    decode_mode: str,
) -> str:
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long
    )
    new_ids = generate(
        model, input_ids, max_new_tokens, temperature, top_k, top_p, device, decode_mode
    )
    return tokenizer.decode(new_ids[0].tolist(), skip_special_tokens=True)


def run_chat(
    model: torch.nn.Module,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: str,
    decode_mode: str,
) -> None:
    history_ids: list[int] = []

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt.lower() in {"quit", "exit", "q"}:
            break
        if not prompt:
            continue

        new_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        new_tokens.append(tokenizer.eos_token_id)
        history_ids.extend(new_tokens)

        input_ids = torch.tensor([history_ids], dtype=torch.long)
        reply_ids = generate(
            model, input_ids, max_new_tokens, temperature, top_k, top_p, device, decode_mode
        )
        reply_ids_list = reply_ids[0].tolist()
        reply_text = tokenizer.decode(reply_ids_list, skip_special_tokens=True)

        history_ids.extend(reply_ids_list)
        # Trim if grows beyond max_seq_len
        if len(history_ids) > model.cfg.max_seq_len:
            history_ids = history_ids[-model.cfg.max_seq_len:]

        print(f"Model: {reply_text}\n")


def main():
    parser = argparse.ArgumentParser(description="GrassmannLM / TransformerLM inference")

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--chat", action="store_true")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--decode_mode",
        choices=["auto", "full", "stateful"],
        default="auto",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    tokenizer = get_tokenizer()

    if args.chat:
        run_chat(
            model, tokenizer,
            args.max_new_tokens, args.temperature, args.top_k, args.top_p, device,
            args.decode_mode,
        )
    else:
        output = run_prompt(
            model, tokenizer, args.prompt,
            args.max_new_tokens, args.temperature, args.top_k, args.top_p, device,
            args.decode_mode,
        )
        print(output)


if __name__ == "__main__":
    main()
