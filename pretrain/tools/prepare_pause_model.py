#!/usr/bin/env python3
"""Add one trainable pause token to a standard Qwen3/OpenOneRec checkpoint.

This is a one-time preprocessing step before pause-prefix SFT. The model remains
``Qwen3ForCausalLM``; only the tokenizer vocabulary and embedding/lm_head rows
are resized. The new row is initialized near the vocabulary center, following
the practical motivation of pause-token latent reasoning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AddedToken, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model_dir", required=True)
    parser.add_argument("--output_model_dir", required=True)
    parser.add_argument("--pause_token", default="<|latent_pause|>")
    parser.add_argument("--pause_count", type=int, default=5)
    parser.add_argument(
        "--init_mode",
        choices=("full_vocab_mean", "text_itemic_bridge"),
        default="full_vocab_mean",
    )
    parser.add_argument("--itemic_start", type=int, default=151669)
    parser.add_argument("--itemic_end", type=int, default=169742)
    parser.add_argument(
        "--device_map",
        default="cpu",
        help="Transformers device_map, e.g. cpu or auto.",
    )
    return parser.parse_args()


def _mean_initialization(weight, old_vocab_size, mode, itemic_start, itemic_end):
    source = weight[:old_vocab_size].float()
    if mode == "full_vocab_mean":
        center = source.mean(dim=0)
    else:
        if not (0 <= itemic_start <= itemic_end < old_vocab_size):
            raise ValueError(
                "Invalid Itemic range for text_itemic_bridge initialization: "
                f"[{itemic_start}, {itemic_end}] vs vocab={old_vocab_size}"
            )
        text = source[:itemic_start]
        itemic = source[itemic_start : itemic_end + 1]
        center = 0.5 * (text.mean(dim=0) + itemic.mean(dim=0))
    return center.to(dtype=weight.dtype, device=weight.device)


def main():
    args = parse_args()
    if args.pause_count <= 0:
        raise ValueError("--pause_count must be positive")

    input_dir = Path(args.input_model_dir).resolve()
    output_dir = Path(args.output_model_dir).resolve()
    if input_dir == output_dir:
        raise ValueError("Use a new output directory; in-place resizing is disabled")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(input_dir, trust_remote_code=True)
    old_vocab_size = len(tokenizer)
    encoded_before = tokenizer.encode(args.pause_token, add_special_tokens=False)
    token_already_exists = (
        len(encoded_before) == 1
        and tokenizer.convert_ids_to_tokens(encoded_before[0]) == args.pause_token
    )
    if token_already_exists:
        added = 0
    else:
        added = tokenizer.add_tokens(
            [
                AddedToken(
                    args.pause_token,
                    single_word=False,
                    lstrip=False,
                    rstrip=False,
                    normalized=False,
                    special=True,
                )
            ],
            special_tokens=True,
        )
    pause_ids = tokenizer.encode(args.pause_token, add_special_tokens=False)
    if len(pause_ids) != 1:
        raise RuntimeError(
            f"Pause token must encode to one ID after insertion, got {pause_ids}"
        )
    pause_token_id = int(pause_ids[0])

    dtype = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        input_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    if added:
        with torch.no_grad():
            input_weight = model.get_input_embeddings().weight
            center = _mean_initialization(
                input_weight,
                old_vocab_size=old_vocab_size,
                mode=args.init_mode,
                itemic_start=args.itemic_start,
                itemic_end=args.itemic_end,
            )
            input_weight[pause_token_id].copy_(center)

            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None:
                output_weight = output_embeddings.weight
                if output_weight.data_ptr() != input_weight.data_ptr():
                    output_center = _mean_initialization(
                        output_weight,
                        old_vocab_size=old_vocab_size,
                        mode=args.init_mode,
                        itemic_start=args.itemic_start,
                        itemic_end=args.itemic_end,
                    )
                    output_weight[pause_token_id].copy_(output_center)

    model.config.vocab_size = len(tokenizer)
    model.config.pause_token = args.pause_token
    model.config.pause_token_id = pause_token_id
    model.config.pause_count = int(args.pause_count)
    model.config.pause_prefix_type = "fixed_repeated_token"
    model.config.pause_itemic_start = int(args.itemic_start)
    model.config.pause_itemic_end = int(args.itemic_end)
    model.config.architectures = ["Qwen3ForCausalLM"]

    tokenizer.init_kwargs["pause_token"] = args.pause_token
    tokenizer.init_kwargs["pause_token_id"] = pause_token_id
    tokenizer.init_kwargs["pause_count"] = int(args.pause_count)

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    sidecar = {
        "pause_token": args.pause_token,
        "pause_token_id": pause_token_id,
        "pause_count": int(args.pause_count),
        "init_mode": args.init_mode,
        "old_vocab_size": old_vocab_size,
        "new_vocab_size": len(tokenizer),
        "itemic_start": int(args.itemic_start),
        "itemic_end": int(args.itemic_end),
    }
    (output_dir / "pause_prefix_config.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(sidecar, ensure_ascii=False, indent=2))
    print(f"Prepared pause-prefix model: {output_dir}")


if __name__ == "__main__":
    main()
