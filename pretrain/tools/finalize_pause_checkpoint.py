#!/usr/bin/env python3
"""Copy pause tokenizer/config metadata into a converted SFT HF checkpoint.

OpenOneRec's distributed checkpoint conversion may focus on model tensors. This
small post-step guarantees the converted directory contains the expanded
tokenizer and pause metadata required by vLLM prompt injection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

from transformers import AutoConfig, AutoTokenizer


def _safetensor_shape(model_dir: Path, tensor_name: str) -> Optional[Tuple[int, ...]]:
    """Read one tensor shape without materializing the full checkpoint."""

    try:
        from safetensors import safe_open
    except ImportError:
        return None

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_name = index.get("weight_map", {}).get(tensor_name)
        if shard_name is None:
            return None
        shard_path = model_dir / shard_name
    else:
        shard_path = model_dir / "model.safetensors"
        if not shard_path.exists():
            return None

    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        if tensor_name not in handle.keys():
            return None
        return tuple(handle.get_slice(tensor_name).get_shape())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_model_dir", required=True)
    parser.add_argument("--converted_sft_dir", required=True)
    args = parser.parse_args()

    prepared = Path(args.prepared_model_dir).resolve()
    target = Path(args.converted_sft_dir).resolve()
    if not target.exists():
        raise FileNotFoundError(target)

    tokenizer = AutoTokenizer.from_pretrained(prepared, trust_remote_code=True)
    expected_vocab = len(tokenizer)

    # Verify converted tensor rows before writing config metadata. A mismatch
    # means the distributed conversion used the old vocabulary and must be fixed.
    checked_any = False
    for tensor_name in ("model.embed_tokens.weight", "lm_head.weight"):
        shape = _safetensor_shape(target, tensor_name)
        if shape is None:
            continue
        checked_any = True
        if not shape or shape[0] != expected_vocab:
            raise ValueError(
                f"{tensor_name} has shape {shape}, but pause tokenizer requires "
                f"vocab_size={expected_vocab}. Re-run conversion from the pause SFT "
                "checkpoint/config instead of only copying tokenizer files."
            )
    if not checked_any:
        print(
            "WARNING: could not inspect safetensors shapes; config/tokenizer will "
            "be synchronized, but manually verify embedding rows before vLLM."
        )

    tokenizer.save_pretrained(target)
    source_config = AutoConfig.from_pretrained(prepared, trust_remote_code=True)
    target_config = AutoConfig.from_pretrained(target, trust_remote_code=True)
    fields = (
        "pause_token",
        "pause_token_id",
        "pause_count",
        "pause_prefix_type",
        "pause_itemic_start",
        "pause_itemic_end",
    )
    for field in fields:
        if not hasattr(source_config, field):
            raise ValueError(f"Prepared config is missing {field}")
        setattr(target_config, field, getattr(source_config, field))
    target_config.vocab_size = expected_vocab
    target_config.architectures = ["Qwen3ForCausalLM"]
    target_config.save_pretrained(target)

    metadata = {field: getattr(target_config, field) for field in fields}
    metadata["vocab_size"] = expected_vocab
    (target / "pause_prefix_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Finalized pause checkpoint: {target}")


if __name__ == "__main__":
    main()
