#!/usr/bin/env python3
"""Create a pause-prefix dataset config from the existing SFT JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original sft.json")
    parser.add_argument("--output", required=True, help="New pause SFT config")
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--pause_token", default="<|latent_pause|>")
    parser.add_argument("--pause_count", type=int, default=5)
    parser.add_argument("--pause_insert_mode", choices=("last", "all"), default="last")
    parser.add_argument(
        "--pause_all_sft_samples",
        action="store_true",
        help="Insert pauses for text-only assistant targets too.",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.input).read_text(encoding="utf-8"))
    config["base_model_dir"] = str(Path(args.base_model_dir).resolve())
    config["model_class"] = "Qwen3ForCausalLM"
    config["pause_token"] = args.pause_token
    config["pause_count"] = int(args.pause_count)
    config["pause_insert_mode"] = args.pause_insert_mode
    config["pause_only_for_itemic"] = not args.pause_all_sft_samples

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
