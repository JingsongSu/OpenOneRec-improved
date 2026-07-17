"""Pause-prefix SFT dataset for OpenOneRec's Qwen3 chat data.

A repeated trainable token is inserted after the final assistant header and
before the answer. The inserted positions are part of the causal sequence, so
successive pause hidden states can attend to the prompt and earlier pause states.
They are fixed input/prefill tokens rather than generated output tokens.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch

from onerec_llm.data.pause_prefix_utils import (
    apply_pause_loss_mask,
    insert_pause_after_assistant_header,
)
from onerec_llm.data.qwen3_dataset import Qwen3ChatCompletionParquetDataset
from onerec_llm.utils.common import print_rank_0

logger = logging.getLogger(__name__)


class Qwen3PausePrefixParquetDataset(Qwen3ChatCompletionParquetDataset):
    """OpenOneRec chat dataset with a fixed latent pause prefix."""

    def __init__(self, *args, **kwargs):
        self.pause_token = str(kwargs.pop("pause_token", "<|latent_pause|>"))
        self.pause_count = int(kwargs.pop("pause_count", 5))
        self.pause_insert_mode = str(kwargs.pop("pause_insert_mode", "last"))
        self.pause_only_for_itemic = bool(kwargs.pop("pause_only_for_itemic", True))
        if self.pause_count <= 0:
            raise ValueError("pause_count must be positive")
        if self.pause_insert_mode not in {"last", "all"}:
            raise ValueError("pause_insert_mode must be 'last' or 'all'")

        super().__init__(*args, **kwargs)

        pause_ids = self.tokenizer.encode(
            self.pause_token,
            add_special_tokens=False,
        )
        if len(pause_ids) != 1:
            raise ValueError(
                f"{self.pause_token!r} must map to exactly one token ID, got {pause_ids}. "
                "Run tools/prepare_pause_model.py first and use that directory "
                "as base_model_dir/model_dir."
            )
        self.pause_token_id = int(pause_ids[0])
        model_pause_id = getattr(self.tokenizer, "pause_token_id", None)
        if model_pause_id is not None and int(model_pause_id) != self.pause_token_id:
            raise ValueError(
                f"Tokenizer pause token mismatch: {model_pause_id} != {self.pause_token_id}"
            )
        print_rank_0(
            "PausePrefixDataset: "
            f"token={self.pause_token!r}, id={self.pause_token_id}, "
            f"count={self.pause_count}, mode={self.pause_insert_mode}, "
            f"only_itemic={self.pause_only_for_itemic}"
        )

    def _process_chat(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Reuse all original message conversion/chat-template behavior first.
        inputs = super()._process_chat(sample)
        original_ids = inputs["input_ids"][0].tolist()

        new_ids, spans = insert_pause_after_assistant_header(
            input_ids=original_ids,
            assistant_start_pattern=self.assistant_start_pattern,
            assistant_end_pattern=self.im_end_pattern,
            pause_token_id=self.pause_token_id,
            pause_count=self.pause_count,
            insert_mode=self.pause_insert_mode,
            required_target_id_range=(
                tuple(self.itemic_id_range)
                if self.pause_only_for_itemic and self.itemic_id_range is not None
                else None
            ),
        )
        if not spans:
            # This is expected for text-only SFT samples when pause_only_for_itemic=True.
            return inputs
        if len(new_ids) > self.max_length:
            raise ValueError(
                f"Sample becomes too long after pause insertion: {len(new_ids)} > "
                f"max_length={self.max_length}. Reduce max_sample_length or pause_count."
            )

        input_ids = torch.tensor(new_ids, dtype=inputs["input_ids"].dtype).unsqueeze(0)
        assistant_mask = self._get_assistant_mask(
            input_ids,
            start_pattern=self.assistant_start_pattern,
            end_pattern=self.im_end_pattern,
        )[0].tolist()
        loss_mask = apply_pause_loss_mask(assistant_mask, spans)

        # Keep the original repository convention: final pad/eos position is ignored.
        if loss_mask:
            loss_mask[-1] = 0

        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = torch.ones_like(input_ids)
        inputs["loss_mask"] = torch.tensor(loss_mask, dtype=torch.long).unsqueeze(0)

        itemic_id_mask = torch.zeros_like(input_ids)
        if self.itemic_id_range is not None:
            itemic_id_mask[
                (input_ids >= self.itemic_id_range[0])
                & (input_ids <= self.itemic_id_range[1])
            ] = 1
        inputs["itemic_id_mask"] = itemic_id_mask
        inputs["position_ids"] = self._get_rope_index_qwen3(input_ids)
        return inputs


__all__ = ["Qwen3PausePrefixParquetDataset"]
