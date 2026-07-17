"""Pure helpers for inserting fixed latent pause-prefix tokens.

The functions in this file deliberately have no OpenOneRec dependency so they
can be unit-tested independently. A pause token is a normal vocabulary token,
but its hidden state is used as latent scratch space before recommendation
output tokens are decoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class InsertedPauseSpan:
    """Location of one inserted pause block in the new token sequence."""

    start: int
    end: int  # exclusive
    had_target_content: bool


def find_subsequence_starts(sequence: Sequence[int], pattern: Sequence[int]) -> List[int]:
    """Return all start offsets of ``pattern`` in ``sequence``."""

    if not pattern:
        raise ValueError("pattern must not be empty")
    width = len(pattern)
    return [
        index
        for index in range(0, len(sequence) - width + 1)
        if list(sequence[index : index + width]) == list(pattern)
    ]


def _find_next_pattern(
    sequence: Sequence[int], pattern: Sequence[int], start: int
) -> int | None:
    if not pattern:
        return None
    width = len(pattern)
    for index in range(start, len(sequence) - width + 1):
        if list(sequence[index : index + width]) == list(pattern):
            return index
    return None


def insert_pause_after_assistant_header(
    input_ids: Sequence[int],
    assistant_start_pattern: Sequence[int],
    assistant_end_pattern: Sequence[int],
    pause_token_id: int,
    pause_count: int,
    insert_mode: str = "last",
    required_target_id_range: Tuple[int, int] | None = None,
) -> Tuple[List[int], List[InsertedPauseSpan]]:
    """Insert repeated pause tokens before assistant content.

    Args:
        input_ids: Complete tokenized chat, including assistant answer(s).
        assistant_start_pattern: Token sequence for ``<|im_start|>assistant\\n``.
        assistant_end_pattern: Token sequence marking the end of an assistant turn.
        pause_token_id: Vocabulary ID of the trainable pause token.
        pause_count: Number of causal latent slots to insert.
        insert_mode: ``last`` inserts before the final assistant answer; ``all``
            inserts before every assistant answer.
        required_target_id_range: When provided, insert only for assistant turns
            containing at least one token in this inclusive ID range.

    Returns:
        The new token IDs and inserted spans, expressed in the new sequence.
    """

    if pause_count <= 0:
        raise ValueError(f"pause_count must be positive, got {pause_count}")
    if insert_mode not in {"last", "all"}:
        raise ValueError(f"insert_mode must be 'last' or 'all', got {insert_mode!r}")

    original = list(input_ids)
    header_starts = find_subsequence_starts(original, assistant_start_pattern)
    if not header_starts:
        return original, []
    chosen = header_starts[-1:] if insert_mode == "last" else header_starts

    # Record whether the original assistant turn contains at least one token.
    insertions: List[Tuple[int, bool]] = []
    header_width = len(assistant_start_pattern)
    for header_start in chosen:
        content_start = header_start + header_width
        end_start = _find_next_pattern(
            original, assistant_end_pattern, start=content_start
        )
        content_end = len(original) if end_start is None else end_start
        had_content = content_start < content_end
        if required_target_id_range is not None:
            range_start, range_end = required_target_id_range
            if range_start > range_end:
                raise ValueError("required_target_id_range must be ordered")
            content = original[content_start:content_end]
            if not any(range_start <= token_id <= range_end for token_id in content):
                continue
        insertions.append((content_start, had_content))

    output = list(original)
    # Right-to-left insertion keeps original offsets valid.
    for content_start, _ in reversed(insertions):
        output[content_start:content_start] = [int(pause_token_id)] * pause_count

    spans: List[InsertedPauseSpan] = []
    cumulative_shift = 0
    for content_start, had_content in insertions:
        start = content_start + cumulative_shift
        end = start + pause_count
        spans.append(
            InsertedPauseSpan(
                start=start,
                end=end,
                had_target_content=had_content,
            )
        )
        cumulative_shift += pause_count
    return output, spans


def apply_pause_loss_mask(
    assistant_mask: Sequence[int],
    spans: Iterable[InsertedPauseSpan],
) -> List[int]:
    """Mask pause-token prediction while preserving the first target-token CE.

    OpenOneRec shifts labels *before* applying ``loss_mask``. Therefore the loss
    value at position ``i`` supervises token ``i+1``. For a block of repeated
    pause tokens, all positions except the final pause are set to zero, and the
    final pause is set to one when real assistant content follows it. This means:

    - the model is not trained to predict pause tokens;
    - the final pause hidden state predicts the first real target token;
    - subsequent target tokens keep the repository's original assistant mask.
    """

    mask = list(int(value) for value in assistant_mask)
    for span in spans:
        if span.start < 0 or span.end > len(mask) or span.start >= span.end:
            raise ValueError(f"invalid inserted span {span} for mask length {len(mask)}")
        for index in range(span.start, span.end):
            mask[index] = 0
        if span.had_target_content:
            mask[span.end - 1] = 1
    return mask


def append_pause_suffix(
    prompt_token_ids: Sequence[int], pause_token_id: int, pause_count: int
) -> List[int]:
    """Append a pause block to an inference prompt, without duplicating it."""

    if pause_count <= 0:
        raise ValueError(f"pause_count must be positive, got {pause_count}")
    prompt = list(prompt_token_ids)
    suffix = [int(pause_token_id)] * pause_count
    if len(prompt) >= pause_count and prompt[-pause_count:] == suffix:
        return prompt
    return prompt + suffix


__all__ = [
    "InsertedPauseSpan",
    "append_pause_suffix",
    "apply_pause_loss_mask",
    "find_subsequence_starts",
    "insert_pause_after_assistant_header",
]
