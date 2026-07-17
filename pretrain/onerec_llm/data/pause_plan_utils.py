"""Pure tensor helpers for optional pause-to-Itemic planning supervision."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def collect_pause_plan_pairs_with_positions(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    sample_idx: Optional[torch.Tensor],
    pause_token_id: int,
    itemic_start: int,
    itemic_end: int,
    max_pairs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return selected ``[batch, sequence]`` pause positions and target IDs.

    If a sample contains ``k`` pauses and ``m`` future Itemic tokens, the final
    ``min(k, m)`` pause positions are paired with the first ``min(k, m)`` Itemic
    tokens. Earlier pause states remain free latent computation slots.

    Returning positions instead of only materialized hidden states lets the
    training recipe compute the small auxiliary head gradient on detached leaves
    and later scatter the resulting state gradients back into the full hidden
    tensor before the single FSDP-safe transformer backward pass.
    """

    if hidden_states.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("Expected hidden_states [B,L,H] and input_ids [B,L]")
    if hidden_states.shape[:2] != input_ids.shape:
        raise ValueError("hidden_states and input_ids sequence shapes differ")

    if sample_idx is None:
        sample_idx = torch.zeros_like(input_ids, dtype=torch.int32)
    if sample_idx.shape != input_ids.shape:
        raise ValueError("sample_idx and input_ids must have identical shapes")

    chosen_positions: List[torch.Tensor] = []
    chosen_targets: List[torch.Tensor] = []

    for batch_index in range(input_ids.shape[0]):
        row_ids = input_ids[batch_index]
        row_samples = sample_idx[batch_index]
        valid_sample_ids = torch.unique(row_samples[row_samples >= 0]).tolist()

        for sid in valid_sample_ids:
            sample_positions = torch.nonzero(
                row_samples == int(sid), as_tuple=False
            ).flatten()
            if sample_positions.numel() == 0:
                continue

            sample_tokens = row_ids.index_select(0, sample_positions)
            pause_local = torch.nonzero(
                sample_tokens == pause_token_id, as_tuple=False
            ).flatten()
            if pause_local.numel() == 0:
                continue

            last_pause_local = int(pause_local[-1].item())
            future_local = torch.arange(
                last_pause_local + 1,
                sample_tokens.numel(),
                device=sample_tokens.device,
            )
            if future_local.numel() == 0:
                continue

            future_tokens = sample_tokens.index_select(0, future_local)
            itemic_mask = (future_tokens >= itemic_start) & (
                future_tokens <= itemic_end
            )
            target_ids = future_tokens[itemic_mask]
            if target_ids.numel() == 0:
                continue

            pair_count = min(int(pause_local.numel()), int(target_ids.numel()))
            pause_for_plan_local = pause_local[-pair_count:]
            target_ids = target_ids[:pair_count]
            pause_positions = sample_positions.index_select(
                0, pause_for_plan_local
            )

            batch_positions = torch.full_like(pause_positions, batch_index)
            chosen_positions.append(
                torch.stack([batch_positions, pause_positions], dim=-1).long()
            )
            chosen_targets.append(target_ids)

    if not chosen_positions:
        empty_positions = torch.empty(
            (0, 2), dtype=torch.long, device=input_ids.device
        )
        empty_targets = input_ids.new_empty((0,))
        return empty_positions, empty_targets

    positions = torch.cat(chosen_positions, dim=0)
    targets = torch.cat(chosen_targets, dim=0)

    if max_pairs > 0 and targets.numel() > max_pairs:
        indices = torch.linspace(
            0,
            targets.numel() - 1,
            steps=max_pairs,
            device=targets.device,
        ).long()
        positions = positions.index_select(0, indices)
        targets = targets.index_select(0, indices)

    return positions, targets


def collect_pause_plan_pairs(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    sample_idx: Optional[torch.Tensor],
    pause_token_id: int,
    itemic_start: int,
    itemic_end: int,
    max_pairs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialize pause hidden states and targets for compatibility/tests."""

    positions, targets = collect_pause_plan_pairs_with_positions(
        hidden_states=hidden_states,
        input_ids=input_ids,
        sample_idx=sample_idx,
        pause_token_id=pause_token_id,
        itemic_start=itemic_start,
        itemic_end=itemic_end,
        max_pairs=max_pairs,
    )
    if positions.numel() == 0:
        states = hidden_states.new_empty((0, hidden_states.shape[-1]))
    else:
        states = hidden_states[positions[:, 0], positions[:, 1]]
    return states, targets


__all__ = [
    "collect_pause_plan_pairs",
    "collect_pause_plan_pairs_with_positions",
]
