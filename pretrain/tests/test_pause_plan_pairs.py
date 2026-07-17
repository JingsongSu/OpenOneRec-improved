import torch

from onerec_llm.data.pause_plan_utils import (
    collect_pause_plan_pairs,
    collect_pause_plan_pairs_with_positions,
)


def test_collects_late_pauses_for_future_itemic_hierarchy():
    pause = 99
    itemic_start, itemic_end = 1000, 1100
    # One packed row containing two samples. The first has five pauses and three
    # Itemic targets; only the final three pause states should be supervised.
    input_ids = torch.tensor(
        [[10, pause, pause, pause, pause, pause, 1001, 1002, 1003, 11,
          20, pause, pause, 1004, 1005, 21]]
    )
    sample_idx = torch.tensor([[0] * 10 + [1] * 6], dtype=torch.int32)
    hidden = torch.arange(16 * 4, dtype=torch.float32).reshape(1, 16, 4)

    states, targets = collect_pause_plan_pairs(
        hidden_states=hidden,
        input_ids=input_ids,
        sample_idx=sample_idx,
        pause_token_id=pause,
        itemic_start=itemic_start,
        itemic_end=itemic_end,
        max_pairs=64,
    )
    positions, position_targets = collect_pause_plan_pairs_with_positions(
        hidden_states=hidden,
        input_ids=input_ids,
        sample_idx=sample_idx,
        pause_token_id=pause,
        itemic_start=itemic_start,
        itemic_end=itemic_end,
        max_pairs=64,
    )

    assert targets.tolist() == [1001, 1002, 1003, 1004, 1005]
    assert torch.equal(position_targets, targets)

    expected_positions = [3, 4, 5, 11, 12]
    assert torch.equal(states, hidden[0, expected_positions])
    assert positions.tolist() == [[0, p] for p in expected_positions]


def test_max_pairs_keeps_positions_targets_and_states_aligned():
    pause = 99
    input_ids = torch.tensor(
        [[pause, pause, pause, pause, 1000, 1001, 1002, 1003]]
    )
    sample_idx = torch.zeros_like(input_ids, dtype=torch.int32)
    hidden = torch.arange(8 * 3, dtype=torch.float32).reshape(1, 8, 3)

    states, targets = collect_pause_plan_pairs(
        hidden_states=hidden,
        input_ids=input_ids,
        sample_idx=sample_idx,
        pause_token_id=pause,
        itemic_start=1000,
        itemic_end=1100,
        max_pairs=2,
    )
    positions, position_targets = collect_pause_plan_pairs_with_positions(
        hidden_states=hidden,
        input_ids=input_ids,
        sample_idx=sample_idx,
        pause_token_id=pause,
        itemic_start=1000,
        itemic_end=1100,
        max_pairs=2,
    )

    assert torch.equal(position_targets, targets)
    assert torch.equal(states, hidden[positions[:, 0], positions[:, 1]])
    assert targets.numel() == 2
