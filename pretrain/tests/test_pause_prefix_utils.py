from onerec_llm.data.pause_prefix_utils import (
    append_pause_suffix,
    apply_pause_loss_mask,
    insert_pause_after_assistant_header,
)


def test_insert_last_assistant_and_shift_aware_mask():
    start = [10, 11]
    end = [12]
    original = [1, 10, 11, 20, 21, 12, 2, 10, 11, 30, 31, 12, 3]
    new_ids, spans = insert_pause_after_assistant_header(
        original,
        assistant_start_pattern=start,
        assistant_end_pattern=end,
        pause_token_id=99,
        pause_count=3,
        insert_mode="last",
    )
    assert new_ids == [1, 10, 11, 20, 21, 12, 2, 10, 11, 99, 99, 99, 30, 31, 12, 3]
    assert len(spans) == 1
    span = spans[0]
    assert (span.start, span.end, span.had_target_content) == (9, 12, True)

    # Simulate an assistant mask after insertion: first and second turns' content.
    assistant_mask = [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
    masked = apply_pause_loss_mask(assistant_mask, spans)
    assert masked[9:12] == [0, 0, 1]
    assert masked[12:14] == [1, 1]


def test_append_pause_suffix_is_idempotent():
    once = append_pause_suffix([1, 2], pause_token_id=9, pause_count=4)
    twice = append_pause_suffix(once, pause_token_id=9, pause_count=4)
    assert once == [1, 2, 9, 9, 9, 9]
    assert twice == once


def test_itemic_filter_skips_text_only_turn():
    original = [10, 11, 20, 21, 12]
    new_ids, spans = insert_pause_after_assistant_header(
        original,
        assistant_start_pattern=[10, 11],
        assistant_end_pattern=[12],
        pause_token_id=99,
        pause_count=3,
        required_target_id_range=(1000, 1100),
    )
    assert new_ids == original
    assert spans == []
