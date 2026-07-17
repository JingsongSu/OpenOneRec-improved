"""Stateful dataloader using the pause-prefix Qwen3 dataset."""

from __future__ import annotations

from torchdata.stateful_dataloader import StatefulDataLoader

from onerec_llm.data.qwen3_pause_dataset import Qwen3PausePrefixParquetDataset


def get_chat_completion_parquet_dataloader(
    sources: str,
    max_length,
    base_model_dir,
    num_epochs=1,
    shuffle_seed=1024,
    num_workers=8,
    datasource_config=None,
    **kwargs,
):
    datasource_config = datasource_config or {}
    num_readers = kwargs.get("num_readers", 1)
    shuffle_window = kwargs.get("shuffle_window", 0)

    def input_creator():
        return Qwen3PausePrefixParquetDataset(
            sources=sources,
            num_workers=num_workers,
            num_epochs=num_epochs,
            shuffle_seed=shuffle_seed,
            max_length=max_length,
            base_model_dir=base_model_dir,
            datasource_config=datasource_config,
            num_readers=num_readers,
            shuffle_window=shuffle_window,
            **kwargs,
        )

    dataset = input_creator()
    return StatefulDataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=lambda rows: rows[0],
    )


def get_dataloader(name: str, **kwargs):
    if name == "chat_completion_parquet":
        return get_chat_completion_parquet_dataloader(**kwargs)
    raise NotImplementedError(f"Unsupported dataloader: {name}")


__all__ = ["get_dataloader", "get_chat_completion_parquet_dataloader"]
