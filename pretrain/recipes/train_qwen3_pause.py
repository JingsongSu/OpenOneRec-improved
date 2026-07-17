"""SFT entrypoint for fixed latent pause-prefix reasoning.

The persistent model architecture remains the repository's standard
``Qwen3ForCausalLM``. The only mandatory changes are in the dataset: a learned
special token is repeated before the assistant target and used as causal latent
scratch space. Consequently, converted checkpoints remain natively loadable by
vLLM.

An optional hierarchy-planning auxiliary loss can directly supervise the last
few pause hidden states to predict future Itemic tokens. It is disabled by
default so the first experiment matches the simple PauseRec-style objective.

FSDP2 note
----------
OpenOneRec's ``ChunkedLossComputer`` computes lm_head gradients manually and
only then calls backward once on the transformer hidden states. The pause-plan
auxiliary objective must follow the same rule. Calling ``aux_loss.backward()``
before ``ChunkedLossComputer`` finishes causes FSDP2's root module to run its
post-backward reshard logic, turning ``lm_head.weight`` back into a DTensor;
the subsequent chunked ``lm_head(torch.Tensor)`` call then fails with mixed
Tensor/DTensor operands.

This file therefore computes the small pause-plan head gradients on detached
leaf tensors, merges them into the chunked CE gradients, and performs exactly
one backward traversal through the transformer graph.
"""

from __future__ import annotations

import functools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

PRETRAIN_DIR = Path(__file__).resolve().parents[1]
if str(PRETRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAIN_DIR))

from recipes import train_qwen3 as base  # noqa: E402
from onerec_llm.data.dataloaders_pause import get_dataloader as pause_get_dataloader  # noqa: E402
from onerec_llm.data.pause_plan_utils import (  # noqa: E402
    collect_pause_plan_pairs_with_positions,
)

_ORIGINAL_GET_ARGUMENT_PARSER = base.get_argument_parser
_ORIGINAL_COMPUTE_FORWARD_BACKWARD = base.compute_forward_backward
_ORIGINAL_LOG_TRAINING_STEP = base.log_training_step


@dataclass
class _PauseMetricTracker:
    aux_sum: Optional[torch.Tensor] = None
    matched_sum: Optional[torch.Tensor] = None
    steps: int = 0

    def update(self, aux_loss: torch.Tensor, matched: int) -> None:
        aux = aux_loss.detach().float()
        matched_tensor = torch.tensor(float(matched), device=aux.device)
        self.aux_sum = aux if self.aux_sum is None else self.aux_sum + aux
        self.matched_sum = (
            matched_tensor
            if self.matched_sum is None
            else self.matched_sum + matched_tensor
        )
        self.steps += 1

    def reduce(self) -> Optional[Dict[str, float]]:
        if self.steps == 0 or self.aux_sum is None or self.matched_sum is None:
            return None
        payload = torch.stack(
            [
                self.aux_sum,
                self.matched_sum,
                torch.tensor(float(self.steps), device=self.aux_sum.device),
            ]
        )
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        steps = max(payload[2].item(), 1.0)
        return {
            "training/pause_plan_aux_loss": payload[0].item() / steps,
            "training/pause_plan_pairs_per_rank_step": payload[1].item() / steps,
        }

    def reset(self) -> None:
        self.aux_sum = None
        self.matched_sum = None
        self.steps = 0


@dataclass
class _PauseAuxGradients:
    """Detached auxiliary loss values plus gradients to merge into main CE."""

    raw_loss: torch.Tensor
    weighted_loss: torch.Tensor
    positions: torch.Tensor
    state_grad: Optional[torch.Tensor]
    weight_grad: Optional[torch.Tensor]
    bias_grad: Optional[torch.Tensor]
    itemic_start: int
    itemic_end: int
    matched_pairs: int


_PAUSE_METRICS = _PauseMetricTracker()


def get_argument_parser():
    parser = _ORIGINAL_GET_ARGUMENT_PARSER()
    group = parser.add_argument_group("latent pause-prefix")
    group.add_argument(
        "--pause_plan_aux_weight",
        type=float,
        default=0.0,
        help=(
            "Optional CE weight that maps late pause states to future Itemic "
            "tokens. Set 0 for the simplest PauseRec-style SFT."
        ),
    )
    group.add_argument(
        "--pause_plan_temperature",
        type=float,
        default=1.0,
        help="Temperature for the optional Itemic hierarchy planning loss.",
    )
    group.add_argument(
        "--pause_plan_warmup_steps",
        type=int,
        default=200,
        help="Linear warmup steps for the optional planning loss.",
    )
    group.add_argument(
        "--pause_plan_max_pairs",
        type=int,
        default=64,
        help="Maximum pause-to-Itemic supervision pairs per rank and step.",
    )
    group.add_argument(
        "--pause_itemic_start",
        type=int,
        default=151669,
        help="Inclusive first Itemic token ID.",
    )
    group.add_argument(
        "--pause_itemic_end",
        type=int,
        default=169742,
        help="Inclusive last Itemic token ID.",
    )
    return parser


def _make_labels(input_ids, loss_mask, ignore_index):
    pad = torch.full(
        (input_ids.shape[0], 1),
        ignore_index,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    shifted = torch.cat([input_ids[:, 1:], pad], dim=-1)
    return shifted * loss_mask + ignore_index * (1 - loss_mask)


def _get_pause_token_id(model) -> int:
    token_id = getattr(model.config, "pause_token_id", None)
    if token_id is None:
        token_ids = getattr(model.config, "pause_token_ids", None)
        if token_ids:
            token_id = token_ids[0]
    if token_id is None:
        raise ValueError(
            "config.pause_token_id is missing. Run tools/prepare_pause_model.py "
            "and train from the prepared model directory."
        )
    return int(token_id)


def _unwrap_chunked_loss_computer(compute_loss_fn):
    """Return the ChunkedLossComputer instance behind a bound/partial method."""

    fn = compute_loss_fn
    while isinstance(fn, functools.partial):
        fn = fn.func
    computer = getattr(fn, "__self__", None)
    required = ("lm_head", "loss_fn", "minibatch_size", "shift_labels", "ticker")
    if computer is None or any(not hasattr(computer, name) for name in required):
        raise TypeError(
            "pause_plan_aux_weight > 0 requires compute_loss_fn to be the bound "
            "ChunkedLossComputer.forward_and_backward method."
        )
    return computer


def _empty_aux_gradients(hidden_states: torch.Tensor) -> _PauseAuxGradients:
    zero = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
    return _PauseAuxGradients(
        raw_loss=zero,
        weighted_loss=zero,
        positions=torch.empty((0, 2), dtype=torch.long, device=hidden_states.device),
        state_grad=None,
        weight_grad=None,
        bias_grad=None,
        itemic_start=0,
        itemic_end=-1,
        matched_pairs=0,
    )


def _compute_pause_aux_gradients(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    sample_idx: Optional[torch.Tensor],
    args,
    aux_scale: float,
) -> _PauseAuxGradients:
    """Compute pause-plan gradients without backpropagating through FSDP yet.

    The selected hidden states and the relevant lm_head weight slice are detached
    into leaf tensors. Autograd therefore only differentiates the tiny auxiliary
    head computation and cannot trigger the root FSDP module's post-backward
    reshard hook. The resulting gradients are merged into the main chunked CE
    gradients later, immediately before the single transformer backward call.
    """

    pause_token_id = _get_pause_token_id(model)
    positions, target_ids = collect_pause_plan_pairs_with_positions(
        hidden_states=hidden_states,
        input_ids=input_ids,
        sample_idx=sample_idx,
        pause_token_id=pause_token_id,
        itemic_start=int(args.pause_itemic_start),
        itemic_end=int(args.pause_itemic_end),
        max_pairs=int(args.pause_plan_max_pairs),
    )
    if target_ids.numel() == 0:
        return _empty_aux_gradients(hidden_states)

    itemic_start = int(args.pause_itemic_start)
    itemic_end = int(args.pause_itemic_end)
    if itemic_start < 0 or itemic_end < itemic_start:
        raise ValueError("Invalid pause Itemic token range")

    lm_head = model.lm_head
    if not hasattr(lm_head, "weight") or lm_head.weight.ndim != 2:
        raise TypeError("The pause planning auxiliary loss requires a linear lm_head")
    if itemic_end >= lm_head.weight.shape[0]:
        raise ValueError(
            f"pause_itemic_end={itemic_end} exceeds lm_head vocab size "
            f"{lm_head.weight.shape[0]}"
        )

    # IMPORTANT: do not call weighted_aux_loss.backward() here. The selected
    # tensors are detached leaves so this local autograd.grad cannot walk into
    # the FSDP-wrapped transformer graph or trigger its post-backward reshard.
    selected_states = hidden_states[
        positions[:, 0], positions[:, 1]
    ].detach().requires_grad_(True)

    weight_requires_grad = bool(lm_head.weight.requires_grad)
    weight_leaf = (
        lm_head.weight[itemic_start : itemic_end + 1]
        .detach()
        .requires_grad_(weight_requires_grad)
    )

    bias_param = getattr(lm_head, "bias", None)
    bias_requires_grad = bias_param is not None and bool(bias_param.requires_grad)
    bias_leaf = None
    if bias_param is not None:
        bias_leaf = (
            bias_param[itemic_start : itemic_end + 1]
            .detach()
            .requires_grad_(bias_requires_grad)
        )

    logits = F.linear(selected_states, weight_leaf, bias_leaf).float()
    temperature = float(args.pause_plan_temperature)
    if temperature <= 0:
        raise ValueError("--pause_plan_temperature must be positive")
    logits = logits / temperature
    targets = target_ids.long() - itemic_start
    raw_aux_loss = F.cross_entropy(logits, targets)
    weighted_aux_loss = raw_aux_loss * float(aux_scale)

    grad_inputs = [selected_states]
    if weight_requires_grad:
        grad_inputs.append(weight_leaf)
    if bias_requires_grad and bias_leaf is not None:
        grad_inputs.append(bias_leaf)

    grads = torch.autograd.grad(
        weighted_aux_loss,
        grad_inputs,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    grad_index = 0
    state_grad = grads[grad_index]
    grad_index += 1
    weight_grad = None
    bias_grad = None
    if weight_requires_grad:
        weight_grad = grads[grad_index]
        grad_index += 1
    if bias_requires_grad:
        bias_grad = grads[grad_index]

    return _PauseAuxGradients(
        raw_loss=raw_aux_loss.detach(),
        weighted_loss=weighted_aux_loss.detach(),
        positions=positions,
        state_grad=state_grad.detach(),
        weight_grad=None if weight_grad is None else weight_grad.detach(),
        bias_grad=None if bias_grad is None else bias_grad.detach(),
        itemic_start=itemic_start,
        itemic_end=itemic_end,
        matched_pairs=int(target_ids.numel()),
    )


def _chunked_forward_backward_with_pause_aux(
    computer,
    input: torch.Tensor,
    labels: torch.Tensor,
    aux: _PauseAuxGradients,
    loss_fn_args: Optional[dict] = None,
):
    """OpenOneRec chunked CE with pause-plan gradients merged before backward.

    This mirrors ``ChunkedLossComputer.forward_and_backward`` from
    OpenOneRec-improved, with two additions:

    1. the auxiliary lm_head weight/bias slice gradients are added to the main
       CE lm_head gradients;
    2. the auxiliary selected-state gradients are scattered into the full
       hidden-state gradient.

    Only after both objectives have been merged do we call ``input.backward``.
    Hence FSDP2 sees one normal backward traversal and cannot reshard lm_head
    between the auxiliary objective and the main chunked CE.
    """

    if loss_fn_args is None:
        loss_fn_args = {}

    computer.ticker.tick("lm_head")
    params = list(computer.lm_head.parameters())
    grad_accs = [torch.zeros_like(p) for p in params]
    grad_input_full = torch.zeros_like(input)
    total_loss_sum_for_reporting = torch.tensor(0.0, device=input.device)
    all_per_token_losses = []

    seq_len = input.size(1)
    labels_to_count = labels[:, 1:] if computer.shift_labels else labels
    ignore_index = getattr(computer.loss_fn, "ignore_index", -100)
    total_elements = (labels_to_count != ignore_index).sum()
    has_main_ce = bool(total_elements.item() > 0)

    if has_main_ce:
        for i in range(0, seq_len, computer.minibatch_size):
            start, end = i, min(i + computer.minibatch_size, seq_len)
            input_chunk = input[:, start:end, :].detach().requires_grad_()
            logits_chunk = computer.lm_head(input_chunk)

            if computer.shift_labels:
                label_start, label_end = start + 1, end + 1
                labels_chunk = labels[:, label_start:label_end]
                if logits_chunk.size(1) > labels_chunk.size(1):
                    logits_chunk = logits_chunk[:, : labels_chunk.size(1), :]
            else:
                labels_chunk = labels[:, start:end]

            if labels_chunk.numel() == 0:
                continue

            logits_flat = logits_chunk.reshape(-1, computer.lm_head.out_features)
            labels_flat = labels_chunk.reshape(-1)
            loss_chunk_avg, per_token_loss_chunk = computer.loss_fn(
                logits_flat, labels_flat, **loss_fn_args
            )
            valid_tokens_in_chunk = (labels_flat != ignore_index).sum()
            if valid_tokens_in_chunk.item() == 0:
                all_per_token_losses.append(per_token_loss_chunk.detach())
                continue

            loss_chunk_sum = loss_chunk_avg * valid_tokens_in_chunk
            tensors_to_grad = [p for p in params if p.requires_grad] + [input_chunk]
            grads = torch.autograd.grad(
                outputs=loss_chunk_sum,
                inputs=tensors_to_grad,
                retain_graph=False,
            )

            grad_idx = 0
            for j, param in enumerate(params):
                if param.requires_grad:
                    grad_accs[j] += grads[grad_idx]
                    grad_idx += 1
            grad_input_full[:, start:end, :] = grads[grad_idx]
            total_loss_sum_for_reporting += loss_chunk_sum.detach()
            all_per_token_losses.append(per_token_loss_chunk.detach())

        # Convert main CE sum-gradients to mean-gradients, matching the original
        # ChunkedLossComputer behavior.
        grad_input_full.div_(total_elements)
        for j, param in enumerate(params):
            if param.requires_grad:
                grad_accs[j].div_(total_elements)

    # Merge the pause-plan state gradient at its original [batch, sequence]
    # locations. index_add also remains correct if future pairing logic ever
    # produces duplicate positions.
    if aux.state_grad is not None and aux.positions.numel() > 0:
        flat_positions = (
            aux.positions[:, 0].long() * input.shape[1]
            + aux.positions[:, 1].long()
        )
        grad_input_full.view(-1, input.shape[-1]).index_add_(
            0,
            flat_positions,
            aux.state_grad.to(dtype=grad_input_full.dtype),
        )

    # Add the auxiliary lm_head gradients without allocating a second full-vocab
    # gradient tensor. Only the Itemic rows supervised by the planner are touched.
    weight_param = computer.lm_head.weight
    bias_param = getattr(computer.lm_head, "bias", None)
    for j, param in enumerate(params):
        if not param.requires_grad:
            continue
        if param is weight_param and aux.weight_grad is not None:
            grad_accs[j][aux.itemic_start : aux.itemic_end + 1].add_(
                aux.weight_grad.to(dtype=grad_accs[j].dtype)
            )
        elif bias_param is not None and param is bias_param and aux.bias_grad is not None:
            grad_accs[j][aux.itemic_start : aux.itemic_end + 1].add_(
                aux.bias_grad.to(dtype=grad_accs[j].dtype)
            )
        param.grad = grad_accs[j]

    computer.ticker.tick("llm")
    if has_main_ce or (aux.state_grad is not None and aux.state_grad.numel() > 0):
        input.backward(gradient=grad_input_full)
    computer.ticker.tick("done")

    if has_main_ce:
        final_avg_loss = (total_loss_sum_for_reporting / total_elements).detach()
    else:
        final_avg_loss = torch.tensor(0.0, device=input.device)
    per_token_loss = (
        torch.cat(all_per_token_losses)
        if all_per_token_losses
        else torch.empty(0, device=input.device)
    )

    # Keep the same public behavior as the repository implementation.
    final_avg_loss.requires_grad = True
    computer.loss_info = {
        "loss": final_avg_loss,
        "per_token_loss": per_token_loss,
    }
    return final_avg_loss, per_token_loss


def compute_forward_backward(
    model: torch.nn.Module,
    batch: Dict,
    compute_loss_fn,
    loss_fn,
    args,
    embedding_masker,
    optimizer: torch.optim.Optimizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # With no optional auxiliary loss, preserve the repository's exact training
    # function and only change the data sequence.
    if float(args.pause_plan_aux_weight) <= 0:
        return _ORIGINAL_COMPUTE_FORWARD_BACKWARD(
            model,
            batch,
            compute_loss_fn,
            loss_fn,
            args,
            embedding_masker,
            optimizer,
        )

    if not args.use_chunked_loss_computer:
        raise ValueError(
            "The optional pause planning loss requires "
            "--use_chunked_loss_computer."
        )
    if bool(getattr(args, "reshard_after_forward", False)):
        raise ValueError(
            "OpenOneRec's ChunkedLossComputer requires root parameters to stay "
            "unsharded between model forward and its manual lm_head computation. "
            "Do not combine --reshard_after_forward with --use_chunked_loss_computer."
        )

    chunked_loss_computer = _unwrap_chunked_loss_computer(compute_loss_fn)
    if chunked_loss_computer.lm_head is not model.lm_head:
        raise RuntimeError(
            "ChunkedLossComputer.lm_head is not model.lm_head; cannot safely merge "
            "pause-plan gradients."
        )

    input_ids = batch["input_ids"]
    loss_mask = batch["loss_mask"]
    attention_mask = batch.get("attention_mask")
    cu_seqlens = batch.get("cu_seqlens")
    position_ids = batch.get("position_ids")
    sample_idx = batch.get("sample_idx")
    input_ids = input_ids * (input_ids > 0).to(torch.int64, non_blocking=True)

    with base.Timer("Fwd"):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            cu_seqlens=cu_seqlens,
            position_ids=position_ids,
        )
        hidden_states = output.logits
        labels = _make_labels(input_ids, loss_mask, loss_fn.ignore_index)

        update_step = int(getattr(args, "_pause_update_step", 0)) + 1
        args._pause_update_step = update_step
        warmup_steps = max(int(args.pause_plan_warmup_steps), 0)
        warmup = min(1.0, update_step / warmup_steps) if warmup_steps > 0 else 1.0
        aux_scale = warmup * float(args.pause_plan_aux_weight)

        aux = _compute_pause_aux_gradients(
            model=model,
            hidden_states=hidden_states,
            input_ids=input_ids,
            sample_idx=sample_idx,
            args=args,
            aux_scale=aux_scale,
        )

    with base.Timer("bwd"):
        # The crucial difference from the previous implementation is that there
        # is NO auxiliary backward here. Main CE and auxiliary gradients are
        # merged first, then the transformer graph is traversed exactly once.
        ce_loss, per_token_loss = _chunked_forward_backward_with_pause_aux(
            computer=chunked_loss_computer,
            input=hidden_states,
            labels=labels,
            aux=aux,
        )
        report_loss = ce_loss.detach() + aux.weighted_loss
        per_token_loss = per_token_loss.to(report_loss.device)

        if args.start_optimize_embedding_index > 0 and embedding_masker is not None:
            embedding_masker.apply_gradient_mask(optimizer)
        if args.max_grad_norm and args.max_grad_norm > 0:
            base.clip_grad_norm(model, args.max_grad_norm)

    _PAUSE_METRICS.update(aux.raw_loss, aux.matched_pairs)
    return report_loss, per_token_loss


def log_training_step(*args, **kwargs):
    global_step = kwargs.get("global_step", args[0] if args else 0)
    tb_logger = kwargs.get("tb_logger")
    if tb_logger is None and len(args) > 11:
        tb_logger = args[11]

    pause_metrics = _PAUSE_METRICS.reduce()
    end_time = _ORIGINAL_LOG_TRAINING_STEP(*args, **kwargs)
    if pause_metrics is not None and dist.get_rank() == 0:
        base.print_rank_0("Pause metrics:", base.format_dict_or_list(pause_metrics))
        writer = getattr(tb_logger, "tb_writer", None)
        if writer is not None:
            for name, value in pause_metrics.items():
                writer.add_scalar(name, value, global_step=global_step, new_style=True)
    _PAUSE_METRICS.reset()
    return end_time


# Narrow extension points; the distributed train loop/checkpointing remain original.
base.get_argument_parser = get_argument_parser
base.get_dataloader = pause_get_dataloader
base.compute_forward_backward = compute_forward_backward
base.log_training_step = log_training_step


if __name__ == "__main__":
    base.train()
