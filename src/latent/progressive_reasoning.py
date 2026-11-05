"""Progressive Reasoning Learning (ReaRec PRL) on the L7 latent-only bottleneck.

ReaRec (arXiv 2503.22675, official ``models/PRL.py``) runs K reasoning steps and
scores every step's output r_0..r_K against the items:

  final loss  CE(r_K E^T / tau)                                   (unchanged)
  PL loss     mean_{j=0..K-1} CE(r_j E^T / tau_j),  tau_j = tau * s^(K-j)

(s = ``temp_scale``, default 5: earlier steps are softer).  Here "scoring r_j" is
decoding the target title from step j's state:

  j >= 1   the target sees, as in the L7 bottleneck, only the fixed instruction
           prefix, thoughts t_1..t_j and earlier target tokens (thoughts j+1..K
           hidden, position ids as if the prompt had j thoughts); the first target
           token is predicted at thought j, which is where r_j sits.
  j = 0    r_0 is ReaRec's output at the last item, i.e. the first-pass state
           h_0 with the history visible: the target sees the whole prompt but no
           thought, and the first target token is predicted at the token before
           the first thought.
  j = K    the ordinary main pass (HF loss, temperature 1 = the LLM's tau).

Token logits of step j are divided by ``step_temperature(j)``: the paper's
``temp_scale ** (K - j)``, or, with ``max_temp`` set, ``max_temp ** ((K - j) / K)``.
The paper's schedule grows geometrically with K (K=4, s=5: 625/125/25/5); at the
LLM's tau = 1 a divisor above ~25 makes the softmax over the vocabulary uniform,
so those steps get no gradient and their loss cannot fall below ~ln V.  With
``max_temp`` the earliest step is fixed at ``max_temp`` for every K and the
ratios shrink instead (max_temp = 25, K=4: 25/11.2/5/2.24); K=2 with
max_temp = s^2 is identical to the paper.  Per step the loss
is the token-level CE (normalised like the HF loss: by ``num_items_in_batch``
under gradient accumulation, else the batch mean), averaged over steps (the
official code averages over the B*K step rows).  ``from_step`` = 1 drops j = 0.
Training only; evaluation and generation are untouched.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from transformers import TrainerCallback

from latent_bottleneck import bottleneck_mask_and_positions


def step_block(history_block: torch.Tensor, thought_pos: torch.Tensor, n_thoughts: int, step: int) -> torch.Tensor:
    """Keys hidden from target-side queries for PRL step ``step``: the bottleneck's
    history block (steps >= 1) plus thoughts ``step+1..K`` (1-based)."""
    length = history_block.size(1)
    k_abs = torch.arange(length, device=history_block.device).unsqueeze(0)
    start = thought_pos.unsqueeze(1)
    later_thoughts = (k_abs >= start + step) & (k_abs < start + n_thoughts)
    base = history_block if step >= 1 else torch.zeros_like(history_block)
    return base | later_thoughts


def step_sources(labels: torch.Tensor, thought_pos: torch.Tensor, n_thoughts: int, step: int):
    """``(rows, src, tgt)``: label position ``tgt`` is predicted from hidden state
    ``src``; normally ``tgt - 1``, but the first target token (right after the K
    thoughts) is predicted at thought ``step`` (``step = 0``: the last prompt token
    before the thoughts)."""
    shifted = labels[:, 1:]
    rows, tgt = torch.nonzero(shifted.ne(-100), as_tuple=True)
    tgt = tgt + 1
    src = tgt - 1
    first = thought_pos[rows] + n_thoughts
    src = torch.where(tgt == first, thought_pos[rows] + step - 1, src)
    return rows, src, tgt


def step_logits(model, inputs_embeds, attention_mask, labels, history_block, thought_pos, n_thoughts, step):
    """Untempered logits ``[N, V]`` for every label token of PRL step ``step`` and
    their targets ``[N]``."""
    block = step_block(history_block, thought_pos, n_thoughts, step)
    mask, positions = bottleneck_mask_and_positions(attention_mask, block, thought_pos,
                                                    attention_mask.size(1), inputs_embeds.dtype)
    hidden = model.model(inputs_embeds=inputs_embeds, attention_mask=mask, position_ids=positions,
                         use_cache=False).last_hidden_state
    rows, src, tgt = step_sources(labels, thought_pos, n_thoughts, step)
    return model.lm_head(hidden[rows, src]).float(), labels[rows, tgt]


def step_loss(model, inputs_embeds, attention_mask, labels, history_block, thought_pos, n_thoughts, step,
              temperature: float, num_items_in_batch: Optional[int] = None):
    """Target CE of PRL step ``step`` (``step = n_thoughts`` reproduces the main
    pass).  Returns ``(loss normalised like the HF loss, per-token mean for logging)``."""
    logits, targets = step_logits(model, inputs_embeds, attention_mask, labels, history_block, thought_pos,
                                  n_thoughts, step)
    loss = F.cross_entropy(logits / temperature, targets, reduction="sum")
    denom = num_items_in_batch if num_items_in_batch is not None else targets.numel()
    if torch.is_tensor(denom):
        denom = denom.to(loss.device)
    return loss / denom, loss.detach() / max(targets.numel(), 1)


def step_temperature(step: int, n_thoughts: int, spec: dict) -> float:
    """Divisor of step ``step``'s logits (see the module docstring)."""
    max_temp = spec.get("max_temp")
    if max_temp is not None:
        return float(max_temp) ** ((n_thoughts - step) / n_thoughts)
    return float(spec.get("temp_scale", 5.0)) ** (n_thoughts - step)


def prl_loss(model, inputs_embeds, attention_mask, labels, spec: dict, num_items_in_batch=None):
    """Mean over steps ``from_step..K-1`` of the temperature-annealed step losses.
    Returns ``(loss, {step: per-token mean CE of this micro-batch, for logging})``."""
    history_block = model._bottleneck_block
    thought_pos = model._bottleneck_thought_pos
    n_thoughts = model.thoughts.k
    steps = list(range(int(spec.get("from_step", 0)), n_thoughts))
    if not steps:
        return None, {}
    losses, means = [], {}
    for j in steps:
        loss, mean = step_loss(model, inputs_embeds, attention_mask, labels, history_block, thought_pos,
                               n_thoughts, j, step_temperature(j, n_thoughts, spec), num_items_in_batch)
        losses.append(loss)
        means[j] = mean
    return torch.stack(losses).mean(), means


class PRLLogger(TrainerCallback):
    """Log the last micro-batch's PRL step losses (per-token mean CE, temperature
    applied) and contrastive loss next to the Trainer's loss: stdout + wandb."""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        stats = getattr(model, "last_prl_stats", None) if model is not None else None
        if not stats or not logs or "loss" not in logs:
            return
        record = {f"prl/{k}": v for k, v in stats.items()}
        if state.global_step <= 3 or state.global_step % 10 == 0:
            mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            print(f"[PRL] step {state.global_step}: loss={logs['loss']:.4f} "
                  + " ".join(f"{k}={v:.4f}" for k, v in record.items()) + f" peak_mem={mem:.1f}GB", flush=True)
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(record, step=state.global_step)
        except ImportError:
            pass
