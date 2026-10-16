"""Collaborative multi-positive InfoNCE added to R3's stage-2 noise-GRPO.

Stage 2 trains only the latent producer (``model.attention``); the LLM is frozen.
Here the collaborative InfoNCE of Attempt 10 / L8 (relative NLL gate, diversity-aware
queue, user and target projection heads) is added on the same latent ``r``:

    loss = GRPO loss + lambda * (L_user + L_target)

* Anchor: the noise-free copy of each GRPO group (row 0 of the micro-batch; the
  trainer adds exploration noise to every other row).  Noisy copies never enter the
  queue: they are exploration, and 8 copies of one sample would be fake positives.
* Gate NLL: from the noise-free row's target log-probs, already computed by the trainer.
* Queue: the queue logic of ``CollaborativeContrastiveLatentModel`` is reused as a
  stand-alone module (no LLM inside) attached to the model as ``model.collab`` so the
  projection heads are trained and saved.  It is prefilled once from the starting
  checkpoint before GRPO.
* lambda: calibrated over the first ``calibration_steps`` optimizer steps with lambda=0:
  gradient norms of both losses on ``model.attention`` are recorded per micro-batch, and
  lambda is set so the InfoNCE gradient is ``target_ratio`` of the GRPO gradient
  (ratio of medians over micro-batches where InfoNCE is active).
"""

import os
import sys

import torch
import torch.nn as nn
from safetensors import safe_open

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), "latent"))

from collaborative_contrastive_attention import CollaborativeContrastiveLatentModel  # noqa: E402
from collaborative_contrastive_dataset import stable_hash  # noqa: E402
from grpo_trainer import NoiseGRPORecTrainer  # noqa: E402
from latent_grpo_dataset import D3Dataset  # noqa: E402


def build_collab_module(hidden, projection_dim=256, bank_size=10_000, nll_gate_window=1024,
                        checkpoint=None):
    """Queue + projection heads of the collaborative model, without the LLM."""
    module = object.__new__(CollaborativeContrastiveLatentModel)
    nn.Module.__init__(module)
    module.bank_size = bank_size
    module.title_feature_dim = 0
    module.cf_topk = 0
    module.user_projection = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, projection_dim, bias=False))
    module.target_projection = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, projection_dim, bias=False))
    buffers = {
        "bank_r": torch.zeros(bank_size, hidden, dtype=torch.float32),
        "bank_user": torch.zeros(bank_size, dtype=torch.int64),
        "bank_target": torch.zeros(bank_size, dtype=torch.int64),
        "bank_sample": torch.zeros(bank_size, dtype=torch.int64),
        "bank_nll": torch.zeros(bank_size, dtype=torch.float32),
        "bank_pointer": torch.zeros((), dtype=torch.int64),
        "bank_count": torch.zeros((), dtype=torch.int64),
        "bank_time": torch.zeros(bank_size, dtype=torch.int64),
        "bank_valid": torch.zeros(bank_size, dtype=torch.bool),
        "bank_clock": torch.zeros((), dtype=torch.int64),
        "nll_history": torch.zeros(nll_gate_window, dtype=torch.float32),
        "nll_history_pointer": torch.zeros((), dtype=torch.int64),
        "nll_history_count": torch.zeros((), dtype=torch.int64),
    }
    for name, value in buffers.items():
        module.register_buffer(name, value, persistent=False)
    module.nll_gate_window = nll_gate_window
    for name, value in dict(
        temperature=0.1, max_user_positives=16, max_target_positives=16, num_negatives=256,
        positive_min_token_accuracy=0.0, positive_max_nll=float("inf"), nll_gate_quantile=0.0,
        bank_max_per_user=0, bank_max_per_target=0, bank_dedup_cos=0.0, grad_accum_steps=1,
        last_queue_replaced=None,
    ).items():
        setattr(module, name, value)
    if checkpoint:
        load_projection_heads(module, checkpoint)
    return module


def load_projection_heads(module, checkpoint):
    """Copy ``user_projection.*`` / ``target_projection.*`` from a stage-1 collaborative checkpoint."""
    path = os.path.join(checkpoint, "model.safetensors")
    state = {}
    with safe_open(path, "pt") as handle:
        for key in handle.keys():
            if key.startswith(("user_projection.", "target_projection.")):
                state[key] = handle.get_tensor(key)
    missing = [k for k in ("user_projection.1.weight", "target_projection.1.weight") if k not in state]
    if missing:
        raise KeyError(f"{checkpoint} has no collaborative projection heads: {missing}")
    module.load_state_dict(state, strict=False)
    print(f"collab: loaded {len(state)} projection-head tensors from {checkpoint}", flush=True)


class D3CollabDataset(D3Dataset):
    """GRPO dataset rows with the identifiers the collaborative queue needs."""

    def pre(self, idx):
        item = super().pre(idx)
        if not self.test:
            row = self.data.iloc[idx]
            item["user_hash"] = stable_hash(row["user_id"])
            item["target_hash"] = stable_hash(row["item_id"])
            item["sample_hash"] = int(idx)
        return item


def thought_latents(model, input_ids, attention_mask):
    """``(embeds, r)``: input embeddings with ``r`` spliced in, and ``r`` itself."""
    embeds = model.generate_embs(input_ids, attention_mask)
    thought = torch.nonzero(input_ids == model.model.embed_tokens.num_embeddings - 1)
    rows = torch.arange(input_ids.size(0), device=input_ids.device)
    return embeds, embeds[rows, thought[:, 1]], thought


def target_nll(per_token_logps, completion_mask):
    mask = completion_mask.float()
    return -(per_token_logps.float() * mask).sum(-1) / mask.sum(-1).clamp(min=1)


@torch.no_grad()
def prefill_queue(model, dataset, collator, batch_size, device, log_every=4096):
    """Fill ``model.collab`` from the starting checkpoint (noise-free latents, gated as in training)."""
    collab = model.collab
    model.eval()
    seen = 0
    for start in range(0, len(dataset), batch_size):
        batch = collator([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])
        batch = {k: v.to(device) for k, v in batch.items()}
        embeds, r, _ = thought_latents(model, batch["input_ids"], batch["attention_mask"])
        logits = model(inputs_embeds=embeds, attention_mask=batch["attention_mask"]).logits[:, :-1].float()
        labels = batch["labels"][:, 1:]
        mask = labels != -100
        token_nll = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), labels.clamp(min=0), reduction="none")
        nll = (token_nll * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        quality = collab._nll_gate(nll)
        collab._push_bank(r.float(), batch["user_hash"], batch["target_hash"], batch["sample_hash"], nll, quality)
        seen += r.size(0)
        if seen % log_every < batch_size:
            print(f"collab prefill: seen={seen}, queue={int(collab.bank_count)}", flush=True)
    model.train()
    print(f"collab prefill done: seen={seen}, queue={int(collab.bank_count)}, "
          f"gate threshold={float(collab.last_nll_gate_threshold):.4f}", flush=True)


class CollaborativeGRPOTrainer(NoiseGRPORecTrainer):
    def __init__(self, *args, calibration_steps=20, target_ratio=0.3, min_active_micro_batches=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.calibration_steps = calibration_steps
        self.target_ratio = target_ratio
        self.min_active_micro_batches = min_active_micro_batches
        self.collab_lambda = None if calibration_steps > 0 else target_ratio
        self._grad_norms = {"grpo": [], "infonce": []}
        self._calibration_wait_logged = False

    def _maybe_finish_calibration(self):
        """Calibrate lambda once ``calibration_steps`` have passed AND enough micro-batches
        had an active InfoNCE term.  Until then lambda stays 0: anchors without positives
        contribute zero loss (as in stage 1) and training is pure GRPO."""
        if self.collab_lambda is not None or self.state.global_step < self.calibration_steps:
            return
        if len(self._grad_norms["infonce"]) >= self.min_active_micro_batches:
            self._finish_calibration()
        elif not self._calibration_wait_logged:
            print(f"collab: only {len(self._grad_norms['infonce'])} micro-batches with an active InfoNCE "
                  f"term after {self.state.global_step} steps (need {self.min_active_micro_batches}); "
                  "lambda stays 0 (pure GRPO) until enough positives appear", flush=True)
            self._calibration_wait_logged = True

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        for key in ("user_hash", "target_hash", "sample_hash"):
            out[key] = inputs[key].to(self.accelerator.device)
        return out

    def _infonce(self, model, r, inputs):
        """InfoNCE on the noise-free row(s); pushes them to the queue."""
        collab = self.model.collab
        clean = (inputs["noise"].abs().sum(-1) == 0)
        if not clean.any():
            return r.sum() * 0.0, None
        nll = target_nll(inputs["old_per_token_logps"][clean], inputs["completion_mask"][clean])
        quality = collab._nll_gate(nll)
        anchor = r[clean].float()
        user, target, sample = (inputs[k][clean] for k in ("user_hash", "target_hash", "sample_hash"))
        user_loss, target_loss, _, user_rate, target_rate, _ = collab._collaborative_losses(
            anchor, user, target, sample, quality)
        collab._push_bank(anchor, user, target, sample, nll, quality)
        stats = {"user_infonce": user_loss.item(), "target_infonce": target_loss.item(),
                 "user_positive_rate": float(user_rate), "target_positive_rate": float(target_rate),
                 "queue_count": float(collab.bank_count), "queue_accept": quality.float().mean().item(),
                 "nll_gate_threshold": float(collab.last_nll_gate_threshold),
                 "queue_user_diversity": float(collab._queue_user_diversity())}
        return user_loss + target_loss, stats

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Same as NoiseGRPORecTrainer.compute_loss, keeping r for the collaborative term.
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        embeds, r, thought = thought_latents(self.model, input_ids, attention_mask)
        embeds = embeds.clone()
        embeds[torch.arange(input_ids.size(0)), thought[:, 1]] += inputs["noise"]
        per_token_logps = self.my_get_per_token_logps(model, input_ids, embeds, attention_mask, logits_to_keep)

        if self.beta != 0.0:
            ref = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref - per_token_logps) - (ref - per_token_logps) - 1
        advantages = inputs["advantages"]
        old = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        loss1, loss2 = coef_1 * advantages.unsqueeze(1), coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(loss1, loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        grpo_loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        infonce, stats = self._infonce(model, r, inputs)
        mode = "eval" if self.control.should_evaluate else "train"
        if stats is not None:
            for key, value in stats.items():
                self._metrics[mode][key].append(value)

        calibrating = self.collab_lambda is None
        if calibrating and stats is not None and infonce.requires_grad and infonce.item() > 0:
            params = [p for p in self.model.attention.parameters() if p.requires_grad]
            g_grpo = torch.autograd.grad(grpo_loss, params, retain_graph=True, allow_unused=True)
            g_info = torch.autograd.grad(infonce, params, retain_graph=True, allow_unused=True)
            norm = lambda grads: torch.sqrt(sum((g.float() ** 2).sum() for g in grads if g is not None)).item()
            self._grad_norms["grpo"].append(norm(g_grpo))
            self._grad_norms["infonce"].append(norm(g_info))
        if calibrating:
            self._maybe_finish_calibration()

        weight = 0.0 if self.collab_lambda is None else self.collab_lambda
        self._metrics[mode]["collab_lambda"].append(weight)
        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        clip_ratio = ((loss1 < loss2).float() * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return grpo_loss + weight * infonce

    def _finish_calibration(self):
        grpo, info = self._grad_norms["grpo"], self._grad_norms["infonce"]
        if not info:
            raise RuntimeError("lambda calibration saw no micro-batch with an active InfoNCE term")
        med = lambda xs: float(torch.tensor(xs).median())
        self.collab_lambda = self.target_ratio * med(grpo) / max(med(info), 1e-12)
        print(f"collab lambda calibrated over {len(info)} micro-batches: median |g_grpo|={med(grpo):.3e}, "
              f"median |g_infonce|={med(info):.3e}, lambda={self.collab_lambda:.4e} "
              f"(target ratio {self.target_ratio})", flush=True)
