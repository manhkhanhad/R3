"""Coconut-style continuous thoughts for LatentModel (latent-utilization line).

Plain R3 attention-pools the history into one vector with a learned query; in
every original checkpoint that query collapses onto a few constant instruction
tokens, so ``r`` is a learned task bias (research log U2/U3).  Here the prompt
ends in K ``<|Thought|>`` tokens and the thoughts are produced the way Coconut
produces continuous thoughts:

  h_0      = last hidden state at the token before the first thought
             (prompt prefix pass, causal, full history visible)
  t_k      = project(h_{k-1})            fed back as the k-th thought's input
  h_k      = last hidden state at thought k (KV cache over prefix + t_1..t_k)

``project`` is an fp32 linear map (identity init) followed by rescaling to the
mean input-token-embedding norm, so the thought sits at token scale.  The
thoughts are then spliced into the K thought positions for the recommendation
pass; with ``latent_bottleneck`` the target tokens see only the instruction
prefix, the thoughts and earlier target tokens.

``source="constant"`` replaces the thoughts with K learned vectors (baseline C:
"R3 is a learned task bias" made explicit).

Config: ``config.latent_continuous_thoughts = {"k": K, "source": "hidden" |
"constant", "token_norm": float}``.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback


class ContinuousThoughts(nn.Module):
    def __init__(self, hidden_size: int, spec: dict):
        super().__init__()
        self.k = int(spec["k"])
        self.source = spec.get("source", "hidden")
        if self.k < 1:
            raise ValueError("continuous thoughts need k >= 1")
        self.register_buffer("token_norm", torch.tensor(float(spec.get("token_norm", 1.0))), persistent=False)
        if self.source == "hidden":
            self.proj = nn.Linear(hidden_size, hidden_size)
        elif self.source == "constant":
            self.constant = nn.Parameter(torch.empty(self.k, hidden_size))
        else:
            raise ValueError(f"unknown continuous-thought source {self.source}")
        self.reset_parameters()
        # Interventions at evaluation: a callable [B, K, H] -> [B, K, H] (see latent_attention_eval.py).
        self.override = None
        self.last_thoughts: Optional[torch.Tensor] = None

    def reset_parameters(self) -> None:
        """Identity projection / random constant, in fp32.  Call again after
        ``from_pretrained`` on a base model: HF re-initialises missing weights."""
        self.float()
        with torch.no_grad():
            if self.source == "hidden":
                self.proj.weight.copy_(torch.eye(self.proj.in_features))
                self.proj.bias.zero_()
            else:
                self.constant.normal_(0.0, 1.0)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.proj(hidden.to(self.proj.weight.dtype))
        return F.normalize(x.float(), dim=-1) * self.token_norm

    def forward(self, backbone, input_ids, attention_mask, first_thought_pos):
        """Return thought input embeddings ``[B, K, H]`` (fp32)."""
        batch = input_ids.size(0)
        if self.source == "constant":
            thoughts = F.normalize(self.constant.float(), dim=-1) * self.token_norm
            thoughts = thoughts.unsqueeze(0).expand(batch, -1, -1)
        else:
            thoughts = self._recurrent(backbone, input_ids, attention_mask, first_thought_pos)
        if self.override is not None:
            thoughts = self.override(thoughts)
        self.last_thoughts = thoughts.detach()
        return thoughts

    def _recurrent(self, backbone, input_ids, attention_mask, first_thought_pos):
        # Left-align each row's real prompt tokens before its first thought so the
        # K thought steps can be appended to one shared KV cache (training batches
        # are right-padded, generation batches left-padded).
        batch, device = input_ids.size(0), input_ids.device
        rows = []
        for b in range(batch):
            keep = attention_mask[b, : first_thought_pos[b]].bool()
            rows.append(input_ids[b, : first_thought_pos[b]][keep])
        width = max(r.numel() for r in rows)
        prefix = input_ids.new_zeros(batch, width)
        prefix_mask = attention_mask.new_zeros(batch, width)
        for b, r in enumerate(rows):
            prefix[b, width - r.numel():] = r
            prefix_mask[b, width - r.numel():] = 1
        lengths = prefix_mask.sum(1)
        # Original position of every aligned slot (-1 = pad), for steered_reading.py.
        positions = torch.full((batch, width), -1, dtype=torch.long, device=device)
        for b, r in enumerate(rows):
            positions[b, width - r.numel():] = torch.nonzero(attention_mask[b, : first_thought_pos[b]].bool()).squeeze(1)
        self.last_prefix_positions = positions
        out = backbone(input_ids=prefix, attention_mask=prefix_mask,
                       position_ids=(prefix_mask.long().cumsum(1) - 1).clamp(min=0), use_cache=True)
        self.last_prefix_hidden = out.last_hidden_state
        hidden, cache = out.last_hidden_state[:, -1], out.past_key_values
        thoughts, mask = [], prefix_mask
        embed_dtype = backbone.embed_tokens.weight.dtype
        for k in range(self.k):
            t = self.project(hidden)
            thoughts.append(t)
            if k == self.k - 1:
                break
            mask = torch.cat([mask, mask.new_ones(batch, 1)], dim=1)
            out = backbone(inputs_embeds=t.to(embed_dtype).unsqueeze(1), attention_mask=mask,
                           position_ids=(lengths + k).unsqueeze(1), past_key_values=cache, use_cache=True)
            hidden, cache = out.last_hidden_state[:, -1], out.past_key_values
        return torch.stack(thoughts, dim=1)


def first_thought_positions(input_ids: torch.Tensor, thought_token_id: int, k: int) -> torch.Tensor:
    """Position of the first of exactly ``k`` contiguous thought tokens in every row."""
    is_thought = input_ids == thought_token_id
    counts = is_thought.sum(1)
    if not torch.all(counts == k):
        raise ValueError(f"expected {k} thought tokens per row, got {counts.tolist()}")
    first = is_thought.long().argmax(1)
    span = torch.arange(k, device=input_ids.device)
    if not torch.all(input_ids.gather(1, first[:, None] + span[None]) == thought_token_id):
        raise ValueError("thought tokens must be contiguous")
    return first


def load_fp32_thought_params(model, checkpoint: str) -> None:
    """``from_pretrained(torch_dtype=bf16)`` rounds every weight to bf16; reload the
    thought module from the checkpoint in fp32 (it is trained and saved in fp32)."""
    import os
    from safetensors import safe_open

    files = [f for f in os.listdir(checkpoint) if f.endswith(".safetensors")] if os.path.isdir(checkpoint) else []
    model.thoughts.float()
    loaded = 0
    for name in files:
        with safe_open(os.path.join(checkpoint, name), "pt") as f:
            for key in f.keys():
                if key.startswith("thoughts."):
                    param = dict(model.thoughts.named_parameters())[key[len("thoughts."):]]
                    with torch.no_grad():
                        param.copy_(f.get_tensor(key).float())
                    loaded += 1
    if files and loaded == 0:
        raise RuntimeError(f"no thought parameters found in {checkpoint}")


class ThoughtDiagnostics(TrainerCallback):
    """TrainerCallback-compatible probe run at every evaluation on a fixed set of
    validation rows: is the thought sample-specific, and does the decoder use it?

    Logs (and appends to ``<output_dir>/thought_diagnostics.jsonl``):
      residual_share  mean ||t - mean_t|| / ||t|| over rows (K thoughts flattened)
      shuffle_dnll    target NLL with another row's thoughts minus raw NLL
      mean_dnll       target NLL with the mean thoughts minus raw NLL
    """

    def __init__(self, rows: list, collator, output_dir: str):
        self.batch = collator(rows)
        self.path = f"{output_dir}/thought_diagnostics.jsonl"

    def _nll(self, model, batch, override=None):
        model.thoughts.override = override
        try:
            extra = {k: batch[k] for k in ("item_index", "item_sim") if k in batch}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], **extra).logits
        finally:
            model.thoughts.override = None
        labels = batch["labels"][:, 1:]
        loss = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
        valid = labels.ne(-100)
        return (loss * valid).sum(1) / valid.sum(1).clamp_min(1)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        import json
        if model is None or not hasattr(model, "thoughts"):
            return
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        n, step = self.batch["input_ids"].size(0), 4  # full-vocabulary logits: keep chunks small
        keys = [k for k in ("input_ids", "attention_mask", "labels", "item_index", "item_sim") if k in self.batch]
        chunks = [{k: self.batch[k][i:i + step].to(device) for k in keys} for i in range(0, n, step)]
        steer = getattr(model, "steer", None)
        with torch.no_grad():
            raw, blocks, stats = [], [], []
            for c in chunks:
                raw.append(self._nll(model, c))
                blocks.append(model.thoughts.last_thoughts)
                if steer is not None:
                    stats.append(dict(steer.last_stats, rows=int(c["input_ids"].size(0))))
            raw, blocks = torch.cat(raw), torch.cat(blocks)
            t = blocks.flatten(1).float()
            share = ((t - t.mean(0)).norm(dim=1) / t.norm(dim=1).clamp_min(1e-12)).mean()
            variants = {"shuffle": blocks.roll(1, dims=0), "mean": blocks.mean(0, keepdim=True).expand_as(blocks)}
            nll = {}
            for name, replacement in variants.items():
                nll[name] = torch.cat([
                    self._nll(model, c, lambda x, r=replacement[i * step:(i + 1) * step]: r.to(x))
                    for i, c in enumerate(chunks)])
        shuffled, mean = nll["shuffle"], nll["mean"]
        record = {"step": state.global_step, "epoch": state.epoch, "raw_nll": float(raw.mean()),
                  "residual_share": float(share), "shuffle_dnll": float((shuffled - raw).mean()),
                  "mean_dnll": float((mean - raw).mean())}
        for key in ("rows_selecting", "same_as_recent", "same_as_oracle", "oracle_recall"):
            have = [s for s in stats if key in s]
            if have:  # row-weighted over chunks (approximate for the long-row statistics)
                record[f"select_{key}"] = sum(s[key] * s["rows"] for s in have) / sum(s["rows"] for s in have)
        print("thought diagnostics:", json.dumps(record), flush=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({f"thoughts/{k}": v for k, v in record.items() if k not in ("step", "epoch")},
                          step=state.global_step)
        except ImportError:
            pass
        if was_training:
            model.train()
