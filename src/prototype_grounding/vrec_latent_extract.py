#!/usr/bin/env python3
"""Extract the reasoning latent of R3 and VRec checkpoints with one shared procedure.

For the R3-vs-VRec comparison (document/vrec_vs_r3_latent_comparison.md). Each
checkpoint's own modules compute the latent and the sequence its decoder sees
(single reasoning step):

* ``r3``: r = pooling(h); the decoder reads [..., r, response].
* ``vrec_stage1`` (``LatentModel_MS``): r = pooling(h); VRec's reasoning loop inserts a
  fresh <|Thought|> embedding after the latent, so the decoder reads
  [..., r, <Thought>, response].
* ``vrec_verifier`` (stage 2 and 3, ``LatentModelwithVerifier_RATT_MS_MoV``):
  r = pooling(h); (w, g) = verify(r) with w the router-weighted normalized
  entropy and g the router-weighted argmax prototype; slot r* = (1 - w) r + w g;
  the decoder reads [..., r*, <Thought>, response].

Per example it saves r (``thoughts.pt``, fp16, the vector pooled from the
backbone), the slot actually fed to the decoder (``slot.pt``; equal to r for r3
and stage 1), the verifier's outputs (``verifier.pt``) and the mean NLL of the
target title with the true slot, a zero slot and a slot taken from another row of
the batch (``metadata.csv``: raw_nll, masked_nll, shuffled_nll, utility).

Every batch is checked against the model's own forward loss on the same inputs.

Output: analysis/evidence/<category>/<tag>/<split>/ (the layout `experiment.py
latent-distribution` reads).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, Qwen2ForCausalLM

import experiment as E

VREC_SRC = Path("/common/home/users/m/mk.ngohuu.2024/research/Verifiable-Rec/code/src")
KINDS = ("r3", "vrec_stage1", "vrec_verifier")


def vrec_class(stage_dir: str, name: str):
    """Import a VRec model class from its own stage directory (each has a layers.py)."""
    path = VREC_SRC / stage_dir / "layers.py"
    spec = importlib.util.spec_from_file_location(f"vrec_{stage_dir}_layers", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, name)


def load(kind: str, checkpoint: Path):
    if kind == "r3":
        model, tokenizer, device = E.load_model(checkpoint)
        return model, tokenizer, device
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    config = AutoConfig.from_pretrained(checkpoint)
    if int(getattr(config, "n_thought", 1)) != 1:
        raise ValueError("only single-step reasoning (n_thought = 1) is supported")
    cls = (vrec_class("pretrain_LLM", "LatentModel_MS") if kind == "vrec_stage1"
           else vrec_class("finetune", "LatentModelwithVerifier_RATT_MS_MoV"))
    model, info = cls.from_pretrained(checkpoint, config=config, torch_dtype=torch.bfloat16, output_loading_info=True)
    missing = [k for k in info["missing_keys"] if k != "lm_head.weight"]
    if missing:
        raise RuntimeError(f"missing keys when loading {checkpoint}: {missing}")
    model = model.to(device).eval()
    model.config.use_cache = False
    return model, tokenizer, device


def per_row_nll(logits, labels):
    """Mean NLL over each row's target tokens, and the token-weighted batch mean."""
    shift_logits, shift_labels = logits[:, :-1].float(), labels[:, 1:]
    tok = F.cross_entropy(shift_logits.transpose(1, 2), shift_labels.clamp_min(0), reduction="none")
    valid = (shift_labels != -100).float()
    per_row = (tok * valid).sum(1) / valid.sum(1).clamp_min(1)
    return per_row, float((tok * valid).sum() / valid.sum())


class Extractor:
    def __init__(self, kind, model, device):
        self.kind, self.model, self.device = kind, model, device
        self.thought_id = model.model.embed_tokens.num_embeddings - 1

    @torch.no_grad()
    def latent(self, ids, mask, pos):
        """Pooled r, the slot fed to the decoder, and the verifier outputs."""
        hidden = self.model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        r = self.model.attention(hidden_states=hidden, attention_mask=mask, thought_id_idx=pos)
        emb_dtype = self.model.model.embed_tokens.weight.dtype
        if self.kind != "vrec_verifier":
            return r.to(emb_dtype), r.to(emb_dtype), None
        r = r.to(emb_dtype)
        weight, guidance, entropy, gate = self.model.verify(r)
        w = weight.unsqueeze(1).to(emb_dtype)
        slot = (1 - w) * r + w * guidance
        probs = [F.softmax(v(r).float(), dim=-1) for v in self.model.verifiers]
        info = {"weight": weight.float(), "gate": gate.float(), "entropy": entropy.float().T,
                "argmax": torch.stack([p.argmax(-1) for p in probs], 1),
                "max_prob": torch.stack([p.max(-1).values for p in probs], 1)}
        return r, slot, info

    @torch.no_grad()
    def decode(self, ids, mask, labels, pos, slot):
        """Logits and labels of the decoder pass with ``slot`` at the thought position."""
        embeds = self.model.model.embed_tokens(ids).clone()
        rows = torch.arange(ids.size(0), device=ids.device)
        embeds[rows, pos] = slot.to(embeds.dtype)
        if self.kind != "r3":
            # VRec inserts a fresh <|Thought|> embedding after the latent (one reasoning step)
            thought = self.model.model.embed_tokens(torch.tensor([self.thought_id], device=ids.device))
            embeds = torch.stack([torch.cat([e[:p + 1], thought, e[p + 1:]]) for e, p in zip(embeds, pos)])
            mask = torch.stack([torch.cat([m[:p + 1], m.new_ones(1), m[p + 1:]]) for m, p in zip(mask, pos)])
            labels = torch.stack([torch.cat([l[:p + 1], l.new_full((1,), -100), l[p + 1:]]) for l, p in zip(labels, pos)])
        hidden = self.model.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False).last_hidden_state
        return self.model.lm_head(hidden), labels

    @torch.no_grad()
    def reference_loss(self, ids, mask, labels):
        """The model's own forward loss (same inputs), to validate this reimplementation."""
        out = self.model(input_ids=ids, attention_mask=mask, labels=labels)
        return float(out.loss)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=KINDS, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--split", choices=("train", "valid", "test"), required=True)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--check_batches", type=int, default=3, help="batches validated against the model's forward loss")
    a = p.parse_args()
    E.configure_category("Musical_Instruments")
    E.seed_all(42)
    model, tokenizer, device = load(a.kind, a.checkpoint)
    ex = Extractor(a.kind, model, device)
    ds = E.PromptDataset(a.split, tokenizer, a.limit)
    pad = tokenizer.pad_token_id
    out = E.ANALYSIS / "evidence" / E.CATEGORY / a.tag / a.split
    out.mkdir(parents=True, exist_ok=True)

    thoughts, slots, nll = [], [], {"raw_nll": [], "masked_nll": [], "shuffled_nll": []}
    verifier, meta, checks = {}, [], []
    start = time.time()
    for step, s in enumerate(range(0, len(ds), a.batch_size)):
        batch = E.collate(ds.rows[s:s + a.batch_size], pad)
        ids, mask, labels, pos = [batch[k].to(device) for k in ("input_ids", "attention_mask", "labels", "thought_pos")]
        r, slot, info = ex.latent(ids, mask, pos)
        variants = {"raw_nll": slot, "masked_nll": torch.zeros_like(slot), "shuffled_nll": slot.roll(1, 0)}
        for name, vec in variants.items():
            logits, lab = ex.decode(ids, mask, labels, pos, vec)
            per_row, batch_mean = per_row_nll(logits, lab)
            nll[name].append(per_row.cpu())
            if name == "raw_nll" and step < a.check_batches:
                ref = ex.reference_loss(ids, mask, labels)
                checks.append({"batch": step, "reimplemented": batch_mean, "model_forward": ref})
                if abs(batch_mean - ref) > 0.02 * max(1.0, abs(ref)):
                    raise RuntimeError(f"reimplemented loss {batch_mean:.4f} != model forward loss {ref:.4f}")
        thoughts.append(r.float().cpu().half())
        slots.append(slot.float().cpu().half())
        if info is not None:
            for k, v in info.items():
                verifier.setdefault(k, []).append(v.cpu())
        meta.extend(batch["meta"])
        if step % 200 == 0:
            print(f"[{a.tag}/{a.split}] {min(s + a.batch_size, len(ds))}/{len(ds)} {time.time() - start:.0f}s", flush=True)

    vec = torch.cat(thoughts)
    torch.save(vec, out / "thoughts.pt")
    torch.save(torch.cat(slots), out / "slot.pt")
    if verifier:
        torch.save({k: torch.cat(v) for k, v in verifier.items()}, out / "verifier.pt")
    frame = pd.DataFrame(meta).drop(columns=["history_item_ids"])
    for k, v in nll.items():
        frame[k] = torch.cat(v).numpy()
    frame["utility"] = frame.masked_nll - frame.raw_nll
    frame["content_utility"] = frame.shuffled_nll - frame.raw_nll
    frame.to_csv(out / "metadata.csv", index=False)
    E.atomic_json(out / "summary.json", {
        "kind": a.kind, "checkpoint": str(a.checkpoint), "split": a.split, "n": len(ds),
        "shape": list(vec.shape), "loss_checks": checks,
        "raw_nll": float(frame.raw_nll.mean()), "masked_nll": float(frame.masked_nll.mean()),
        "shuffled_nll": float(frame.shuffled_nll.mean()),
        "mean_utility": float(frame.utility.mean()), "mean_content_utility": float(frame.content_utility.mean()),
        "elapsed_seconds": time.time() - start,
    })
    print(f"wrote {out}: {checks}", flush=True)


if __name__ == "__main__":
    main()
