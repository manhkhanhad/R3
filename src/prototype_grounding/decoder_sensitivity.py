#!/usr/bin/env python3
"""Measure how strongly the second-pass decoder responds to the thought vector r.

The E1 ranking interventions only show whether top-10 lists change. This script
measures the decoder's sensitivity to r directly, under teacher forcing on the
held-out target title, with three complementary probes:

1. Output shift: per-token KL(p_raw || p_variant) over the target-title
   positions, target NLL delta and argmax agreement, for r-interventions
   (mask / mean / shuffle) and, as a scale reference, for a small history-side
   intervention (drop the most recent history item, with r either recomputed
   or held fixed at the original value).
2. Attention: per layer, the attention mass that target-predicting queries
   place on the thought position versus an average prompt token.
3. Gradient x input: |<dNLL/de, e>| and ||dNLL/de|| at the thought position
   versus the average prompt token, plus |<dNLL/dr, r - mean_r>| as the
   linearized sensitivity to r's sample-specific content.

Everything is computed on the raw checkpoint; nothing is trained or modified.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import experiment as E

R_VARIANTS = ("mask", "mean", "shuffle")
X_VARIANTS = ("drop_last_recomputed_r", "drop_last_fixed_r")


class DropLastHistory(E.PromptDataset):
    """Same prompts with the most recent history item removed (kept if only one)."""

    def _encode(self, sample_id, row):
        row = row.copy()
        history = ast.literal_eval(row.history_item_title)
        if len(history) > 1:
            row.history_item_title = repr(history[:-1])
        return super()._encode(sample_id, row)


def target_positions(labels: torch.Tensor) -> list[torch.Tensor]:
    """Positions t whose logits predict a supervised target token labels[t+1]."""
    valid = labels[:, 1:].ne(-100)
    return [torch.nonzero(v).squeeze(-1) for v in valid]


def target_log_probs(model, embeds, mask, labels) -> list[torch.Tensor]:
    """Per example: [n_target_tokens, vocab] log-probabilities at target-predicting positions."""
    with torch.no_grad():
        hidden = model.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False).last_hidden_state
    out = []
    for b, pos in enumerate(target_positions(labels)):
        out.append(F.log_softmax(model.lm_head(hidden[b, pos]).float(), dim=-1))
    return out


def compare(raw_lp, var_lp, labels_raw, labels_var) -> dict[str, np.ndarray]:
    kl, kl_first, dnll, agree = [], [], [], []
    for b in range(len(raw_lp)):
        p, q = raw_lp[b], var_lp[b]
        if p.shape != q.shape:  # truncation changed the target length; skip example
            kl.append(np.nan); kl_first.append(np.nan); dnll.append(np.nan); agree.append(np.nan)
            continue
        tok = labels_raw[b, 1:][labels_raw[b, 1:].ne(-100)]
        tok_v = labels_var[b, 1:][labels_var[b, 1:].ne(-100)]
        assert torch.equal(tok, tok_v)
        per_tok = (p.exp() * (p - q)).sum(-1)
        kl.append(per_tok.mean().item())
        kl_first.append(per_tok[0].item())
        nll_p = -p.gather(1, tok[:, None]).mean()
        nll_q = -q.gather(1, tok[:, None]).mean()
        dnll.append((nll_q - nll_p).item())
        agree.append(p.argmax(-1).eq(q.argmax(-1)).float().mean().item())
    return {"kl": np.array(kl), "kl_first": np.array(kl_first),
            "dnll": np.array(dnll), "argmax_agree": np.array(agree)}


def attention_stats(model, embeds, mask, labels, pos) -> dict[str, np.ndarray]:
    """Per layer, averaged over heads, queries and batch examples."""
    with torch.no_grad():
        out = model.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False, output_attentions=True)
    n_layers = len(out.attentions)
    thought_q_later = np.zeros((len(pos), n_layers))   # mass on r from target-token queries (q > r)
    thought_q_self = np.zeros((len(pos), n_layers))    # mass on r from the query at r's own position
    prompt_tok_avg = np.zeros((len(pos), n_layers))    # average mass per prompt token (excl. first token)
    first_tok = np.zeros((len(pos), n_layers))         # attention-sink token
    for b, qpos in enumerate(target_positions(labels)):
        t = int(pos[b])
        later = qpos[qpos > t]
        for layer, att in enumerate(out.attentions):
            a = att[b].float().mean(0)  # [L, L], mean over heads
            thought_q_self[b, layer] = a[t, t].item()
            if len(later):
                rows = a[later]
                thought_q_later[b, layer] = rows[:, t].mean().item()
                prompt_tok_avg[b, layer] = rows[:, 1:t].sum(-1).mean().item() / max(t - 1, 1)
                first_tok[b, layer] = rows[:, 0].mean().item()
    return {"thought_mass_target_queries": thought_q_later, "thought_self_mass": thought_q_self,
            "avg_prompt_token_mass": prompt_tok_avg, "first_token_mass": first_tok}


def gradient_stats(model, embeds, mask, labels, pos, mean_r) -> dict[str, np.ndarray]:
    embeds = embeds.detach().clone().requires_grad_(True)
    logits = model(inputs_embeds=embeds, attention_mask=mask, use_cache=False).logits[:, :-1].float()
    shifted = labels[:, 1:]
    loss = F.cross_entropy(logits.transpose(1, 2), shifted, ignore_index=-100, reduction="none")
    valid = shifted.ne(-100)
    per_example = (loss * valid).sum(1) / valid.sum(1).clamp_min(1)
    (grad,) = torch.autograd.grad(per_example.sum(), embeds)
    grad, e = grad.float(), embeds.detach().float()
    gxi = (grad * e).sum(-1).abs()   # [B, L]
    gnorm = grad.norm(dim=-1)
    res = {k: np.zeros(len(pos)) for k in ("thought_gxi", "prompt_gxi_avg", "thought_grad_norm", "prompt_grad_norm_avg",
                                           "thought_embed_norm", "prompt_embed_norm_avg",
                                           "thought_content_gxd")}
    enorm = e.norm(dim=-1)
    for b in range(len(pos)):
        t = int(pos[b])
        res["thought_gxi"][b] = gxi[b, t].item()
        # linearized effect of moving r to the population mean: content, not presence
        res["thought_content_gxd"][b] = (grad[b, t] * (e[b, t] - mean_r)).sum().abs().item()
        res["prompt_gxi_avg"][b] = gxi[b, 1:t].mean().item()
        res["thought_grad_norm"][b] = gnorm[b, t].item()
        res["prompt_grad_norm_avg"][b] = gnorm[b, 1:t].mean().item()
        res["thought_embed_norm"][b] = enorm[b, t].item()
        res["prompt_embed_norm_avg"][b] = enorm[b, 1:t].mean().item()
    return res


def ci(x: np.ndarray, seed: int = 42, reps: int = 2000) -> dict[str, float]:
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(reps)])
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "ci95_low": float(np.quantile(means, .025)), "ci95_high": float(np.quantile(means, .975)), "n": int(len(x))}


def run(checkpoint: Path, category: str, split: str, limit: int, batch_size: int, tag: str) -> None:
    E.configure_category(category)
    model, tokenizer, device = E.load_model(checkpoint)
    model.config._attn_implementation = "eager"  # needed for output_attentions
    ds = E.PromptDataset(split, tokenizer, limit)
    ds_drop = DropLastHistory(split, tokenizer, limit)
    pad = tokenizer.pad_token_id
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda x: E.collate(x, pad))
    loader_drop = DataLoader(ds_drop, batch_size=batch_size, shuffle=False, collate_fn=lambda x: E.collate(x, pad))

    # Pass 1: raw thoughts for the whole subset (mean and shuffle donors).
    thoughts = []
    for batch in loader:
        ids, mask, pos = [batch[k].to(device) for k in ("input_ids", "attention_mask", "thought_pos")]
        thoughts.append(E.thought_embeds(model, ids, mask, pos)[0].cpu())
    thoughts = torch.cat(thoughts)
    mean_r = thoughts.mean(0)
    users = [r["user_id"] for r in ds.rows]
    n = len(ds)
    donor = []
    for i in range(n):
        j = (i + n // 2) % n
        while users[j] == users[i]:
            j = (j + 1) % n
        donor.append(j)
    donor = torch.tensor(donor)

    results = {v: {} for v in R_VARIANTS + X_VARIANTS}
    attn, grads = {}, {}
    offset = 0
    for bi, (batch, batch_d) in enumerate(zip(loader, loader_drop)):
        ids, mask, labels, pos = [batch[k].to(device) for k in ("input_ids", "attention_mask", "labels", "thought_pos")]
        ids_d, mask_d, labels_d, pos_d = [batch_d[k].to(device) for k in ("input_ids", "attention_mask", "labels", "thought_pos")]
        B = len(ids)
        r = thoughts[offset:offset + B].to(device)
        _, embeds = E.thought_embeds(model, ids, mask, pos)
        raw_lp = target_log_probs(model, embeds, mask, labels)

        swaps = {"mask": torch.zeros_like(r), "mean": mean_r.to(device).expand_as(r),
                 "shuffle": thoughts[donor[offset:offset + B]].to(device)}
        for name, vec in swaps.items():
            lp = target_log_probs(model, E.substitute(embeds, pos, vec), mask, labels)
            for k, v in compare(raw_lp, lp, labels, labels).items():
                results[name].setdefault(k, []).append(v)

        _, embeds_d = E.thought_embeds(model, ids_d, mask_d, pos_d)
        for name, emb in (("drop_last_recomputed_r", embeds_d),
                          ("drop_last_fixed_r", E.substitute(embeds_d, pos_d, r))):
            lp = target_log_probs(model, emb, mask_d, labels_d)
            for k, v in compare(raw_lp, lp, labels, labels_d).items():
                results[name].setdefault(k, []).append(v)

        for k, v in attention_stats(model, embeds, mask, labels, pos).items():
            attn.setdefault(k, []).append(v)
        for k, v in gradient_stats(model, embeds, mask, labels, pos, mean_r.to(device).float()).items():
            grads.setdefault(k, []).append(v)
        offset += B
        if bi % 50 == 0:
            print(f"[{tag}] {offset}/{n}", flush=True)

    out_dir = E.ANALYSIS / "results/decoder_sensitivity" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    summary = {"checkpoint": str(checkpoint), "category": category, "split": split, "n": n,
               "output_shift": {}, "attention_by_layer": {}, "gradient": {}}
    for name, d in results.items():
        summary["output_shift"][name] = {}
        for k, chunks in d.items():
            x = np.concatenate(chunks)
            arrays[f"{name}__{k}"] = x
            summary["output_shift"][name][k] = ci(x)
    for k, chunks in attn.items():
        x = np.concatenate(chunks)  # [n, layers]
        arrays[f"attn__{k}"] = x
        summary["attention_by_layer"][k] = x.mean(0).tolist()
    ratio = arrays["attn__thought_mass_target_queries"] / np.maximum(arrays["attn__avg_prompt_token_mass"], 1e-12)
    summary["attention_by_layer"]["thought_over_avg_prompt_token"] = ratio.mean(0).tolist()
    summary["attention_overall"] = {
        "thought_mass_target_queries": ci(arrays["attn__thought_mass_target_queries"].mean(1)),
        "avg_prompt_token_mass": ci(arrays["attn__avg_prompt_token_mass"].mean(1)),
        "thought_over_avg_prompt_token": ci(ratio.mean(1)),
    }
    for k, chunks in grads.items():
        arrays[f"grad__{k}"] = np.concatenate(chunks)
        summary["gradient"][k] = ci(arrays[f"grad__{k}"])
    centered = thoughts - mean_r
    summary["latent_spread"] = {
        "norm_mean": float(thoughts.norm(dim=-1).mean()),
        "mean_vector_norm": float(mean_r.norm()),
        "centered_norm_mean": float(centered.norm(dim=-1).mean()),
        "centered_over_norm_mean": float((centered.norm(dim=-1) / thoughts.norm(dim=-1)).mean()),
        "cosine_to_mean_mean": float(F.cosine_similarity(thoughts, mean_r[None], dim=-1).mean()),
    }
    summary["gradient"]["gxi_ratio_thought_over_avg_prompt"] = ci(
        arrays["grad__thought_gxi"] / np.maximum(arrays["grad__prompt_gxi_avg"], 1e-12))
    summary["gradient"]["grad_norm_ratio_thought_over_avg_prompt"] = ci(
        arrays["grad__thought_grad_norm"] / np.maximum(arrays["grad__prompt_grad_norm_avg"], 1e-12))
    E.atomic_json(out_dir / "summary.json", summary)
    np.savez_compressed(out_dir / "per_example.npz", **arrays)
    print(f"wrote {out_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--category", default="Musical_Instruments")
    p.add_argument("--split", choices=("valid", "test"), default="test")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    a = p.parse_args()
    E.seed_all(42)
    run(a.checkpoint, a.category, a.split, a.limit, a.batch_size, a.tag)


if __name__ == "__main__":
    main()
