"""Test A: do rank-GRPO per-row gradients on the pooling attention point the same way?

Hypothesis (2026-11-22): rank GRPO gains on R3_ori because its r is one shared vector, so every
row's update pushes the attention the same way; MW-R-H's r is per user, so per-row updates spread
out and cancel (copy-row gain +.005 n.s. vs +.018 on R3_ori).

For each model and the SAME stratified training rows (copy / non-copy, rule as in
analysis/results/copy_rows_mw_vs_r3/compare.py), exactly the rank-GRPO step of rank_grpo.py:
  constrained beam search, G = 8 beams -> reward +1 hit / -alpha/log2(k+1) miss
  -> advantage (r - mean)/(std + 1e-4); rows without a hit are skipped (skip_nohit)
  -> loss = -mean_k A_k * mean_t log p(beam_k,t | prompt, r)   (TRL surrogate at ratio 1, no KL)
and records the per-row gradient on the pooling attention (fp32), plus dL/dr.

Outputs per model and row group (all / copy / non-copy):
  mean pairwise cosine between per-row attention gradients (count sketch, d = 16384) and between
  per-row dL/dr (exact), with bootstrap CIs over rows; coherence ratio |sum g| / sum |g| (exact)
  at a common row count.
"""
import argparse, ast, json, math, os, re, sys
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from transformers import AutoTokenizer, GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper
from model import LatentModel
from latent_grpo_dataset import D3Dataset, get_hash, get_prefix_data
from latent_grpo_processor import CFEnhancedLogitsProcessor
from rank_grpo import rank_rewards, _strip

CATEGORY_TEXT = {"Toys_and_Games": "toys and games", "CDs_and_Vinyl": "musics"}
norm = lambda s: str(s).strip().strip('\n').strip('"').strip()
tok_set = lambda s: set(re.findall(r"[a-z0-9]+", norm(s).lower()))
brand = lambda s: (re.findall(r"[a-z0-9']+", norm(s).lower()) + [""])[0]
jac = lambda a, b: len(tok_set(a) & tok_set(b)) / max(1, len(tok_set(a) | tok_set(b)))


def copy_flags(df):
    out = []
    for hs, hts, t, ts in zip(df.history_item_title, df.history_timestamp, df.item_title, df.timestamp):
        hs, hts = ast.literal_eval(hs), ast.literal_eval(hts)
        out.append(any(abs(ts - ht) < 86400 and brand(t) == brand(h) and jac(t, h) >= 0.3 for h, ht in zip(hs, hts)))
    return np.array(out)


def sample_rows(train_file, n_copy, n_other, seed, out_csv):
    df = pd.read_csv(train_file)
    c = copy_flags(df)
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(c)[0], n_copy, replace=False),
                          rng.choice(np.where(~c)[0], n_other, replace=False)])
    sub = df.iloc[idx].reset_index(drop=True)
    sub.to_csv(out_csv, index=False)
    print(f"train rows {len(df)}, copy {c.mean():.3f}; sampled {n_copy} copy + {n_other} other", flush=True)
    return np.array([True] * n_copy + [False] * n_other)


@torch.no_grad()
def beam_search(model, tok, prompt, G, prefix_fn, max_new_tokens):
    ids = prompt.unsqueeze(0)
    att = torch.ones_like(ids)
    proc = CFEnhancedLogitsProcessor(guidance_scale=1.0, cf_logits=None, prefix_allowed_tokens_fn=prefix_fn,
                                     cf_dict=None, unconditional_ids=None, model=model, tokenizer=tok, num_beams=G)
    cfg = GenerationConfig(num_beams=G, num_return_sequences=G, length_penalty=1.0, max_new_tokens=max_new_tokens,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=True,
                           do_sample=False)
    return model.generate(ids, attention_mask=att, generation_config=cfg,
                          logits_processor=LogitsProcessorList([TemperatureLogitsWarper(1.0), proc]))


def run_model(name, path, rows, is_copy, args, sketch_h, sketch_s, device):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tok.pad_token, tok.pad_token_id, tok.padding_side = tok.eos_token, tok.eos_token_id, "left"
    model = LatentModel.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    model.attention.float()
    for p in model.model.parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = False
    params = [p for p in model.attention.parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in params)
    assert n_par == sketch_h.numel(), (n_par, sketch_h.numel())
    model.eval()
    hash_dict = get_prefix_data(args.info_file, tok)
    prefix_fn = lambda b, ids: hash_dict.get(get_hash(ids), [])
    thought_id = model.model.embed_tokens.num_embeddings - 1

    sums = {g: torch.zeros(n_par, device=device) for g in ("all", "copy", "other")}
    rec = []
    for i, row in enumerate(rows):
        ids, lab = torch.tensor(row["input_ids"]), torch.tensor(row["labels"])
        prompt = ids[lab == -100].to(device)
        target = _strip(tok.decode(ids[lab != -100], skip_special_tokens=True))
        out = beam_search(model, tok, prompt, args.G, prefix_fn, args.max_new_tokens)
        P = prompt.numel()
        comp = out[:, P:]
        texts = tok.batch_decode(comp, skip_special_tokens=True)
        hits = [_strip(t) == target for t in texts]
        r = {"row": i, "copy": bool(is_copy[i]), "hit": any(hits),
             "hit_rank": hits.index(True) + 1 if any(hits) else 0}
        if not any(hits):
            rec.append(r)
            continue
        rew = torch.tensor(rank_rewards(hits, args.alpha), device=device)
        adv = (rew - rew.mean()) / (rew.std() + 1e-4)
        is_eos = comp == tok.eos_token_id
        first = torch.where(is_eos.any(1), is_eos.int().argmax(1),
                            torch.full((comp.size(0),), comp.size(1) - 1, device=device))
        cmask = (torch.arange(comp.size(1), device=device)[None] <= first[:, None]).float()
        full = out
        att = torch.ones_like(full)
        # r with grad (generate_embs inline so dL/dr can be read)
        with torch.no_grad():
            hs = model.model(input_ids=full, attention_mask=att, output_hidden_states=True).hidden_states[-1]
        where = torch.nonzero(full == thought_id)
        lat = model.attention(hidden_states=hs.float(), attention_mask=att, thought_id_idx=where[:, 1])
        lat.retain_grad()
        emb = model.model.embed_tokens(full).detach().clone()
        emb[where[:, 0], where[:, 1]] = lat.to(emb.dtype)
        logits = model(inputs_embeds=emb, attention_mask=att).logits[:, P - 1:-1].float()
        logp = torch.log_softmax(logits, -1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
        seq = (logp * cmask).sum(1) / cmask.sum(1)
        loss = -(adv * seq).mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        g = torch.cat([p.grad.reshape(-1).float() for p in params])
        gr = lat.grad.float().sum(0)  # same r for all G beams of the row
        gn = g.norm().item()
        sk = torch.zeros(args.sketch_dim, device=device).index_add_(0, sketch_h, g * sketch_s)
        sums["all"] += g
        sums["copy" if r["copy"] else "other"] += g
        r.update({"gnorm": gn, "grnorm": gr.norm().item(), "sketch": (sk / max(gn, 1e-30)).cpu().numpy(),
                  "gr_unit": (gr / max(gr.norm().item(), 1e-30)).cpu().numpy()})
        rec.append(r)
        if i % 50 == 0:
            print(f"[{name}] row {i}/{len(rows)} hits so far {sum(x['hit'] for x in rec)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return rec, {k: v.norm().item() for k, v in sums.items()}


def pairwise_cos(U):
    n = len(U)
    s = U.sum(0)
    return (s @ s - (U * U).sum()) / (n * (n - 1)) if n > 1 else float("nan")


def boot(U, n_boot=2000, seed=42):
    # resampling with replacement duplicates rows; a row paired with its own copy has cos 1 and
    # biases the mean upward (374737 CIs sat above the point estimate), so drop same-row pairs
    G, n = U @ U.T, len(U)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        c = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        vals.append((c @ G @ c - (c * c).sum()) / (n * n - (c * c).sum()))
    return np.percentile(vals, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path")
    ap.add_argument("--category", default="Toys_and_Games")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--info_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_copy", type=int, default=300)
    ap.add_argument("--n_other", type=int, default=300)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--sketch_dim", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    sub_csv = os.path.join(args.out, "rows.csv")
    is_copy = sample_rows(args.train_file, args.n_copy, args.n_other, args.seed, sub_csv)
    first_tok = AutoTokenizer.from_pretrained(args.models[0].split("=", 1)[1], trust_remote_code=True)
    rows = D3Dataset(train_file=sub_csv, tokenizer=first_tok, max_len=2048, sample=-1, seed=args.seed,
                     category=CATEGORY_TEXT[args.category], test=False).get_inputs_list()

    n_par = 3 * 1536 * 1536 + 3 * 1536  # q, k, v weights + biases = 7,082,496 (asserted against the model)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    sketch_h = torch.randint(0, args.sketch_dim, (n_par,), generator=gen).to(device)
    sketch_s = (torch.randint(0, 2, (n_par,), generator=gen).float() * 2 - 1).to(device)

    summary, lines = {}, []
    for spec in args.models:
        name, path = spec.split("=", 1)
        rec, sumnorm = run_model(name, path, rows, is_copy, args, sketch_h, sketch_s, device)
        hit = [r for r in rec if r["hit"]]
        res = {"rows": len(rec), "hit_rows": len(hit),
               "hit_rate_copy": float(np.mean([r["hit"] for r in rec if r["copy"]])),
               "hit_rate_other": float(np.mean([r["hit"] for r in rec if not r["copy"]]))}
        for grp, sel in (("all", lambda r: True), ("copy", lambda r: r["copy"]), ("other", lambda r: not r["copy"])):
            H = [r for r in hit if sel(r)]
            if len(H) < 3:
                res[grp] = {"n": len(H)}
                continue
            S = np.stack([r["sketch"] for r in H]).astype(np.float64)
            S /= np.linalg.norm(S, axis=1, keepdims=True)
            Rr = np.stack([r["gr_unit"] for r in H]).astype(np.float64)
            gsum = sum(r["gnorm"] for r in H)
            res[grp] = {"n": len(H),
                        "cos_attn_grad": float(pairwise_cos(S)), "cos_attn_grad_ci": boot(S).tolist(),
                        "cos_dL_dr": float(pairwise_cos(Rr)), "cos_dL_dr_ci": boot(Rr).tolist(),
                        "coherence_ratio": sumnorm[grp] / gsum,
                        "mean_grad_norm": gsum / len(H)}
        summary[name] = res
        lines.append(f"\n== {name}: rows {res['rows']}, hit rows {res['hit_rows']} "
                     f"(hit rate copy {res['hit_rate_copy']:.3f}, other {res['hit_rate_other']:.3f})")
        for grp in ("all", "copy", "other"):
            x = res[grp]
            if "cos_attn_grad" not in x:
                lines.append(f"  {grp:5s} n={x['n']} (too few)")
                continue
            lines.append(f"  {grp:5s} n={x['n']:3d}  cos(attn grad) {x['cos_attn_grad']:+.4f} "
                         f"[{x['cos_attn_grad_ci'][0]:+.4f},{x['cos_attn_grad_ci'][1]:+.4f}]  "
                         f"cos(dL/dr) {x['cos_dL_dr']:+.4f} [{x['cos_dL_dr_ci'][0]:+.4f},{x['cos_dL_dr_ci'][1]:+.4f}]  "
                         f"|sum g|/sum|g| {x['coherence_ratio']:.4f} (1/sqrt(n) = {1 / math.sqrt(x['n']):.4f})  "
                         f"mean |g| {x['mean_grad_norm']:.3e}")
        print("\n".join(lines[-4:]), flush=True)
        np.save(os.path.join(args.out, f"{name}_records.npy"),
                np.array([{k: v for k, v in r.items() if k != "sketch"} for r in rec], dtype=object), allow_pickle=True)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    open(os.path.join(args.out, "output.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
