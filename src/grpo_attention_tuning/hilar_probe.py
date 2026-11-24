"""HiLaR-style GRPO signal probe on a single-latent model (no parameter update).

HiLaR (arXiv 2607.27760) samples G rollouts of Gaussian-perturbed latents per user and scores the decoded
title.  GRPO only learns if rollouts of one user get different rewards.  An earlier reimplementation
(worktree HiLaR, Musical) found 1.0-1.2 distinct titles per 6 rollouts and reward std ~0 at the paper's sigma.
Here: MW-R-H has ONE thinking token r, so the policy is N(r, sigma^2 I).  sigma is set relative to the typical
distance D between two users' r (median ||r_i - r_j||):  per-coordinate sigma = sigma_rel * D / sqrt(d), so a
noise vector has norm ~ sigma_rel * D.  Rollouts are antithetic pairs (+eps, -eps).

Per row and sigma_rel: greedy constrained decode of one title per rollout; rewards
  hit (exact title), prefix (common-prefix chars / max title length), f1 (word-token F1),
  dll = clip(mean target-token logp under noisy r - under noise-free r, -1, 1),
  combined = hit + 0.5 prefix + 0.5 f1 + dll   (SASRec term not in the probe).
Reports distinct titles per group and the within-group reward std (what GRPO's advantage is made of).
"""
import argparse, json, os, re, sys
from collections import Counter
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from transformers import AutoTokenizer, GenerationConfig, LogitsProcessorList
from model import LatentModel
from latent_grpo_dataset import D3Dataset, get_hash, get_prefix_data
from latent_grpo_processor import CFEnhancedLogitsProcessor

CATEGORY_TEXT = {"Toys_and_Games": "toys and games", "CDs_and_Vinyl": "musics"}
_norm = lambda s: str(s).strip().strip('\n').strip('"').strip()
_brand = lambda s: (re.findall(r"[a-z0-9']+", _norm(s).lower()) + [""])[0]


def _strip(s):
    return s.strip().strip("\n").strip('"').strip()


def copy_flags(df):
    """Same-day, same-brand, title-Jaccard >= 0.3 copy rows (rule of analysis/results/copy_rows_mw_vs_r3)."""
    import ast
    tokset = lambda s: set(re.findall(r"[a-z0-9]+", _norm(s).lower()))
    jac = lambda a, b: len(tokset(a) & tokset(b)) / max(1, len(tokset(a) | tokset(b)))
    out = []
    for hs, hts, t, ts in zip(df.history_item_title, df.history_timestamp, df.item_title, df.timestamp):
        hs, hts = ast.literal_eval(hs), ast.literal_eval(hts)
        out.append(any(abs(ts - ht) < 86400 and _brand(t) == _brand(h) and jac(t, h) >= 0.3 for h, ht in zip(hs, hts)))
    return np.array(out)


def sample_rows(train_file, n_copy, n_other, seed, out_csv):
    import pandas as pd
    df = pd.read_csv(train_file)
    c = copy_flags(df)
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(c)[0], n_copy, replace=False),
                          rng.choice(np.where(~c)[0], n_other, replace=False)])
    df.iloc[idx].reset_index(drop=True).to_csv(out_csv, index=False)
    return np.array([True] * n_copy + [False] * n_other)


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def f1(a, b):
    ca, cb = Counter(words(a)), Counter(words(b))
    ov = sum((ca & cb).values())
    if ov == 0:
        return 0.0
    p, r = ov / sum(ca.values()), ov / sum(cb.values())
    return 2 * p * r / (p + r)


def prefix_score(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n / max(1, max(len(a), len(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path")
    ap.add_argument("--category", default="Toys_and_Games")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--info_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_copy", type=int, default=128)
    ap.add_argument("--n_other", type=int, default=128)
    ap.add_argument("--G", type=int, default=12)
    ap.add_argument("--sigmas", default="0.1,0.25,0.5,1.0")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    sub_csv = os.path.join(args.out, "rows.csv")
    is_copy = sample_rows(args.train_file, args.n_copy, args.n_other, args.seed, sub_csv)
    first_tok = AutoTokenizer.from_pretrained(args.models[0].split("=", 1)[1], trust_remote_code=True)
    rows = D3Dataset(train_file=sub_csv, tokenizer=first_tok, max_len=2048, sample=-1, seed=args.seed,
                     category=CATEGORY_TEXT[args.category], test=False).get_inputs_list()
    sigmas = [float(x) for x in args.sigmas.split(",")]
    G = args.G
    assert G % 2 == 0
    summary, lines = {}, []
    for spec in args.models:
        name, path = spec.split("=", 1)
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        tok.pad_token, tok.pad_token_id, tok.padding_side = tok.eos_token, tok.eos_token_id, "left"
        model = LatentModel.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
        model.attention.float()
        model.eval()
        hash_dict = get_prefix_data(args.info_file, tok)
        prefix_fn = lambda b, ids: hash_dict.get(get_hash(ids), [])
        H = model.config.hidden_size
        state = {"noise": None, "rec": None}

        def hook(module, inp, out):
            if state["rec"] is not None:
                state["rec"].append(out.detach().float().cpu())
            if state["noise"] is not None:
                return out + state["noise"].to(out.dtype)
        model.attention.register_forward_hook(hook)

        data = []
        for row in rows:
            ids, lab = torch.tensor(row["input_ids"]), torch.tensor(row["labels"])
            data.append((ids[lab == -100].to(device), ids[lab != -100].to(device),
                         _strip(tok.decode(ids[lab != -100], skip_special_tokens=True))))

        @torch.no_grad()
        def decode(prompt, B):
            ids = prompt.unsqueeze(0).repeat(B, 1)
            proc = CFEnhancedLogitsProcessor(guidance_scale=1.0, cf_logits=None, prefix_allowed_tokens_fn=prefix_fn,
                                             cf_dict=None, unconditional_ids=None, model=model, tokenizer=tok, num_beams=1)
            cfg = GenerationConfig(num_beams=1, do_sample=False, max_new_tokens=args.max_new_tokens,
                                   pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=True)
            out = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=cfg,
                                 logits_processor=LogitsProcessorList([proc]))
            return [_strip(t) for t in tok.batch_decode(out[:, prompt.numel():], skip_special_tokens=True)]

        @torch.no_grad()
        def target_logp(prompt, tgt, B):
            full = torch.cat([prompt, tgt]).unsqueeze(0).repeat(B, 1)
            T = tgt.numel()
            # transformers 4.48 ignores logits_to_keep (it is num_logits_to_keep there), so slice the full logits ourselves
            lg = model(input_ids=full, attention_mask=torch.ones_like(full)).logits[:, -(T + 1):-1].float()
            lp = torch.log_softmax(lg, -1).gather(-1, tgt.view(1, -1, 1).expand(B, -1, 1)).squeeze(-1)
            return lp.mean(1).cpu().numpy()

        # noise-free r for every row -> typical distance between two users' r
        state["rec"] = []
        clean = []
        for prompt, tgt, target in data:
            state["rec"].clear()
            with torch.no_grad():
                ids = prompt.unsqueeze(0)
                model.generate_embs(ids, torch.ones_like(ids))
            clean.append(state["rec"][0][0])
        state["rec"] = None
        R = torch.stack(clean)
        rng = np.random.default_rng(args.seed)
        i, j = rng.integers(0, len(R), 4000), rng.integers(0, len(R), 4000)
        keep = i != j
        D = float((R[i[keep]] - R[j[keep]]).norm(dim=1).median())
        rnorm = float(R.norm(dim=1).mean())
        print(f"[{name}] median inter-user ||r_i - r_j|| = {D:.3f}; mean ||r|| = {rnorm:.3f}", flush=True)

        # noise-free references
        ref_title, ref_ll = [], []
        for prompt, tgt, target in data:
            ref_title.append(decode(prompt, 1)[0])
            ref_ll.append(float(target_logp(prompt, tgt, 1)[0]))
        ref_hit = np.mean([t == d[2] for t, d in zip(ref_title, data)])

        res = {"D": D, "r_norm": rnorm, "clean_hit_rate": float(ref_hit), "sigma": {}}
        lines.append(f"\n== {name}: median inter-user ||r_i-r_j|| {D:.3f}, mean ||r|| {rnorm:.3f}, noise-free hit rate {ref_hit:.3f} (n={len(data)})")
        lines.append("sigma_rel  noise/||r||  distinct/12  P(>=2 distinct)  P(title==clean)  hit(noisy)  mean dLL   std: hit   prefix  f1     dLL    combined | rows with combined std>0.01")
        for srel in sigmas:
            sig = srel * D / np.sqrt(H)
            acc = {k: [] for k in ("distinct", "same_clean", "hit", "dll", "sd_hit", "sd_prefix", "sd_f1", "sd_dll", "sd_comb")}
            for (prompt, tgt, target), rt, rl in zip(data, ref_title, ref_ll):
                eps = torch.randn(G // 2, H, device=device)
                state["noise"] = sig * torch.cat([eps, -eps])
                titles = decode(prompt, G)
                ll = target_logp(prompt, tgt, G)
                state["noise"] = None
                hit = np.array([t == target for t in titles], float)
                pre = np.array([prefix_score(t, target) for t in titles])
                f1s = np.array([f1(t, target) for t in titles])
                dll = np.clip(ll - rl, -1, 1)
                comb = hit + 0.5 * pre + 0.5 * f1s + dll
                acc["distinct"].append(len(set(titles)))
                acc["same_clean"].append(np.mean([t == rt for t in titles]))
                acc["hit"].append(hit.mean()); acc["dll"].append(dll.mean())
                for k, v in (("sd_hit", hit), ("sd_prefix", pre), ("sd_f1", f1s), ("sd_dll", dll), ("sd_comb", comb)):
                    acc[k].append(v.std())
            a = {k: np.array(v) for k, v in acc.items()}
            r_ = {"distinct_mean": float(a["distinct"].mean()), "p_ge2_distinct": float((a["distinct"] >= 2).mean()),
                  "p_same_as_clean": float(a["same_clean"].mean()), "hit_noisy": float(a["hit"].mean()),
                  "dll_mean": float(a["dll"].mean()), "sd_hit": float(a["sd_hit"].mean()),
                  "sd_prefix": float(a["sd_prefix"].mean()), "sd_f1": float(a["sd_f1"].mean()),
                  "sd_dll": float(a["sd_dll"].mean()), "sd_comb": float(a["sd_comb"].mean()),
                  "rows_comb_sd_gt_0.01": float((a["sd_comb"] > 0.01).mean())}
            for grp, m in (("copy", is_copy), ("other", ~is_copy)):
                r_[grp] = {"distinct_mean": float(a["distinct"][m].mean()), "sd_comb": float(a["sd_comb"][m].mean()),
                           "hit_noisy": float(a["hit"][m].mean())}
            res["sigma"][str(srel)] = r_
            lines.append(f"{srel:9.2f}  {srel * D / rnorm:10.2f}  {r_['distinct_mean']:11.2f}  {r_['p_ge2_distinct']:15.2f}  "
                         f"{r_['p_same_as_clean']:15.2f}  {r_['hit_noisy']:10.3f}  {r_['dll_mean']:8.3f}   "
                         f"{r_['sd_hit']:.3f}  {r_['sd_prefix']:.3f}  {r_['sd_f1']:.3f}  {r_['sd_dll']:.3f}  {r_['sd_comb']:.3f} | "
                         f"{r_['rows_comb_sd_gt_0.01']:.2f}")
            print(lines[-1], flush=True)
        summary[name] = res
        del model
        torch.cuda.empty_cache()
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    open(os.path.join(args.out, "output.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
