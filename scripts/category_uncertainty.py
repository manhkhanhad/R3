"""Uncertainty of the category predictors (Toys, level-3) and how much calibrated
trust in the predicted category could buy.

Inputs (existing files, read-only):
  prep   output_dir/category_latent/prep/Toys_category_prep.pt
         target / top-5 lists / scores per split: train = SASRec 5-fold out-of-fold
         log-probs, valid = SASRec (all-rows model) log-probs, test = LLM beam scores.
  llm    output_dir/category_prediction/Toys_and_Games_category_level3/eval_category.json
         (10 beams + scores per test row)
  items  per-row item results: R3 (R3_ori Toys final_result.json) and the additive
         category-latent model with the LLM top-1 category (eval_pred1.json).

Part 1 (both predictors): accuracy, AUROC of confidence features for "top-1 correct",
calibration (ECE, reliability bins), selective accuracy at coverage.
Part 2 (test, LLM confidence): per-row gate -- the pred1 model's list where the LLM's
confidence >= tau, R3's list otherwise. A diagnostic upper bound on calibrated trust
(tau swept on the test set itself, so the best tau is optimistic), not a method.
"""
import argparse
import ast
import json

import numpy as np
import torch


def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):  # average ranks for ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = label.sum(), (~label).sum()
    return (ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_auroc(score, label, n=500, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(score), size=(n, len(score)))
    vals = [auroc(score[i], label[i]) for i in idx]
    return np.percentile(vals, [2.5, 97.5])


def calibration(prob, label, bins=10):
    prob, label = np.asarray(prob), np.asarray(label, float)
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (prob >= a) & (prob < b) if b < 1 else (prob >= a) & (prob <= b)
        if m.sum() == 0:
            continue
        rows.append((a, b, int(m.sum()), prob[m].mean(), label[m].mean()))
        ece += m.mean() * abs(prob[m].mean() - label[m].mean())
    return ece, rows


def selective(conf, label, coverages=(0.05, 0.1, 0.2, 0.3, 0.5, 1.0)):
    order = np.argsort(-np.asarray(conf), kind="mergesort")
    label = np.asarray(label, float)[order]
    return [(c, label[: max(1, int(round(c * len(label))))].mean()) for c in coverages]


def per_row_hits(path):
    rows = json.load(open(path))
    hr, nd, tgt = [], [], []
    for r in rows:
        t = str(r["output"]).strip(' "\n')
        rank = None
        for i, p in enumerate(r["predict"][:10]):
            if str(p).strip(' "\n') == t:
                rank = i
                break
        hr.append(rank is not None)
        nd.append(np.log(2) / np.log(rank + 2) if rank is not None else 0.0)
        tgt.append(t)
    return np.array(hr, float), np.array(nd), tgt


def predictor_report(name, target, lists, scores, missing, softmax_scores, out):
    target, lists, scores = target.numpy(), lists.numpy(), scores.float().numpy()
    keep = target != missing
    target, lists, scores = target[keep], lists[keep], scores[keep]
    correct = lists[:, 0] == target
    in5 = (lists == target[:, None]).any(1)
    feats = {"top-1 score": scores[:, 0], "margin top1-top2": scores[:, 0] - scores[:, 1]}
    if softmax_scores:  # LLM beam scores: renormalise over the returned beams
        p = np.exp(scores - scores.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        feats["renormalised top-1 prob (over 5)"] = p[:, 0]
        prob = None
    else:
        prob = np.exp(scores[:, 0])
        feats["top-1 prob"] = prob
        p5 = np.exp(scores)
        feats["top-5 mass"] = p5.sum(1)
    out.append(f"\n## {name} (n = {len(target)})\n")
    out.append(f"top-1 accuracy {correct.mean():.4f}, top-5 accuracy {in5.mean():.4f}\n")
    out.append("| confidence feature | AUROC for top-1 correct [95% CI] |\n|---|---|")
    for k, v in feats.items():
        lo, hi = boot_auroc(v, correct)
        out.append(f"| {k} | {auroc(v, correct):.3f} [{lo:.3f}, {hi:.3f}] |")
    best = max(feats, key=lambda k: auroc(feats[k], correct))
    out.append(f"\nSelective top-1 accuracy by coverage (rows ranked by {best}):\n")
    out.append("| coverage | accuracy |\n|---:|---:|")
    for c, a in selective(feats[best], correct):
        out.append(f"| {c:.2f} | {a:.4f} |")
    if prob is not None:
        ece, rows = calibration(prob, correct)
        out.append(f"\nCalibration of the top-1 probability: ECE {ece:.4f}\n")
        out.append("| bin | n | mean prob | accuracy |\n|---|---:|---:|---:|")
        for a, b, n, mp, acc in rows:
            out.append(f"| [{a:.1f}, {b:.1f}) | {n} | {mp:.3f} | {acc:.3f} |")
    return feats, correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="output_dir/category_latent/prep/Toys_category_prep.pt")
    ap.add_argument("--r3", default="../R3_ori/output_dir/Toys_and_Games/latent/final_result.json")
    ap.add_argument("--pred1", default="output_dir/category_latent/Toys_and_Games_cat_latent_additive/eval_pred1.json")
    ap.add_argument("--output", default="output_dir/category_latent/category_uncertainty_report.md")
    args = ap.parse_args()

    d = torch.load(args.prep, weights_only=False)
    missing = len(d["catalog"])  # index used for rows without a level-3 category
    out = ["# Category-prediction uncertainty (Toys, level 3)",
           "Generated by scripts/category_uncertainty.py. Rows without a category excluded."]
    predictor_report("SASRec, train split (5-fold out-of-fold)", d["target"]["train"], d["lists"]["train"],
                     d["scores"]["train"], missing, False, out)
    predictor_report("SASRec, valid split (all-rows model)", d["target"]["valid"], d["lists"]["valid"],
                     d["scores"]["valid"], missing, False, out)
    feats, correct = predictor_report("LLM SFT, test split (beam scores)", d["target"]["test"], d["lists"]["test"],
                                      d["scores"]["test"], missing, True, out)

    # Part 2: gate between the pred1 category-latent model and R3 on test.
    t = d["target"]["test"].numpy()
    keep = t != missing
    r3_hr, r3_nd, r3_t = per_row_hits(args.r3)
    p1_hr, p1_nd, p1_t = per_row_hits(args.pred1)
    assert r3_t == p1_t and len(r3_t) == len(t), "rows not aligned"
    conf_name = max(feats, key=lambda k: auroc(feats[k], correct))
    conf = np.full(len(t), -np.inf)
    conf[keep] = feats[conf_name]  # rows without a category always fall back to R3
    corr_all = np.zeros(len(t), bool)
    corr_all[keep] = correct
    out.append(f"\n## Gate: pred1 category-latent model where LLM confidence >= tau, R3 otherwise (test, n = {len(t)})\n")
    out.append(f"Confidence = {conf_name}. R3 HR@10 {r3_hr.mean():.4f} / NDCG@10 {r3_nd.mean():.4f}; "
               f"pred1 {p1_hr.mean():.4f} / {p1_nd.mean():.4f}. tau swept on test itself: optimistic.\n")
    out.append("| share gated to pred1 | category acc. in gated rows | HR@10 gated | NDCG@10 gated | ΔHR@10 vs R3 [95% CI] | ΔNDCG@10 vs R3 [95% CI] |")
    out.append("|---:|---:|---:|---:|---|---|")
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(t), size=(2000, len(t)))
    finite = np.sort(conf[np.isfinite(conf)])[::-1]
    for share in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0):
        n = max(1, int(round(share * keep.sum())))
        tau = finite[n - 1]
        g = conf >= tau
        hr = np.where(g, p1_hr, r3_hr)
        nd = np.where(g, p1_nd, r3_nd)
        cells = []
        for a, b in ((hr, r3_hr), (nd, r3_nd)):
            diff = a - b
            lo, hi = np.percentile(diff[idx].mean(1), [2.5, 97.5])
            cells.append(f"{diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
        out.append(f"| {g.mean():.3f} | {corr_all[g].mean():.3f} | {hr.mean():.4f} | {nd.mean():.4f} | {cells[0]} | {cells[1]} |")
    # perfect-knowledge ceiling: gate exactly the rows whose top-1 category is right
    hr = np.where(corr_all, p1_hr, r3_hr)
    nd = np.where(corr_all, p1_nd, r3_nd)
    out.append(f"\nOracle gate (pred1 exactly where the LLM top-1 category is right, share {corr_all.mean():.3f}): "
               f"HR@10 {hr.mean():.4f} ({hr.mean() - r3_hr.mean():+.4f}), NDCG@10 {nd.mean():.4f} ({nd.mean() - r3_nd.mean():+.4f}).")
    text = "\n".join(out) + "\n"
    open(args.output, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
