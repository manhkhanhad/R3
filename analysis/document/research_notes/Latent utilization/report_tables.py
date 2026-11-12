"""Tables for the MW → MWC report (Toys test, n = 6,738): ranking at all cutoffs, copy split,
paired bootstrap vs plain R3 (A), latent interventions with paired-bootstrap CIs."""
import json, glob, ast, re, os, numpy as np, pandas as pd
R = "/common/home/users/m/mk.ngohuu.2024/research/R3/"
O = R + "output_dir/"
C = O + "margin_wider_collab/"
test = pd.read_csv(glob.glob(R + "data/Toys_and_Games/test/Toys_and_Games_5_*.csv")[0])
n = len(test)
norm = lambda s: str(s).strip(' "\n')
brand = lambda t: norm(t).split(" ")[0].lower()
tok = lambda t: set(re.findall(r"\w+", norm(t).lower()))
ht = [ast.literal_eval(x) for x in test.history_item_title]
hts = [ast.literal_eval(x) for x in test.history_timestamp]
copy = np.array([test.timestamp[i] == max(hts[i]) and any(brand(test.item_title[i]) == brand(t) and
        len(tok(test.item_title[i]) & tok(t)) / max(1, len(tok(test.item_title[i]) | tok(t))) >= 0.3 for t in ht[i]) for i in range(n)])

def ranks(path):
    d = json.load(open(path))
    if isinstance(d, dict) and "records" in d:
        recs = sorted(d["records"], key=lambda r: r["sample_id"])
        return np.array([r["rank"] if r["rank"] is not None else 11 for r in recs], float)
    return np.array([([norm(x) for x in r["predict"][:10]] + [norm(r["output"])]).index(norm(r["output"])) + 1 for r in d], float)

HR = lambda r, k: (r <= k).astype(float)
ND = lambda r, k: np.where(r <= k, 1 / np.log2(r + 1), 0.0)
rng = np.random.default_rng(42)
IDX = rng.integers(0, n, (2000, n))
def boot(d):
    s = d[IDX].mean(1)
    lo, hi = np.quantile(s, [.025, .975])
    return d.mean(), lo, hi, (lo > 0 or hi < 0)
fmt = lambda b: f"{b[0]:+.4f} [{b[1]:+.4f}, {b[2]:+.4f}]{' *' if b[3] else ''}"

G = "generation_test_all_k32_s42_{}_t1_temp0.1_a1.0.json"
runs = [  # name, ranking file, intervention dir/type
    ("plain R3 (A, retrain)", O + "guidance_scaffold/Toys_and_Games_guidance_A/eval_raw.json", None),
    ("WIDER only", O + "wider/Toys_and_Games_wider_0.1_fixed/eval_raw.json", ("gen", O + "wider/Toys_and_Games_wider_0.1_fixed/")),
    ("MW (margin + WIDER, 356322)", O + "margin_wider/Toys_and_Games_margin_0.2_wider_0.1/eval_raw.json", ("gen", O + "margin_wider/Toys_and_Games_margin_0.2_wider_0.1/")),
    ("MWC-G (donor grounding)", C + "Toys_and_Games_margin_0.2_wider_0.1_grounded/eval_knn.json", ("slot", C + "Toys_and_Games_margin_0.2_wider_0.1_grounded/")),
    ("MWC-I (user+target InfoNCE)", C + "Toys_and_Games_margin_0.2_wider_0.1_infonce/eval_raw.json", ("gen", C + "Toys_and_Games_margin_0.2_wider_0.1_infonce/")),
    ("MWC-I-B (target InfoNCE + hard neg.)", C + "Toys_and_Games_margin_0.2_wider_0.1_infonceB/eval_raw.json", ("gen", C + "Toys_and_Games_margin_0.2_wider_0.1_infonceB/")),
    ("MWC-I-C (B + item head)", C + "Toys_and_Games_margin_0.2_wider_0.1_infonceC/eval_raw.json", ("gen", C + "Toys_and_Games_margin_0.2_wider_0.1_infonceC/")),
    ("G3 (CF-donor evidence)", C + "Toys_and_Games_margin_0.2_wider_0.1_cfg3/eval_knn.json", ("slot", C + "Toys_and_Games_margin_0.2_wider_0.1_cfg3/")),
    ("MWC-M (same-motive InfoNCE)", C + "Toys_and_Games_margin_0.2_wider_0.1_motive/eval_raw.json", ("gen", C + "Toys_and_Games_margin_0.2_wider_0.1_motive/")),
    ("MWC-M2 (SASRec-neighbour motive)", C + "Toys_and_Games_margin_0.2_wider_0.1_motive_sasrec/eval_raw.json", ("gen", C + "Toys_and_Games_margin_0.2_wider_0.1_motive_sasrec/")),
    ("MWC-M m=0.1", C + "Toys_and_Games_margin_0.1_wider_0.1_motive/eval_raw.json", ("gen", C + "Toys_and_Games_margin_0.1_wider_0.1_motive/")),
    ("MWC-M m=0.05", C + "Toys_and_Games_margin_0.05_wider_0.1_motive/eval_raw.json", ("gen", C + "Toys_and_Games_margin_0.05_wider_0.1_motive/")),
    ("MWC-M3a (self-weighted hinge)", C + "Toys_and_Games_margin_0.2_wider_0.1_motive_selfmargin/eval_raw.json", ("gen", C + "Toys_and_Games_margin_0.2_wider_0.1_motive_selfmargin/")),
    ("MWC-M3a-n (normalised self-weight)", C + "Toys_and_Games_margin_0.2_wider_0.1_motive_selfnorm/eval_raw.json", ("gen", C + "Toys_and_Games_margin_0.2_wider_0.1_motive_selfnorm/")),
    ("MWC-M3b (hinge epoch 1 only)", C + "Toys_and_Games_margin_0.2_wider_0.1_motive_marginwarm/eval_raw.json", ("gen", C + "Toys_and_Games_margin_0.2_wider_0.1_motive_marginwarm/")),
    ("MW-3b (MW, hinge epoch 1 only, no motive)", C + "Toys_and_Games_margin_0.2_wider_0.1_marginwarm/eval_raw.json", ("gen", C + "Toys_and_Games_margin_0.2_wider_0.1_marginwarm/")),
]
A = ranks(runs[0][1])
out = []
out.append("### Ranking (all cutoffs; @20 = @10 with 10 beams)\n")
out.append("| model | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 | copy HR@10 (905) | non-copy HR@10 (5,833) |")
out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
boots, interv = [], []
for name, path, iv in runs:
    if not os.path.exists(path):
        out.append(f"| {name} | pending |  |  |  |  |  |  |  |  |"); continue
    r = ranks(path)
    out.append(f"| {name} | {HR(r,1).mean():.4f} | {HR(r,3).mean():.4f} | {HR(r,5).mean():.4f} | {HR(r,10).mean():.4f} | "
               f"{ND(r,3).mean():.4f} | {ND(r,5).mean():.4f} | {ND(r,10).mean():.4f} | {HR(r,10)[copy].mean():.3f} | {HR(r,10)[~copy].mean():.4f} |")
    if name.startswith("plain"):
        continue
    boots.append(f"| {name} | {fmt(boot(HR(r,5)-HR(A,5)))} | {fmt(boot(ND(r,5)-ND(A,5)))} | {fmt(boot(HR(r,10)-HR(A,10)))} | {fmt(boot(ND(r,10)-ND(A,10)))} |")
    if iv is None:
        continue
    kind, d = iv
    if kind == "gen":
        files = {m: d + G.format(m) for m in ("raw", "mask", "mean", "shuffle_user")}
    else:
        files = {"raw": d + "eval_knn.json", "mask": d + "eval_slot_mask.json", "mean": d + "eval_slot_mean.json", "shuffle_user": d + "eval_slot_shuffle_user.json"}
    if not all(os.path.exists(f) for f in files.values()):
        interv.append(f"| {name} | pending | | | |"); continue
    rr = ranks(files["raw"])
    cells = []
    for m in ("mask", "mean", "shuffle_user"):
        rm = ranks(files[m])
        cells.append(f"{fmt(boot(HR(rr,10)-HR(rm,10)))} / {boot(ND(rr,10)-ND(rm,10))[0]:+.4f}")
    interv.append(f"| {name} | {HR(rr,10).mean():.4f} | " + " | ".join(cells) + " |")
out.append("\n### Paired bootstrap vs plain R3 (2,000 resamples, seed 42; * = 95% CI excludes 0)\n")
out.append("| model | ΔHR@5 | ΔNDCG@5 | ΔHR@10 | ΔNDCG@10 |")
out.append("|---|---|---|---|---|")
out += boots
out.append("\n### Latent interventions: raw − intervention, HR@10 [95% CI] / NDCG@10 (positive = replacing the latent hurts = latent used)\n")
out.append("| model | raw HR@10 | zero latent (mask) | mean latent | another user's latent (shuffle) |")
out.append("|---|---:|---|---|---|")
out += interv
open("/common/home/users/m/mk.ngohuu.2024/.claude/jobs/e8c23a07/tmp/report_tables.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
