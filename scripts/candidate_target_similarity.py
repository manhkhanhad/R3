"""How close are SASRec's top-10 candidates to the target item (cosine)?

Per test row, cosine between the target item and each item of a list, in two
spaces: Qwen3 title embeddings (semantic) and the SASRec item embeddings
(collaborative; items outside SASRec's catalog skipped).  Lists: SASRec top-10
(Cornac export), R3's top-10 predictions, the last history item, and 10
random catalog items.  Summaries: cos of the top-1 item, max and mean over the
list; on all rows and on rows where the target is not in the list (near
misses), split into copy rows (target first word in a history title) / other.

Usage: candidate_target_similarity.py --sasrec_csv <test_recommendations.csv>
           --sasrec_model <.pkl> --r3_eval <final_result.json> [--out report.md]
"""

import argparse
import ast
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

SPLIT_FILE = "Toys_and_Games_5_2017-1-2018-11.csv"


def first_word(title):
    words = str(title).lower().split()
    return words[0] if words else ""


def boot(d, n=2000, seed=42):
    d = np.asarray(d, float)
    s = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/Toys_and_Games")
    p.add_argument("--sasrec_csv", required=True)
    p.add_argument("--sasrec_model", required=True)
    p.add_argument("--r3_eval", required=True)
    p.add_argument("--out")
    args = p.parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "latent"))
    from sasrec_pair_features import item_embeddings, load_sasrec

    # asin -> item id over all splits; title -> item id from the info file (first occurrence)
    asin_id = {}
    for split in ("train", "valid", "test"):
        f = pd.read_csv(os.path.join(args.data_dir, split, SPLIT_FILE))
        for a, i in zip(f["item_asin"], f["item_id"]):
            asin_id[a] = int(i)
        for a_s, i_s in zip(f["item_asins"], f["history_item_id"]):
            asin_id.update(zip(ast.literal_eval(a_s), ast.literal_eval(i_s)))
    info = os.path.join(args.data_dir, "info", SPLIT_FILE.replace(".csv", ".txt"))
    title_id = {}
    for line in open(info):
        title, i = line.rstrip("\n").rsplit("\t", 1)
        title_id.setdefault(title.strip(), int(i))
    n_items = len(open(info).readlines())

    qwen = torch.load(os.path.join(args.data_dir, "info", "Toys_and_Games_qwen3emb_1536.pt"),
                      weights_only=False)["embeddings"].float()
    qwen = torch.nn.functional.normalize(qwen, dim=-1).numpy()
    rec = load_sasrec(args.sasrec_model)
    sas_items = item_embeddings(rec)
    id_asin = {i: a for a, i in asin_id.items()}
    sas = np.zeros((n_items, sas_items.shape[1]), np.float32)
    sas_ok = np.zeros(n_items, bool)
    for i in range(n_items):
        j = rec.iid_map.get(id_asin.get(i))
        if j is not None:
            sas[i], sas_ok[i] = sas_items[j], True

    test = pd.read_csv(os.path.join(args.data_dir, "test", SPLIT_FILE))
    srec = pd.read_csv(args.sasrec_csv)
    assert (srec["target_asin"].values == test["item_asin"].values).all()
    r3 = json.load(open(args.r3_eval))
    target = test["item_id"].astype(int).to_numpy()
    copy = np.array([first_word(t) in {first_word(h) for h in ast.literal_eval(hs)}
                     for t, hs in zip(test["item_title"], test["history_item_title"])])
    rng = np.random.default_rng(42)
    lists = {
        "SASRec top-10": [[asin_id[a] for a in json.loads(r)[:10]] for r in srec["recommended_asins"]],
        "R3 top-10": [[title_id[x.strip().strip('"')] for x in row["predict"][:10]] for row in r3],
        "last history item": [[ast.literal_eval(h)[-1]] for h in test["history_item_id"]],
        "random 10 items": [rng.integers(0, n_items, 10).tolist() for _ in range(len(test))],
    }

    lines = [f"# Candidate–target cosine similarity (Toys test, n = {len(test)})", "",
             "top-1 = cos(target, first item); max / mean over the list. "
             "'near-miss rows' = target not in the list. Copy rows: target first word in a history title.", ""]
    stats = {}
    for space, emb, ok in (("Qwen3 title embedding", qwen, np.ones(n_items, bool)), ("SASRec item embedding", sas, sas_ok)):
        lines += [f"## {space}", "",
                  "| list | rows | n | top-1 | max | mean | max ≥ 0.8 | max ≥ 0.9 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for name, ls in lists.items():
            top1, mx, mean, hit = [], [], [], []
            for t, items in zip(target, ls):
                items = [i for i in items if ok[i]]
                if not ok[t] or not items:
                    top1.append(np.nan); mx.append(np.nan); mean.append(np.nan); hit.append(False)
                    continue
                c = emb[items] @ emb[t]
                top1.append(c[0]); mx.append(c.max()); mean.append(c.mean()); hit.append(t in items)
            top1, mx, mean, hit = map(np.array, (top1, mx, mean, hit))
            stats[(space, name)] = (mx, hit)
            for label, m in (("all", np.ones(len(test), bool)), ("near-miss, copy", ~hit & copy),
                             ("near-miss, other", ~hit & ~copy), ("near-miss, all", ~hit)):
                m = m & ~np.isnan(mx)
                lines.append(f"| {name} | {label} | {m.sum()} | {np.mean(top1[m]):.3f} | {np.mean(mx[m]):.3f} | "
                             f"{np.mean(mean[m]):.3f} | {np.mean(mx[m] >= 0.8):.3f} | {np.mean(mx[m] >= 0.9):.3f} |")
        lines.append("")
        a, _ = stats[(space, "SASRec top-10")]
        b, _ = stats[(space, "R3 top-10")]
        for label, m in (("all", np.ones(len(test), bool)), ("copy", copy), ("other", ~copy)):
            m = m & ~np.isnan(a) & ~np.isnan(b)
            d, lo, hi = boot(a[m] - b[m])
            lines.append(f"- {space}, {label} rows: SASRec − R3 max-cos {d:+.4f} [{lo:+.4f}, {hi:+.4f}] (n = {m.sum()})")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
