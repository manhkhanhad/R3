"""Inputs for the LLM next-category pilot (CPU). Validation rows only; targets kept out of the prompt file.

Writes to --out:
  taxonomy.json        the category paths of the catalog (root 'Toys & Games' dropped; 'unknown' excluded)
  inputs.jsonl         per row: row index, history [{title, category}], neighbour statistics
  gold.jsonl           per row: target title + category (scoring only, never sent to the LLM)
  baselines.jsonl      per row: last-item category, classifier top-3, kNN-50 top-3

Neighbour statistics: TF-IDF over history titles, 50 most similar TRAINING rows of other users
(training precedes validation in time); similarity-weighted shares of their next items' categories.
"""

import ast
import json
import os
import random
from collections import Counter

import fire
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

R3 = "/common/home/users/m/mk.ngohuu.2024/research/R3"
STEM = "Toys_and_Games_5_2017-1-2018-11"


def load_paths(data, frames):
    need = set()
    for df in frames:
        need |= set(df["item_asin"])
        for x in df["item_asins"]:
            need |= set(ast.literal_eval(x))
    paths = {}
    with open(f"{data}/meta_Toys_and_Games.json") as handle:
        for line in handle:
            m = json.loads(line)
            if m.get("asin") in need and m["asin"] not in paths:
                paths[m["asin"]] = " > ".join((m.get("category") or [])[1:]) or "unknown"
    return paths


def main(out, n=500, seed=0, k=50, data=f"{R3}/data/Toys_and_Games"):
    os.makedirs(out, exist_ok=True)
    tr = pd.read_csv(f"{data}/train/{STEM}.csv")
    va = pd.read_csv(f"{data}/valid/{STEM}.csv")
    paths = load_paths(data, (tr, va))
    P = lambda a: paths.get(a, "unknown")
    taxonomy = sorted({p for p in paths.values() if p != "unknown"})
    json.dump(taxonomy, open(f"{out}/taxonomy.json", "w"), indent=0)

    rows = sorted(random.Random(seed).sample(range(len(va)), n))
    sub = va.iloc[rows]

    # kNN neighbour statistics (other users' training rows)
    docs_tr = [" ".join(ast.literal_eval(h)) for h in tr["history_item_title"]]
    docs_va = [" ".join(ast.literal_eval(h)) for h in sub["history_item_title"]]
    vec = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True).fit(docs_tr)
    S = (vec.transform(docs_va) @ vec.transform(docs_tr).T).toarray()
    tr_cat = np.array([P(a) for a in tr["item_asin"]])
    users = tr["user_id"].values

    # learned classifier baseline (history titles + history categories + last category)
    def feats(df):
        t = [" ".join(ast.literal_eval(h)) for h in df["history_item_title"]]
        c = [" ".join("C_" + P(a).replace(" ", "_") for a in ast.literal_eval(h)) for h in df["item_asins"]]
        l = ["L_" + P(ast.literal_eval(h)[-1]).replace(" ", "_") for h in df["item_asins"]]
        return t, c, l
    ft, fc, fl = feats(tr)
    v1 = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True, max_features=50000).fit(ft)
    v2 = TfidfVectorizer(token_pattern=r"\S+", lowercase=False).fit(fc)
    v3 = TfidfVectorizer(token_pattern=r"\S+", lowercase=False).fit(fl)
    X = lambda t, c, l: hstack([v1.transform(t), v2.transform(c), v3.transform(l)]).tocsr()
    clf = SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=30, tol=1e-4, random_state=0, n_jobs=8)
    clf.fit(X(ft, fc, fl), tr_cat)
    proba = clf.predict_proba(X(*feats(sub)))

    with open(f"{out}/inputs.jsonl", "w") as fi, open(f"{out}/gold.jsonl", "w") as fg, \
            open(f"{out}/baselines.jsonl", "w") as fb:
        for r, (row_index, row) in enumerate(zip(rows, sub.itertuples())):
            titles = ast.literal_eval(row.history_item_title)
            asins = ast.literal_eval(row.item_asins)
            history = [{"title": t, "category": P(a)} for t, a in zip(titles, asins)]
            s = S[r].copy()
            s[users == row.user_id] = -1
            nb = np.argpartition(-s, k)[:k]
            votes = Counter()
            for j in nb:
                if s[j] > 0:
                    votes[tr_cat[j]] += s[j]
            total = sum(votes.values()) or 1.0
            stats = [{"category": c, "share": round(v / total, 3)} for c, v in votes.most_common(5)]
            fi.write(json.dumps({"row": row_index, "history": history, "neighbour_next_categories": stats}) + "\n")
            fg.write(json.dumps({"row": row_index, "title": row.item_title, "category": P(row.item_asin),
                                 "history_categories": sorted({P(a) for a in asins})}) + "\n")
            top3 = list(clf.classes_[np.argsort(-proba[r])[:3]])
            fb.write(json.dumps({"row": row_index, "last_item": P(asins[-1]), "classifier_top3": top3,
                                 "knn_top3": [x["category"] for x in stats[:3]]}) + "\n")
    print(json.dumps({"rows": n, "taxonomy": len(taxonomy), "out": out}))


if __name__ == "__main__":
    fire.Fire(main)
