"""Score LLM next-category predictions against baselines on the same validation rows."""

import difflib
import json
import os

import fire


def norm_to_taxonomy(cat, taxonomy, lookup):
    if cat in lookup:
        return cat
    close = difflib.get_close_matches(cat, taxonomy, n=1, cutoff=0.85)
    return close[0] if close else None


def main(pilot_dir, variants="A,B,B_think"):
    taxonomy = json.load(open(os.path.join(pilot_dir, "taxonomy.json")))
    lookup = set(taxonomy)
    gold = {g["row"]: g for g in map(json.loads, open(os.path.join(pilot_dir, "gold.jsonl")))}
    base = {b["row"]: b for b in map(json.loads, open(os.path.join(pilot_dir, "baselines.jsonl")))}
    rows = sorted(gold)
    stay = {r for r in rows if gold[r]["category"] in gold[r]["history_categories"]}
    groups = {"all": rows, "stays in a history category": [r for r in rows if r in stay],
              "switches to a new category": [r for r in rows if r not in stay]}
    print(f"rows {len(rows)}: stay {len(stay)}, switch {len(rows) - len(stay)}; gold 'unknown' "
          f"{sum(gold[r]['category'] == 'unknown' for r in rows)}")

    methods = {"last-item category": {r: [base[r]["last_item"]] for r in rows},
               "learned classifier": {r: base[r]["classifier_top3"] for r in rows},
               "kNN-50 neighbour vote": {r: base[r]["knn_top3"] for r in rows}}
    for v in variants.split(","):
        path = os.path.join(pilot_dir, f"{v}.jsonl")
        if not os.path.exists(path):
            continue
        preds, invalid, missing = {}, 0, 0
        for rec in map(json.loads, open(path)):
            if not rec.get("predictions"):
                continue
            cats = []
            for p in rec["predictions"]:
                c = norm_to_taxonomy(p["category"], taxonomy, lookup)
                invalid += c is None
                if c and c not in cats:
                    cats.append(c)
            preds[rec["row"]] = cats
        missing = sum(r not in preds for r in rows)
        methods[f"LLM {v}"] = {r: preds.get(r, []) for r in rows}
        print(f"LLM {v}: parsed {len(preds)}/{len(rows)}, off-list categories {invalid}, missing {missing}")

    header = "method".ljust(26) + "".join(f"| {g[:24]:24s} top-1  top-3 " for g in groups)
    print(header)
    for name, pred in methods.items():
        cells = []
        for g, rs in groups.items():
            t1 = sum(bool(pred[r]) and pred[r][0] == gold[r]["category"] for r in rs) / len(rs)
            t3 = sum(gold[r]["category"] in pred[r][:3] for r in rs) / len(rs)
            cells.append(f"| {'':24s} {t1:5.1%}  {t3:5.1%} ")
        print(name.ljust(26) + "".join(cells))


if __name__ == "__main__":
    fire.Fire(main)
