"""Render external rationale units (external_rational/run_full_dataset.py) as one text per CSV row.

The generator writes ``generation/generations/NNNNNN.json`` for CSV row NNNNNN of the split it
was run on, holding 0-3 validated, target-free units under ``parsed.rationales``. This module
turns each row's units into a single string and saves them aligned with the CSV:

    {"split", "csv", "generations", "counts", "rationales": [str, ...]}   (one entry per CSV row)

Rows whose generation failed or abstained get the empty string; the dataset renders those as
``none`` so every row keeps the same prompt layout.

Usage:
    python src/latent/rationale_text.py build \
        --generations <run>/generation/generations --csv <split csv> --out <dir>/<split>.json
"""

import json
import os

import fire
import pandas as pd

NONE_TEXT = "none"


def _clean(text):
    return " ".join(str(text or "").split())


def _sentence(text):
    text = _clean(text)
    return text if not text or text[-1] in ".!?" else text + "."


def unit_text(unit):
    """One unit as 'Facet: ... Evidence: ... Interpretation: ... Recommendation: ...'."""
    evidence = " ".join(_sentence(e.get("statement")) for e in unit.get("evidence") or [] if e.get("statement"))
    parts = [("Facet", unit.get("facet")), ("Evidence", evidence),
             ("Interpretation", unit.get("interpretation")), ("Recommendation", unit.get("recommendation"))]
    return " ".join(f"{name}: {_sentence(value)}" for name, value in parts if _clean(value))


def row_text(record):
    """All of a row's units, numbered; '' when the row has none (error or abstention)."""
    if not record or record.get("status") != "ok":
        return ""
    units = ((record.get("parsed") or {}).get("rationales")) or []
    texts = [unit_text(u) for u in units]
    texts = [t for t in texts if t]
    if len(texts) == 1:
        return texts[0]
    return " ".join(f"({i + 1}) {t}" for i, t in enumerate(texts))


def build(generations, csv, out):
    n_rows = len(pd.read_csv(csv, usecols=["user_id"]))
    counts = {"rows": n_rows, "ok": 0, "abstained": 0, "error": 0, "missing": 0}
    rationales, split = [], None
    for i in range(n_rows):
        path = os.path.join(generations, f"{i:06d}.json")
        if not os.path.exists(path):
            counts["missing"] += 1
            rationales.append("")
            continue
        record = json.load(open(path))
        # history_id is "<dataset>:<split>:<csv row>"; check alignment instead of trusting file names.
        _, row_split, row_index = record["history_id"].rsplit(":", 2)
        if int(row_index) != i:
            raise ValueError(f"{path} holds {record['history_id']}, expected row {i}")
        if split is None:
            split = row_split
        elif split != row_split:
            raise ValueError(f"mixed splits in {generations}: {split} vs {row_split}")
        text = row_text(record)
        if text:
            counts["ok"] += 1
        elif record.get("status") == "ok":
            counts["abstained"] += 1
        else:
            counts["error"] += 1
        rationales.append(text)
    expected = os.path.basename(os.path.dirname(os.path.abspath(csv)))
    if split is not None and split != expected:
        raise ValueError(f"generations are for split {split!r} but the CSV is in {expected!r}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as handle:
        json.dump({"split": split or expected, "csv": os.path.abspath(csv),
                   "generations": os.path.abspath(generations), "counts": counts,
                   "rationales": rationales}, handle)
    print(json.dumps(counts))
    return counts


def categories(csv, meta, out):
    """ORACLE ablation input: each row's TARGET item category path (e.g. 'Games > Board Games').

    Uses the metadata 'category' list without its root ('Toys & Games'); rows whose target has no
    category get 'unknown'. Deliberately leaks target information; never use it as a method input.
    """
    df = pd.read_csv(csv, usecols=["item_asin"])
    need = set(df["item_asin"])
    paths = {}
    with open(meta) as handle:
        for line in handle:
            m = json.loads(line)
            a = m.get("asin")
            if a in need and a not in paths:
                paths[a] = " > ".join((m.get("category") or [])[1:])
    cats = [paths.get(a) or "unknown" for a in df["item_asin"]]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as handle:
        json.dump({"csv": os.path.abspath(csv), "kind": "target_category_ORACLE", "categories": cats}, handle)
    print(json.dumps({"rows": len(cats), "unknown": cats.count("unknown"), "distinct": len(set(cats))}))


def load_categories(path, csv):
    with open(path) as handle:
        cats = json.load(handle)["categories"]
    n_rows = len(pd.read_csv(csv, usecols=["user_id"]))
    if len(cats) != n_rows:
        raise ValueError(f"{path} has {len(cats)} categories for {n_rows} rows of {csv}")
    return cats


def load(path, csv):
    """Rationale list aligned with ``csv`` (length checked)."""
    with open(path) as handle:
        payload = json.load(handle)
    rationales = payload["rationales"]
    n_rows = len(pd.read_csv(csv, usecols=["user_id"]))
    if len(rationales) != n_rows:
        raise ValueError(f"{path} has {len(rationales)} rationales for {n_rows} rows of {csv}")
    return rationales


if __name__ == "__main__":
    fire.Fire({"build": build, "categories": categories})
