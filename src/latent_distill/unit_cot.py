"""Rationale units (external_rational) as LaRec-style teacher steps (document/latent_distillation.md).

external_rational/run_full_dataset.py writes one JSON per training row: 0-3 structured,
target-free rationale units grounded in item metadata (evidence quotes from categories,
features, brands), each with evidence / interpretation / recommendation components.
This script maps them onto the step-JSON schema of generate_cot.py so teacher_states.py
can extract alignment states, with K = 3 fixed steps per row:

    step 1  Evidence        every unit's evidence statements, in order
    step 2  Interpretation  every unit's interpretation
    step 3  Recommendation  every unit's recommendation

Rows whose generation failed, did not parse, or produced zero units get steps = null and
are excluded from the alignment loss (like unparsed/leaking rows in generate_cot.py).
Unlike LaRec's CoT the units never saw the target, so there is nothing to redact and
"leak" is always false.

Usage: python src/latent_distill/unit_cot.py --generations <dir>/generation/generations \
           --csv <train csv> --out <dir>/train_cot.json
"""

import json
import os
import re

import fire
import pandas as pd

LABELS = ("Evidence", "Interpretation", "Recommendation")

SYSTEM = ("You are an expert shopping analyst. You explain, in three labelled parts, what a "
          "customer's purchase history suggests about their next purchase.")


def user_prompt(titles, k=3):
    lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
    return (f"Purchase history, oldest first:\n{lines}\n\nWrite three labelled parts. \"Evidence:\" cites "
            f"observable facts about the items in the history. \"Interpretation:\" connects them to the "
            f"customer's preferences. \"Recommendation:\" describes the kind of product likely bought next "
            f"(attributes and direction only, never a specific product name).")


def _sentence(text):
    text = " ".join(str(text).split())
    return text if not text or text[-1] in ".!?" else text + "."


def unit_steps(parsed):
    """[evidence, interpretation, recommendation] texts across the row's units, or None."""
    units = (parsed or {}).get("rationales") or []
    evidence = [_sentence(e["statement"]) for u in units for e in u.get("evidence", []) if e.get("statement")]
    interpretation = [_sentence(u["interpretation"]) for u in units if u.get("interpretation")]
    recommendation = [_sentence(u["recommendation"]) for u in units if u.get("recommendation")]
    if not (evidence and interpretation and recommendation):
        return None
    return [" ".join(evidence), " ".join(interpretation), " ".join(recommendation)]


def main(generations: str, csv: str, out: str):
    frame = pd.read_csv(csv, usecols=["history_item_title"])
    steps = [None] * len(frame)
    counts = {"missing": 0, "error": 0, "empty": 0, "ok": 0}
    for name in sorted(os.listdir(generations)):
        m = re.fullmatch(r"(\d+)\.json", name)
        if not m:
            continue
        with open(os.path.join(generations, name)) as handle:
            record = json.load(handle)
        row = int(record["history_id"].rsplit(":", 1)[1])
        if row != int(m.group(1)):
            raise ValueError(f"{name} holds history row {row}")
        if row >= len(frame):  # a smoke CSV holds only the first rows
            continue
        if record.get("status") != "ok":
            counts["error"] += 1
            continue
        steps[row] = unit_steps(record.get("parsed"))
        counts["ok" if steps[row] else "empty"] += 1
    counts["missing"] = len(frame) - sum(counts.values())
    print(f"{len(frame)} rows: {counts} -> {counts['ok']} usable")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as handle:
        json.dump({"source": os.path.abspath(generations), "csv": os.path.abspath(csv), "k": 3,
                   "prompt_mode": "unit", "steps": steps, "leak": [False] * len(steps)}, handle)


if __name__ == "__main__":
    fire.Fire(main)
