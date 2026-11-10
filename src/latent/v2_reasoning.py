"""GEARec v2 TARGET-AWARE reasoning -> per-CSV-row rationale files for rationale_sft_train.py.

Source: GEARec_R3_toys/v2_target_reasoning/<run>/judged.jsonl (one record per CSV row, sample_id = row).
The generator saw the history AND the next item; valid/test reasoning is a reference explanation, so
feeding it at test time is an ORACLE, not a method.

Variants (written as <out>/<variant>/<split>.json, format of rationale_text.build):
  final : best candidate's reasoning only for rows passing every check (rank <= 100 and judge) = final.jsonl
  best  : best candidate's reasoning for every row that has one (V1 format), regardless of rank / judge

Usage: python src/latent/v2_reasoning.py --out <dir>
"""

import json
import os

import fire
import pandas as pd

V2 = "/common/home/users/m/mk.ngohuu.2024/research/GEARec_R3_toys/v2_target_reasoning"
DATA = "/common/home/users/m/mk.ngohuu.2024/research/R3/data/Toys_and_Games"
STEM = "Toys_and_Games_5_2017-1-2018-11"
RUNS = {"train": "full_train_bo8_fullhist_r100", "valid": "full_valid_bo8_r100", "test": "full_test_bo8_r100"}


def clean(text):
    return " ".join(str(text or "").split())


def main(out, v2=V2, data=DATA):
    for split, run in RUNS.items():
        csv = f"{data}/{split}/{STEM}.csv"
        df = pd.read_csv(csv, usecols=["user_id", "item_id"])
        n = len(df)
        final_ids = {json.loads(l)["sample_id"] for l in open(f"{v2}/{run}/final.jsonl")}
        texts = {"final": [""] * n, "best": [""] * n}
        seen = 0
        with open(f"{v2}/{run}/judged.jsonl") as handle:
            for line in handle:
                s = json.loads(line)
                i = s["sample_id"]
                if df.iloc[i]["user_id"] != s["user_id"] or int(df.iloc[i]["item_id"]) != s["target_id"]:
                    raise ValueError(f"{split} row {i}: judged record does not match the CSV row")
                seen += 1
                reasoning = clean((s.get("best") or {}).get("reasoning"))
                if not reasoning:
                    continue
                texts["best"][i] = reasoning
                if i in final_ids:
                    if not s.get("pass_v4"):
                        raise ValueError(f"{split} row {i}: in final.jsonl but pass_v4 is false")
                    texts["final"][i] = reasoning
        if seen != n:
            raise ValueError(f"{split}: judged.jsonl has {seen} rows, CSV has {n}")
        for variant, rationales in texts.items():
            path = f"{out}/{variant}/{split}.json"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            filled = sum(bool(t) for t in rationales)
            counts = {"rows": n, "with_reasoning": filled, "none": n - filled}
            json.dump({"split": split, "csv": csv, "source": f"{v2}/{run}", "variant": variant,
                       "kind": "GEARec_v2_target_aware_reasoning", "counts": counts, "rationales": rationales},
                      open(path, "w"))
            print(f"{variant:5s} {split:5s} {json.dumps(counts)}")


if __name__ == "__main__":
    fire.Fire(main)
