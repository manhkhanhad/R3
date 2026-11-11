"""Split GEARec v2 reasoning (best candidate per row) into a HISTORY part and a TARGET part.

TARGET part = from the first sentence (after the first) containing a forward-looking cue ("next",
"predict", "likely ... seek/want/...", "would ...", "probably", "future", "seeking", "looking for") to the
end; fallback = the last sentence. HISTORY part = the sentences before. Rows without reasoning get "".

Writes <out>/<split>.json: {"history_part": [...], "target_part": [...], "split_by": ["cue"|"last"|""]}
aligned with the R3 CSV rows. (At real inference the history part must be regenerated without the target;
here the target-aware history part is used as a stand-in.)
"""

import json
import os
import re
from collections import Counter

import fire

R3 = "/common/home/users/m/mk.ngohuu.2024/research/R3"
SRC = R3 + "/output_dir/rationale_sft_runs/rationale_sft/gearec_v2/best"
CUE = re.compile(r"\b(next|predict\w*|anticipat\w*|likely (?:to )?(?:be )?(?:seek|look|want|interest|purchas|buy|search|desir|"
                 r"appreciat|enjoy|choos|select|acquir|add|expand|continu)\w*|would (?:likely |probably |be )?\w+|"
                 r"will (?:likely |probably )?\w+|probably|future|upcoming|seeking|looking for|in the market)\b", re.I)
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


def split(text):
    if not text:
        return "", "", ""
    s = SENT.split(text.strip())
    if len(s) == 1:
        return "", s[0], "last"
    for i, x in enumerate(s[1:], 1):
        if CUE.search(x):
            return " ".join(s[:i]), " ".join(s[i:]), "cue"
    return " ".join(s[:-1]), s[-1], "last"


def main(out=f"{R3}/output_dir/reasoning_geometry/parts", src=SRC):
    os.makedirs(out, exist_ok=True)
    for split_name in ("train", "valid", "test"):
        texts = json.load(open(f"{src}/{split_name}.json"))["rationales"]
        h, t, by = zip(*(split(x) for x in texts))
        json.dump({"history_part": list(h), "target_part": list(t), "split_by": list(by)},
                  open(f"{out}/{split_name}.json", "w"))
        c = Counter(by)
        n = sum(bool(x) for x in texts)
        print(f"{split_name}: rows {len(texts)}, with reasoning {n}, split by cue {c['cue'] / n:.1%}, by last sentence {c['last'] / n:.1%}")


if __name__ == "__main__":
    fire.Fire(main)
