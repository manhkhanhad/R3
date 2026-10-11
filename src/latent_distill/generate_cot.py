"""Target-aware step-wise CoT for LaRec-style latent pre-training (document/latent_distillation.md).

LaRec (arXiv 2607.24617, Sec. 3.2.1) prompts a teacher LLM with the user history and
the ground-truth next item and asks for a reasoning path C+ = {c_1..c_K} that derives
the next purchase from the history without revealing the target.  Its hidden states
after each step are the alignment targets of the K latent steps.  The teacher only
supervises training; it is never used at inference.

Output: JSON {"steps": [[c_1..c_K] or null per row], ...} aligned with the CSV rows; a
row whose output does not parse into exactly K steps, or that names the target, is null.

Usage: python src/latent_distill/generate_cot.py --model Qwen/Qwen2.5-7B-Instruct --csv <train csv> --out <dir>/train.json
"""

import ast
import json
import os
import re
import time

import fire
import pandas as pd

SYSTEM = ("You are an expert shopping analyst for musical instruments and accessories. You explain, "
          "step by step, how a customer's purchase history leads to their next purchase.")


def user_prompt(titles, target, k):
    lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
    return (f"Purchase history, oldest first:\n{lines}\n\nThe customer's next purchase was: {target}\n\n"
            f"Write exactly {k} numbered reasoning steps (\"Step 1:\" ... \"Step {k}:\"), one sentence each, that "
            f"start from the purchase history and progressively narrow down to the kind of product bought next. "
            f"Every step must be justified by the history alone, as if you did not know the answer. Never write "
            f"the next product's name, brand, or model.")


STEP = re.compile(r"Step\s*(\d+)\s*[:.)-]\s*(.+?)(?=\s*Step\s*\d+\s*[:.)-]|\Z)", re.S | re.I)


def parse_steps(text, k):
    found = [(int(n), " ".join(s.split())) for n, s in STEP.findall(text)]
    steps = [s for n, s in found if 1 <= n <= k]
    if [n for n, _ in found][:k] != list(range(1, k + 1)) or len(steps) != k or any(not s for s in steps):
        return None
    return steps


def leaks(steps, target):
    """True if the steps contain the target's first distinctive word (brand / model) or a long substring."""
    t = target.lower()
    words = [w for w in re.findall(r"[a-z0-9']+", t) if len(w) > 3]
    joined = " ".join(steps).lower()
    head = words[0] if words else ""
    return (head and re.search(rf"\b{re.escape(head)}\b", joined) is not None) or t[:30] in joined


def main(model: str, csv: str, out: str, k: int = 6, limit: int = -1, max_tokens: int = 384,
         temperature: float = 0.0, seed: int = 0, gpu_memory_utilization: float = 0.85, max_model_len: int = 2048):
    from vllm import LLM, SamplingParams

    frame = pd.read_csv(csv, usecols=["history_item_title", "item_title"])
    if limit > 0:
        frame = frame.iloc[:limit]
    histories = [list(ast.literal_eval(h)) for h in frame.history_item_title]
    targets = [str(t) for t in frame.item_title]

    llm = LLM(model=model, dtype="bfloat16", seed=seed, gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=max_model_len, enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()
    prompts = [tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
                                              {"role": "user", "content": user_prompt(h, t, k)}],
                                             tokenize=False, add_generation_prompt=True)
               for h, t in zip(histories, targets)]
    start = time.time()
    outputs = llm.generate(prompts, SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed))
    steps, unparsed, leaked = [], 0, 0
    for o, t in zip(outputs, targets):
        s = parse_steps(o.outputs[0].text, k)
        if s is None:
            unparsed += 1
        elif leaks(s, t):
            leaked += 1
            s = None
        steps.append(s)
    print(f"{len(steps)} rows in {time.time() - start:.0f}s; unparsed {unparsed}, leaking the target {leaked}, "
          f"kept {sum(s is not None for s in steps)}")
    for h, t, s in list(zip(histories, targets, steps))[:3]:
        print("HISTORY:", h, "\nTARGET:", t, "\nSTEPS:", s, "\n")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as handle:
        json.dump({"model": model, "csv": os.path.abspath(csv), "k": k, "steps": steps}, handle)


if __name__ == "__main__":
    fire.Fire(main)
