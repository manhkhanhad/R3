"""Target-aware step-wise CoT for LaRec-style latent pre-training (document/latent_distillation.md).

LaRec (arXiv 2607.24617, Sec. 3.2.1) prompts a teacher LLM with the user history and
the ground-truth next item and asks for a reasoning path C+ = {c_1..c_K} that derives
the next purchase from the history without revealing the target.  Its hidden states
after each step are the alignment targets of the K latent steps.  The teacher only
supervises training; it is never used at inference.

Output: JSON aligned with the CSV rows: "steps" ([c_1..c_K] with the target's identifiers
redacted, or null when the output does not parse into exactly K steps), "leak" (whether
identifiers remain after redaction) and "raw" (the generated text).  teacher_states.py uses rows with steps and no leak.

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
            f"Every step must be justified by the history alone, as if you did not know the answer. The last step "
            f"describes only the type of product (category and key features). Never write the next product's "
            f"name, model, or a brand that does not appear in the history.")


STEP = re.compile(r"Step\s*(\d+)\s*[:.)-]\s*(.+?)(?=\s*Step\s*\d+\s*[:.)-]|\Z)", re.S | re.I)


def parse_steps(text, k):
    found = [(int(n), " ".join(s.split())) for n, s in STEP.findall(text)]
    steps = [s for n, s in found if 1 <= n <= k]
    if [n for n, _ in found][:k] != list(range(1, k + 1)) or len(steps) != k or any(not s for s in steps):
        return None
    return steps


def words(text):
    return re.findall(r"[a-z0-9][a-z0-9'./-]*", text.lower())


def _pattern(phrase):
    return rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"


def identifiers(target, history=()):
    """Target-identifying phrases absent from the history, longest first: runs of >= 4 title
    words, model-number tokens, and the title's first word (brand).  Shorter phrases
    (e.g. "padded cymbal bag") describe the product type, which the CoT is meant to reach."""
    seen = " ".join(history).lower()
    title = words(target)
    runs = [" ".join(title[i:j]) for i in range(len(title)) for j in range(len(title), i + 3, -1)]
    models = [w for w in title if any(c.isdigit() for c in w) and len(w) > 2]
    brand = [title[0]] if title and len(title[0]) > 2 else []
    found = {p for p in runs + models + brand if re.search(_pattern(p), seen) is None}
    return sorted(found, key=len, reverse=True)


def redact(steps, target, history=()):
    """Steps with the target's identifiers replaced (title runs) or dropped (model, brand)."""
    ids = identifiers(target, history)
    out = []
    for step in steps:
        text = step
        for p in ids:
            repl = "the product" if " " in p else ""
            text = re.sub(_pattern(p), repl, text, flags=re.I)
        out.append(" ".join(text.split()))
    return out


def leaks(steps, target, history=()):
    """Whether the steps contain any of the target's identifiers (see ``identifiers``)."""
    said = " ".join(steps).lower()
    return any(re.search(_pattern(p), said) for p in identifiers(target, history))


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
    raw = [o.outputs[0].text for o in outputs]
    steps = [parse_steps(r, k) for r in raw]
    named = [s is not None and leaks(s, t, h) for s, t, h in zip(steps, targets, histories)]
    steps = [None if s is None else redact(s, t, h) for s, t, h in zip(steps, targets, histories)]
    leak = [s is not None and leaks(s, t, h) for s, t, h in zip(steps, targets, histories)]
    print(f"{len(steps)} rows in {time.time() - start:.0f}s; unparsed {sum(s is None for s in steps)}, "
          f"named the target (redacted) {sum(named)}, still leaking after redaction {sum(leak)}, "
          f"kept {sum(s is not None and not l for s, l in zip(steps, leak))}")
    for h, t, s in list(zip(histories, targets, steps))[:3]:
        print("HISTORY:", h, "\nTARGET:", t, "\nSTEPS:", s, "\n")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as handle:
        json.dump({"model": model, "csv": os.path.abspath(csv), "k": k, "steps": steps, "leak": leak, "raw": raw}, handle)


if __name__ == "__main__":
    fire.Fire(main)
