"""Teacher hidden states h_k for LaRec-style step alignment (document/latent_distillation.md).

LaRec Eq. 3: h_k = LLM_last(prompt_cot(S_u, i_tgt), c_{1:k}).  The LLM is the frozen
backbone that latent pre-training starts from (Qwen2.5-1.5B-Instruct here, as R3 v1);
h_k comes from its final-layer (normed) states over step k's tokens.  The paper does not
give the token position; ``pool`` picks it:

* ``mean`` (default): the mean state over step k's tokens.  The state at a single token
  is dominated by that token's identity, and every step ends in the same ".\\n", which
  inflates the content-free shared component (random rows' last-token states have cosine
  0.94-0.99); averaging over the step's tokens suppresses it.
* ``last``: the state at step k's last token (the most literal reading of Eq. 3).

The steps sit in a chat as the assistant reply.  The prompt depends on the step JSON's
``prompt_mode``:

* ``cot`` (generate_cot.py): the target-aware LaRec prompt, steps as "Step k: ..." lines.
* ``unit`` (unit_cot.py): a history-only prompt, steps as "Evidence: ... / Interpretation:
  ... / Recommendation: ..." lines.  The exact prompt that produced the units
  (external_rational) spans the full metadata records; this compact one reads the same
  step texts in a comparable instruction context.

Output: {"states": [N, K, H] bfloat16, "valid": [N] bool, "k": K, "pool": pool}
"""

import json
import os
import sys

import fire
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_cot  # noqa: E402
import unit_cot  # noqa: E402


def chat_prompt(mode, history, target, k):
    if mode == "unit":
        system, user = unit_cot.SYSTEM, unit_cot.user_prompt(history, k)
    else:
        system, user = generate_cot.SYSTEM, generate_cot.user_prompt(history, target, k)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def step_labels(mode, k):
    return list(unit_cot.LABELS) if mode == "unit" else [f"Step {i + 1}" for i in range(k)]


def encode_row(tokenizer, history, target, steps, mode="cot"):
    """Token ids of the chat prompt + "<label>: c_1\\n..." and each step's token span [start, end]."""
    k = len(steps)
    prompt = tokenizer.apply_chat_template(chat_prompt(mode, history, target, k),
                                           tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    spans = []
    for i, (label, step) in enumerate(zip(step_labels(mode, k), steps)):
        start = len(ids)
        line = f"{label}: {step}" + ("\n" if i + 1 < k else "")
        ids = ids + tokenizer(line, add_special_tokens=False).input_ids
        spans.append((start, len(ids) - 1))
    return ids, spans


@torch.no_grad()
def main(cot: str, csv: str, out: str, model: str = "Qwen/Qwen2.5-1.5B-Instruct", batch_size: int = 32,
         pool: str = "mean"):
    if pool not in ("mean", "last"):
        raise ValueError(f"pool must be 'mean' or 'last', got {pool!r}")
    import ast
    with open(cot) as handle:
        spec = json.load(handle)
    k = int(spec["k"])
    mode = spec.get("prompt_mode", "cot")
    leak = spec.get("leak", [False] * len(spec["steps"]))
    steps = [s if s is not None and not l else None for s, l in zip(spec["steps"], leak)]
    frame = pd.read_csv(csv, usecols=["history_item_title", "item_title"])
    if len(frame) != len(steps):
        raise ValueError(f"{csv} has {len(frame)} rows, {cot} has {len(steps)}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).cuda().eval()
    hidden = llm.config.hidden_size

    states = torch.zeros(len(steps), k, hidden, dtype=torch.bfloat16)
    valid = torch.tensor([s is not None for s in steps])
    rows = [i for i in range(len(steps)) if steps[i] is not None]
    encoded = {i: encode_row(tokenizer, list(ast.literal_eval(frame.history_item_title[i])),
                             str(frame.item_title[i]), steps[i], mode) for i in rows}
    rows.sort(key=lambda i: len(encoded[i][0]))
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        width = max(len(encoded[i][0]) for i in batch)
        ids = torch.full((len(batch), width), tokenizer.pad_token_id or 0)
        mask = torch.zeros(len(batch), width, dtype=torch.long)
        for j, i in enumerate(batch):
            seq = encoded[i][0]
            ids[j, :len(seq)] = torch.tensor(seq)
            mask[j, :len(seq)] = 1
        last = llm.model(input_ids=ids.cuda(), attention_mask=mask.cuda()).last_hidden_state
        for j, i in enumerate(batch):
            if pool == "mean":
                states[i] = torch.stack([last[j, a:b + 1].mean(0) for a, b in encoded[i][1]]).to(torch.bfloat16).cpu()
            else:
                states[i] = last[j, [b for _, b in encoded[i][1]]].to(torch.bfloat16).cpu()
        if start // batch_size % 200 == 0:
            print(f"{start + len(batch)}/{len(rows)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save({"states": states, "valid": valid, "k": k, "model": model, "pool": pool, "prompt_mode": mode,
                "cot": os.path.abspath(cot)}, out)
    print(f"saved {out}: {int(valid.sum())}/{len(valid)} valid rows, states {tuple(states.shape)}")


if __name__ == "__main__":
    fire.Fire(main)
