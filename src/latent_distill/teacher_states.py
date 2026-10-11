"""Teacher hidden states h_k for LaRec-style step alignment (document/latent_distillation.md).

LaRec Eq. 3: h_k = LLM_last(prompt_cot(S_u, i_tgt), c_{1:k}).  The LLM is the frozen
backbone that latent pre-training starts from (Qwen2.5-1.5B-Instruct here, as R3 v1);
h_k is its final-layer (normed) state at the last token of step k.  The CoT prompt is
the one that produced the steps (generate_cot.py), as a chat with the steps as the
assistant reply.

Output: {"states": [N, K, H] bfloat16, "valid": [N] bool, "k": K}
"""

import json
import os
import sys

import fire
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_cot import SYSTEM, user_prompt  # noqa: E402


def encode_row(tokenizer, history, target, steps):
    """Token ids of the chat prompt + "Step 1: c_1\\n..." and the index of each step's last token."""
    k = len(steps)
    prompt = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
                                            {"role": "user", "content": user_prompt(history, target, k)}],
                                           tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    ends = []
    for i, step in enumerate(steps):
        ids = ids + tokenizer(f"Step {i + 1}: {step}" + ("\n" if i + 1 < k else ""), add_special_tokens=False).input_ids
        ends.append(len(ids) - 1)
    return ids, ends


@torch.no_grad()
def main(cot: str, csv: str, out: str, model: str = "Qwen/Qwen2.5-1.5B-Instruct", batch_size: int = 32):
    import ast
    with open(cot) as handle:
        spec = json.load(handle)
    k = int(spec["k"])
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
                             str(frame.item_title[i]), steps[i]) for i in rows}
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
            states[i] = last[j, encoded[i][1]].to(torch.bfloat16).cpu()
        if start // batch_size % 200 == 0:
            print(f"{start + len(batch)}/{len(rows)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save({"states": states, "valid": valid, "k": k, "model": model, "cot": os.path.abspath(cot)}, out)
    print(f"saved {out}: {int(valid.sum())}/{len(valid)} valid rows, states {tuple(states.shape)}")


if __name__ == "__main__":
    fire.Fire(main)
