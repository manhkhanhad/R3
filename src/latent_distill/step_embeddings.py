"""Teacher step targets from a text-embedding model (document/latent_distillation.md).

An alternative to teacher_states.py: instead of the backbone LM's final-layer state in
the CoT context, each step text is embedded standalone by a sentence encoder
(Alibaba-NLP/gte-large-en-v1.5, CLS pooling + L2 normalisation, as in
category_alignment_data.py).  A dedicated embedder has far weaker anisotropy than the
LM's final layer, and its space is built for cosine comparison; the price is that the
targets no longer live in the student's own representation space (LaRec Eq. 3 uses the
student LLM), so step alignment maps z_k through a learned linear head
(larec_latent.py) when the dimensions differ.

``--pooling last`` is the Qwen3-Embedding recipe used for the item-title and
explanation tables (explanation_embeddings.py): left padding, the state of the final
token (the tokenizer appends <|endoftext|>), truncated to ``--dim`` (Matryoshka) and
L2-normalised; documents get no instruction prefix.  Qwen3 needs transformers >= 4.51
(the racer env).

Output: {"states": [N, K, D] float32 (unit norm), "valid": [N] bool, "k": K,
         "model", "pool": "cls", "cot"}  -- same schema as teacher_states.py.

Usage: python src/latent_distill/step_embeddings.py --cot <dir>/train_cot.json --out <dir>/train_states_gte.pt
"""

import json
import os

import fire
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@torch.inference_mode()
def main(cot: str, out: str, model: str = "Alibaba-NLP/gte-large-en-v1.5", batch_size: int = 64,
         pooling: str = "cls", dim: int = 0):
    if pooling not in ("cls", "last"):
        raise ValueError("pooling must be cls or last")
    with open(cot) as handle:
        spec = json.load(handle)
    k = int(spec["k"])
    leak = spec.get("leak", [False] * len(spec["steps"]))
    steps = [s if s is not None and not l else None for s, l in zip(spec["steps"], leak)]

    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left" if pooling == "last" else "right")
    encoder = AutoModel.from_pretrained(model, trust_remote_code=True,
                                        torch_dtype=torch.bfloat16 if pooling == "last" else None).cuda().eval()
    width = dim or encoder.config.hidden_size

    states = torch.zeros(len(steps), k, width, dtype=torch.float32)
    valid = torch.tensor([s is not None for s in steps])
    jobs = [(i, j) for i in range(len(steps)) if steps[i] is not None for j in range(k)]
    jobs.sort(key=lambda ij: len(steps[ij[0]][ij[1]]))
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        tokens = tokenizer([steps[i][j] for i, j in batch], padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        output = encoder(**{key: value.cuda() for key, value in tokens.items()})
        pooled = output.last_hidden_state[:, 0 if pooling == "cls" else -1].float()
        embeddings = F.normalize(pooled[:, :width], dim=-1).cpu()
        for (i, j), e in zip(batch, embeddings):
            states[i, j] = e
        if start // batch_size % 200 == 0:
            print(f"{start + len(batch)}/{len(jobs)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save({"states": states, "valid": valid, "k": k, "model": model, "pool": pooling,
                "cot": os.path.abspath(cot)}, out)
    print(f"saved {out}: {int(valid.sum())}/{len(valid)} valid rows, states {tuple(states.shape)}")


if __name__ == "__main__":
    fire.Fire(main)
