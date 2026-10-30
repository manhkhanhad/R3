"""Self-checks for the gist bottleneck (latent_bottleneck.py, ``"gist": True``), fp32 base model.

1. Leak: perturbing one history token (same length) with the latent tokens deleted
   from the target side leaves the target log-probs unchanged; with the latents
   visible they change (the latents read the history).
2. KV-cache decoding reproduces teacher-forced log-probs.
"""
import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import torch
from transformers import AutoConfig, AutoTokenizer

from layers import LatentModel
from reasoning_dataset import LatentRDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rows", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.add_special_tokens({"additional_special_tokens": tok.additional_special_tokens + ["<|Thought|>"]})
    data = LatentRDataset(train_file=args.test_file, tokenizer=tok, max_len=2560, sample=args.rows,
                          category="toys and games", n_thoughts=args.k)
    config = AutoConfig.from_pretrained(args.base_model)
    config.latent_continuous_thoughts = {"k": args.k, "source": "constant", "token_norm": 1.0}
    config.latent_bottleneck = {"visible_prefix_ids": [int(t) for t in data.instruction_tokens()], "gist": True}
    model = LatentModel.from_pretrained(args.base_model, config=config, torch_dtype=torch.float32).cuda().eval()
    model.resize_token_embeddings(len(tok))
    model.thoughts.token_norm.fill_(float(model.model.embed_tokens.weight.norm(dim=-1).mean()))
    model.thoughts.reset_parameters()
    n_prefix = len(config.latent_bottleneck["visible_prefix_ids"])

    def logprobs(ids, labels):
        with torch.no_grad():
            logits = model(input_ids=ids[None].cuda(), attention_mask=torch.ones(1, ids.numel(), dtype=torch.long, device="cuda")).logits[0]
        pos = torch.nonzero(labels != -100).squeeze(1)
        return torch.log_softmax(logits[pos - 1].float(), -1)

    leak, used = 0.0, 0.0
    for i in range(len(data)):
        row = data[i]
        ids, labels = torch.tensor(row["input_ids"]), torch.tensor(row["labels"])
        thought = int(torch.nonzero(ids == len(tok) - 1)[0])
        j = n_prefix + (thought - n_prefix) // 2  # a token inside the user span
        other = ids.clone()
        other[j] = (int(ids[j]) + 17) % 150000
        config.latent_bottleneck["delete_latents"] = True
        leak = max(leak, float((logprobs(ids, labels) - logprobs(other, labels)).abs().max()))
        config.latent_bottleneck["delete_latents"] = False
        used = max(used, float((logprobs(ids, labels) - logprobs(other, labels)).abs().max()))
    print(f"[1] history perturbation: latents deleted max |dlogp| {leak:.2e}; latents visible {used:.2e}")
    if leak > 1e-3 or used < 1e-3:
        raise AssertionError("gist mask leaks the history or the latents do not read it")

    row = data[0]
    ids, labels = torch.tensor(row["input_ids"]).cuda(), torch.tensor(row["labels"])
    prompt_len = int((labels == -100).sum())
    teacher = logprobs(ids.cpu(), labels)
    with torch.no_grad():
        out = model(input_ids=ids[None, :prompt_len], attention_mask=torch.ones(1, prompt_len, dtype=torch.long, device="cuda"), use_cache=True)
        cache, steps = out.past_key_values, [torch.log_softmax(out.logits[0, -1].float(), -1)]
        for t in range(prompt_len, ids.numel() - 1):
            out = model(input_ids=ids[None, t:t + 1], attention_mask=torch.ones(1, t + 1, dtype=torch.long, device="cuda"),
                        past_key_values=cache, use_cache=True, cache_position=torch.tensor([t], device="cuda"))
            cache = out.past_key_values
            steps.append(torch.log_softmax(out.logits[0, -1].float(), -1))
    d = float((torch.stack(steps) - teacher).abs().max())
    print(f"[2] KV-cache decoding vs teacher forcing max |dlogp| {d:.2e}")
    if d > 1e-3:
        raise AssertionError("decoding mask differs from the training mask")
    print("gist checks passed")


if __name__ == "__main__":
    main()
