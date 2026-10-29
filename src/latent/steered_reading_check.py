"""Self-checks for steered_reading.py (B1).

1. Token -> item assignment: the tokens of every history item decode to that
   item's quoted title (plus its separator).
2. Mask = deletion: with the thought fixed, target log-probs under the mask
   (rule "recent", k items) equal those of a prompt from which the dropped items
   were deleted, and differ from the full-history log-probs (fp32).
3. Generation path: the KV-cache decoding mask reproduces the teacher-forced
   log-probs step by step.
"""
import argparse
import ast
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer, DataCollatorForSeq2Seq

from layers import LatentModel
from steered_reading import SteeredCollator, SteeredReadingDataset

LIST_COLUMNS = ["item_asins", "history_item_id", "history_item_title", "history_rating", "history_timestamp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--rows", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.add_special_tokens({"additional_special_tokens": tok.additional_special_tokens + ["<|Thought|>"]})
    frame = pd.read_csv(args.test_file, index_col=0)
    frame = frame[frame["history_item_id"].map(lambda x: len(ast.literal_eval(x))) > args.k].head(args.rows)
    full_path, cut_path = "steer_check_full.csv", "steer_check_cut.csv"
    cut = frame.copy()
    for col in LIST_COLUMNS:
        cut[col] = cut[col].map(lambda x: str(ast.literal_eval(x)[-args.k:]))
    frame.to_csv(full_path)
    cut.to_csv(cut_path)
    data = {}
    for name, path in (("full", full_path), ("cut", cut_path)):
        data[name] = SteeredReadingDataset(train_file=path, tokenizer=tok, max_len=2560, category="toys and games")
    os.remove(full_path), os.remove(cut_path)

    # 1. assignment
    for i in range(len(frame)):
        row = data["full"][i]
        titles = ast.literal_eval(frame.iloc[i]["history_item_title"])
        for j, t in enumerate(titles):
            text = tok.decode([x for x, idx in zip(row["input_ids"], row["item_index"]) if idx == j])
            if t not in text or len(text) > len(t) + 6:
                raise AssertionError(f"row {i} item {j}: tokens decode to {text!r}, title {t!r}")
    print(f"[1] item assignment ok on {len(frame)} rows")

    # 2. mask = deletion
    config = AutoConfig.from_pretrained(args.base_model)
    config.latent_continuous_thoughts = {"k": 1, "source": "hidden", "token_norm": 1.0}
    config.latent_steered_reading = {"k": args.k, "rule": "recent", "aux_weight": 0.0}
    model = LatentModel.from_pretrained(args.base_model, config=config, torch_dtype=torch.float32).cuda().eval()
    model.resize_token_embeddings(len(tok))
    fixed = torch.randn(1, 1, config.hidden_size, device="cuda") * float(model.model.embed_tokens.weight.norm(dim=-1).mean())
    model.thoughts.override = lambda x: fixed.expand(x.size(0), -1, -1).to(x)
    collator = SteeredCollator(DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8, return_tensors="pt", padding=True))

    def target_logprobs(rows, rule):
        model.steer.rule_override = rule
        batch = {k: v.cuda() for k, v in collator(rows).items()}
        with torch.no_grad():
            logits = model(**{k: batch[k] for k in ("input_ids", "attention_mask", "item_index", "item_sim")}).logits
        out = []
        for b in range(logits.size(0)):
            pos = torch.nonzero(batch["labels"][b] != -100).squeeze(1)
            out.append(torch.log_softmax(logits[b, pos - 1].float(), -1))
        return out

    full_rows = [data["full"][i] for i in range(len(frame))]
    cut_rows = [data["cut"][i] for i in range(len(frame))]
    masked = target_logprobs(full_rows, "recent")
    deleted = target_logprobs(cut_rows, "all")
    unmasked = target_logprobs(full_rows, "all")
    d_mask = max(float((a - b).abs().max()) for a, b in zip(masked, deleted))
    d_full = max(float((a - b).abs().max()) for a, b in zip(masked, unmasked))
    print(f"[2] masked vs deleted max |dlogp| {d_mask:.2e}; masked vs full history {d_full:.2e}")
    if d_mask > 1e-3 or d_full < 1e-2:
        raise AssertionError("mask is not equivalent to deleting the dropped items")

    # 3. KV-cache decoding with the mask reproduces teacher forcing
    model.steer.rule_override = "recent"
    row = full_rows[0]
    labels = torch.tensor(row["labels"])
    prompt_len = int((labels == -100).sum())
    ids = torch.tensor([row["input_ids"]], device="cuda")
    model.steer.item_index = torch.tensor([row["item_index"]])
    model.steer.item_sim = None
    with torch.no_grad():
        out = model(input_ids=ids[:, :prompt_len], attention_mask=torch.ones(1, prompt_len, dtype=torch.long, device="cuda"),
                    use_cache=True)
        cache, steps = out.past_key_values, [torch.log_softmax(out.logits[0, -1].float(), -1)]
        for t in range(prompt_len, ids.size(1) - 1):
            mask = torch.ones(1, t + 1, dtype=torch.long, device="cuda")
            out = model(input_ids=ids[:, t:t + 1], attention_mask=mask, past_key_values=cache, use_cache=True,
                        cache_position=torch.tensor([t], device="cuda"))
            cache = out.past_key_values
            steps.append(torch.log_softmax(out.logits[0, -1].float(), -1))
    model.steer.item_index = None
    d_gen = float((torch.stack(steps) - masked[0]).abs().max())
    print(f"[3] KV-cache decoding vs teacher forcing max |dlogp| {d_gen:.2e}")
    if d_gen > 1e-3:
        raise AssertionError("decoding mask differs from the training mask")
    print("steered reading checks passed")


if __name__ == "__main__":
    main()
