"""Catalog-constrained 10-beam ranking for rationale_sft_train.py models.

The prompt ends in "### Response: \n" (mode off) or in the rationale block, which also ends
in " \n"; plain_sft_eval's catalog trie constrains only the generated title, so it is reused.
Eval-time modes probe whether the model reads the rationale: own / shuffle (another row's) /
empty ("none") / off (no block).
"""

import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment_tracking import init_evaluation_run
from plain_sft_eval import allowed_catalog_tokens, build_catalog_trie
from rationale_sft_dataset import RationaleSFTDataset
from rationale_sft_train import CATEGORY_NAMES
import rationale_text
from utils.ranking_metrics import calculate_ranking_metrics


def evaluate(
    base_model: str,
    category: str = "Toys_and_Games",
    info_file: str = "",
    test_data_path: str = "",
    rationale_file: str = "",
    mode: str = "own",
    result_json_data: str = "",
    batch_size: int = 8,
    num_beams: int = 10,
    max_new_tokens: int = 128,
    limit: int = 0,
    shuffle_seed: int = 0,
    hint_file: str = "",
    hint_mode: str = "none",
):
    if not (info_file and test_data_path and result_json_data):
        raise ValueError("info_file, test_data_path and result_json_data are required")
    if not torch.cuda.is_available():
        raise RuntimeError("Ranking evaluation requires a CUDA GPU")
    rationales = rationale_text.load(rationale_file, test_data_path) if mode != "off" else None
    hints = rationale_text.load_categories(hint_file, test_data_path) if hint_mode != "none" else None
    wandb_run = init_evaluation_run(
        dataset=category,
        config={"method": f"rationale_sft_eval_{mode}", "base_model": base_model,
                "test_data_path": test_data_path, "rationale_file": rationale_file,
                "mode": mode, "hint_mode": hint_mode, "limit": limit, "dataset": category},
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    model = model.to("cuda").eval()

    dataset = RationaleSFTDataset(
        train_file=test_data_path, tokenizer=tokenizer, max_len=4096, sample=-1, test=True,
        category=CATEGORY_NAMES[category], rationales=rationales, mode=mode, shuffle_seed=shuffle_seed,
        hints=hints, hint_mode=hint_mode,
    )
    results = dataset.get_all()
    n = min(limit, len(dataset)) if limit else len(dataset)
    results = results[:n]
    trie = build_catalog_trie(info_file, tokenizer)
    with open(info_file) as handle:
        catalog_titles = {line.split("\t", 1)[0].strip(" \n") for line in handle}

    for start in range(0, n, batch_size):
        examples = [dataset[i] for i in range(start, min(start + batch_size, n))]
        width = max(len(e["input_ids"]) for e in examples)
        input_ids = torch.tensor([[tokenizer.pad_token_id] * (width - len(e["input_ids"])) + e["input_ids"]
                                  for e in examples], device="cuda")
        attention_mask = torch.tensor([[0] * (width - len(e["attention_mask"])) + e["attention_mask"]
                                       for e in examples], device="cuda")

        def prefix_allowed_tokens_fn(batch_id, sequence):
            return allowed_catalog_tokens(trie, sequence[width:].tolist(), tokenizer.eos_token_id)

        with torch.inference_mode():
            generation = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, num_beams=num_beams,
                num_return_sequences=num_beams, do_sample=False, max_new_tokens=max_new_tokens,
                length_penalty=1.0, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, return_dict_in_generate=True,
                output_scores=True, use_cache=True,
            )
        predictions = tokenizer.batch_decode(generation.sequences[:, width:], skip_special_tokens=True)
        scores = generation.sequences_scores.tolist()
        for offset, row in enumerate(results[start:start + len(examples)]):
            lo, hi = offset * num_beams, (offset + 1) * num_beams
            valid = [(p.strip(' "\n'), s) for p, s in zip(predictions[lo:hi], scores[lo:hi])
                     if p.strip(' "\n') in catalog_titles]
            if not valid:
                raise RuntimeError(f"No catalog predictions for test row {start + offset}")
            row["predict"] = [t for t, _ in valid]
            row["predict_score"] = [s for _, s in valid]
            row.pop("dedup", None)
        if start % (batch_size * 50) == 0:
            print(f"{start + len(examples)}/{n}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(result_json_data)), exist_ok=True)
    with open(result_json_data, "w") as handle:
        json.dump(results, handle, indent=2)
    metrics = calculate_ranking_metrics(results, info_file)
    print(json.dumps({"mode": mode, "hint": hint_mode, "rows": n, **metrics}))
    if wandb_run is not None:
        wandb_run.log({f"evaluation/{mode}_hint-{hint_mode}/{k}": v for k, v in metrics.items()})
        wandb_run.finish()
    return metrics


if __name__ == "__main__":
    fire.Fire(evaluate)
