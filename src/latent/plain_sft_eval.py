"""Catalog-constrained ranking evaluation for the latent-free SFT baseline."""

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
from plain_sft_dataset import PlainSFTDataset
from utils.ranking_metrics import calculate_ranking_metrics


def build_catalog_trie(info_file, tokenizer):
    """Tokenize catalog titles in the same response context as SFT targets."""
    response_prefix = "### Response: \n"
    prefix_ids = tokenizer.encode(response_prefix, add_special_tokens=False)
    trie = {}
    with open(info_file) as handle:
        for line in handle:
            title = line.split("\t", 1)[0].strip(" \n")
            if not title:
                continue
            full_ids = tokenizer.encode(
                f'{response_prefix}"{title}"\n', add_special_tokens=False,
            )
            if full_ids[:len(prefix_ids)] != prefix_ids:
                raise ValueError(f"Catalog title changes response prefix tokens: {title!r}")
            node = trie
            for token_id in full_ids[len(prefix_ids):] + [tokenizer.eos_token_id]:
                node = node.setdefault(token_id, {})
    if not trie:
        raise ValueError(f"No catalog titles found in {info_file}")
    return trie


def allowed_catalog_tokens(trie, generated_ids, eos_token_id):
    node = trie
    for token_id in generated_ids:
        if token_id not in node:
            if not node and token_id == eos_token_id:
                continue
            # Beam search may retain a -inf filler beam when a trie branch has
            # fewer continuations than the beam width. End that beam; only
            # complete catalog titles are retained in the final ranking.
            return [eos_token_id]
        node = node[token_id]
    return list(node) or [eos_token_id]


def evaluate(
    base_model: str,
    info_file: str = "data/Musical_Instruments/info/Musical_Instruments_5_2015-1-2018-11.txt",
    test_data_path: str = "data/Musical_Instruments/test/Musical_Instruments_5_2015-1-2018-11.csv",
    result_json_data: str = "/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir/Musical_Instruments_plain_sft/final_result.json",
    batch_size: int = 4,
    sample: int = -1,
    num_beams: int = 10,
    max_new_tokens: int = 128,
    category: str = "music instruments",
    dataset_name: str = "Musical_Instruments",
    instruction: str = None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("Ranking evaluation requires a CUDA GPU")
    wandb_run = init_evaluation_run(
        dataset=dataset_name,
        config={
            "method": "plain_sft",
            "base_model": base_model,
            "test_data_path": test_data_path,
            "dataset": dataset_name, "instruction": instruction,
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    model = model.to("cuda").eval()

    dataset = PlainSFTDataset(
        train_file=test_data_path, tokenizer=tokenizer, max_len=2560,
        sample=sample, test=True, category=category, instruction=instruction,
    )
    results = dataset.get_all()
    trie = build_catalog_trie(info_file, tokenizer)
    with open(info_file) as handle:
        catalog_titles = {
            line.split("\t", 1)[0].strip(" \n") for line in handle
        }

    for start in range(0, len(dataset), batch_size):
        examples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        width = max(len(example["input_ids"]) for example in examples)
        input_ids = torch.tensor([
            [tokenizer.pad_token_id] * (width - len(example["input_ids"])) + example["input_ids"]
            for example in examples
        ], device="cuda")
        attention_mask = torch.tensor([
            [0] * (width - len(example["attention_mask"])) + example["attention_mask"]
            for example in examples
        ], device="cuda")

        def prefix_allowed_tokens_fn(batch_id, sequence):
            return allowed_catalog_tokens(
                trie, sequence[width:].tolist(), tokenizer.eos_token_id,
            )

        with torch.inference_mode():
            generation = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                length_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )
        predictions = tokenizer.batch_decode(
            generation.sequences[:, width:], skip_special_tokens=True,
        )
        scores = generation.sequences_scores.tolist()
        for offset, row in enumerate(results[start:start + len(examples)]):
            lo, hi = offset * num_beams, (offset + 1) * num_beams
            valid = [
                (value.strip(' "\n'), score)
                for value, score in zip(predictions[lo:hi], scores[lo:hi])
                if value.strip(' "\n') in catalog_titles
            ]
            if not valid:
                raise RuntimeError(f"No catalog predictions for test row {start + offset}")
            row["predict"] = [title for title, _ in valid]
            row["predict_score"] = [score for _, score in valid]
            row.pop("dedup", None)

    os.makedirs(os.path.dirname(os.path.abspath(result_json_data)), exist_ok=True)
    with open(result_json_data, "w") as handle:
        json.dump(results, handle, indent=2)
    metrics = calculate_ranking_metrics(results, info_file)
    print(metrics)
    if wandb_run is not None:
        wandb_run.log({f"evaluation/{key}": value for key, value in metrics.items()})
        wandb_run.finish()
    return metrics


if __name__ == "__main__":
    fire.Fire(evaluate)
