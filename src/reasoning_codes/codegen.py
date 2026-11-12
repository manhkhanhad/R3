"""Gate 3: train the latent-free R3 prompt to GENERATE 4 reasoning codes, then the item (single path).

Response format (the prompt itself is unchanged):
    ### Response: \n<c1_a><c2_b><c3_c><c4_d> \n"<title>"\n<eos>
Codes = Cornac-faithful RQ-VAE codes of each training row's GEARec target part (rqvae_cornac.py). Loss on
the 4 codes + the title (the " \n" separator is fixed and not trained). Rows without reasoning are left out of
training/validation.

Evaluation (same trained model):
    gen     codes generated greedily, level l restricted to its 256 tokens, then 10-beam catalog-constrained item
    oracle  the row's TRUE codes forced (target-aware test reasoning: ceiling, not a method)
    shuffle another row's true codes forced (does the item step use the codes?)
Reports HR/NDCG @1..20 and per-level accuracy of generated codes vs the true codes.

    python src/reasoning_codes/codegen.py train --output_dir ... --codes_dir ...
    python src/reasoning_codes/codegen.py evaluate --model_dir ... --codes_dir ... --mode gen --result_json ...
"""

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "latent"))
sys.path.insert(0, os.path.dirname(HERE))

import fire
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

from experiment_tracking import init_evaluation_run, init_training_run, wandb_report_to
from plain_sft_dataset import PlainSFTDataset
from plain_sft_eval import allowed_catalog_tokens, build_catalog_trie
from rationale_sft_train import BestModelCallback
from utils.ranking_metrics import calculate_ranking_metrics

LEVELS, K = 4, 256
CODE_TOKENS = [f"<c{l + 1}_{j}>" for l in range(LEVELS) for j in range(K)]
SEP = " \n"
R3 = "/common/home/users/m/mk.ngohuu.2024/research/R3"
DATA = R3 + "/data/Toys_and_Games"
STEM = "Toys_and_Games_5_2017-1-2018-11"
CODES = R3 + "/output_dir/reasoning_codes/rqvae_cornac_L4_K256_d32"


def add_code_tokens(tokenizer, model=None, seed=0):
    """Add the 4 x 256 code tokens (contiguous ids); init new embeddings to mean + small noise."""
    n_added = tokenizer.add_tokens(CODE_TOKENS, special_tokens=True)
    ids = tokenizer.convert_tokens_to_ids(CODE_TOKENS)
    assert ids == list(range(ids[0], ids[0] + LEVELS * K)), "code token ids must be contiguous"
    if model is not None and n_added:
        old = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(len(tokenizer))
        emb = model.get_input_embeddings().weight
        with torch.no_grad():
            g = torch.Generator().manual_seed(seed)
            mean, std = emb[:old].mean(0), emb[:old].std(0)
            # mean + per-dimension std noise: new tokens as distinct (and of similar norm) as existing ones
            emb[old:] = (mean + std * torch.randn(emb.shape[0] - old, emb.shape[1], generator=g)).to(emb.dtype)
    return ids[0]


def code_ids(base, codes_row):
    return [base + l * K + int(c) for l, c in enumerate(codes_row)]


class CodeGenDataset(PlainSFTDataset):
    def __init__(self, *args, codes=None, code_base=None, sep_ids=None, drop_missing=True, **kwargs):
        self.codes, self.code_base, self.sep_ids = codes, code_base, sep_ids
        super().__init__(*args, **kwargs)
        if len(self.codes) != len(self.data):
            raise ValueError("codes are not aligned with the CSV rows")
        if drop_missing and not self.test:
            keep = [i for i in range(len(self.data)) if int(self.codes[i, 0]) >= 0]
            self.inputs = [self.inputs[i] for i in keep]
            self.data = self.data.iloc[keep].reset_index(drop=True)

    def pre(self, idx):
        instruction = (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request. \n\n"
            f"### Instruction:\n{self.instructs[0]}\n"
        )
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        history = self.get_history(self.data.iloc[idx])
        target_item = history["output"]
        history["output"] = ""
        tokens += self.tokenizer.encode(self.generate_prompt(history), bos=False, eos=False)
        if self.test:
            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}
        if int(self.codes[idx, 0]) < 0:   # dropped after init
            return {"input_ids": tokens, "attention_mask": [1] * len(tokens), "labels": [-100] * len(tokens)}
        cids = code_ids(self.code_base, self.codes[idx].tolist())
        title = self.tokenizer.encode(target_item, bos=False, eos=True)
        seq = tokens + cids + self.sep_ids + title
        labels = [-100] * len(tokens) + cids + [-100] * len(self.sep_ids) + title
        return {"input_ids": seq[-self.max_len:], "attention_mask": [1] * min(len(seq), self.max_len),
                "labels": labels[-self.max_len:]}


def load_codes(codes_dir, split):
    return torch.load(f"{codes_dir}/codes_{split}.pt")


def train(output_dir, codes_dir=CODES, base_model="Qwen/Qwen2.5-1.5B-Instruct", seed=0, batch_size=128,
          micro_batch_size=2, num_epochs=3, learning_rate=3e-4, cutoff_len=1536):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("one GPU expected")
    init_training_run(dataset="Toys_and_Games", config={"method": "reasoning_codegen", "codes_dir": codes_dir,
                                                        "output_dir": output_dir, "learning_rate": learning_rate})
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    base = add_code_tokens(tokenizer, model, seed)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    sep_ids = tokenizer.encode(SEP, add_special_tokens=False)
    common = dict(tokenizer=tokenizer, max_len=cutoff_len, sample=-1, seed=seed, category="toys and games",
                  code_base=base, sep_ids=sep_ids)
    train_data = CodeGenDataset(train_file=f"{DATA}/train/{STEM}.csv", codes=load_codes(codes_dir, "train"), **common)
    eval_data = CodeGenDataset(train_file=f"{DATA}/valid/{STEM}.csv", codes=load_codes(codes_dir, "valid"), **common)
    print(f"train rows {len(train_data)}, valid rows {len(eval_data)}; code token ids {base}..{base + LEVELS * K - 1}; "
          f"sep ids {sep_ids}", flush=True)
    os.makedirs(output_dir, exist_ok=True)
    done = os.path.join(output_dir, "TRAINING_DONE")
    if os.path.exists(done):
        os.remove(done)
    best_cb = BestModelCallback(model, tokenizer, output_dir, patience=1)
    trainer = transformers.Trainer(
        model=model, train_dataset=train_data, eval_dataset=eval_data, processing_class=tokenizer,
        args=transformers.TrainingArguments(
            output_dir=output_dir, per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size, gradient_accumulation_steps=batch_size // micro_batch_size,
            num_train_epochs=num_epochs, learning_rate=learning_rate, warmup_steps=20, bf16=True, logging_steps=1,
            optim="adamw_torch", eval_strategy="epoch", save_strategy="no", report_to=wandb_report_to(),
            remove_unused_columns=False, seed=seed),
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        callbacks=[best_cb])
    trainer.train()
    with open(done, "w") as handle:
        json.dump({"best_eval_loss": best_cb.best, "evals": best_cb.history}, handle, indent=1)


def init_model(output_dir, base_model="Qwen/Qwen2.5-1.5B-Instruct", seed=0):
    """Smoke test helper: save the UNTRAINED base model with the code tokens added."""
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    add_code_tokens(tokenizer, model, seed)
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir); tokenizer.save_pretrained(output_dir)


def derangement(n, seed):
    order = list(range(n)); random.Random(seed).shuffle(order)
    perm = list(range(n))
    for k in range(n):
        perm[order[k]] = order[(k + 1) % n]
    return perm


def evaluate(model_dir, result_json, mode="gen", codes_dir=CODES, batch_size=8, num_beams=10, limit=0, seed=0):
    assert mode in ("gen", "oracle", "shuffle")
    wandb_run = init_evaluation_run(dataset="Toys_and_Games", config={"method": f"reasoning_codegen_eval_{mode}",
                                                                      "model_dir": model_dir})
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = tokenizer.convert_tokens_to_ids(CODE_TOKENS[0])
    assert tokenizer.convert_tokens_to_ids(CODE_TOKENS[-1]) == base + LEVELS * K - 1
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to("cuda").eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    sep_ids = tokenizer.encode(SEP, add_special_tokens=False)
    true_codes = load_codes(codes_dir, "test")
    test_csv = f"{DATA}/test/{STEM}.csv"
    ds = CodeGenDataset(train_file=test_csv, tokenizer=tokenizer, max_len=4096, sample=-1, test=True,
                        category="toys and games", codes=true_codes, code_base=base, sep_ids=sep_ids)
    results = ds.get_all()
    n = min(limit, len(ds)) if limit else len(ds)
    results = results[:n]
    perm = derangement(len(ds), seed)
    info = f"{DATA}/info/{STEM}.txt"
    trie = build_catalog_trie(info, tokenizer)
    catalog = {l.split("\t", 1)[0].strip(" \n") for l in open(info)}
    level_ranges = [list(range(base + l * K, base + (l + 1) * K)) for l in range(LEVELS)]
    gen_codes = torch.full((n, LEVELS), -1, dtype=torch.long)

    for start in range(0, n, batch_size):
        rows = list(range(start, min(start + batch_size, n)))
        ex = [ds[i] for i in rows]
        width = max(len(e["input_ids"]) for e in ex)
        ids = torch.tensor([[tokenizer.pad_token_id] * (width - len(e["input_ids"])) + e["input_ids"] for e in ex], device="cuda")
        att = torch.tensor([[0] * (width - len(e["input_ids"])) + [1] * len(e["input_ids"]) for e in ex], device="cuda")
        # ---- phase 1: the 4 codes
        if mode == "gen":
            with torch.inference_mode():
                out = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=LEVELS, do_sample=False,
                                     num_beams=1, pad_token_id=tokenizer.pad_token_id,
                                     prefix_allowed_tokens_fn=lambda b, seq: level_ranges[min(len(seq) - width, LEVELS - 1)])
            c = out[:, width:width + LEVELS]
        else:
            src = rows if mode == "oracle" else [perm[i] for i in rows]
            fill = []
            for i, j in zip(rows, src):
                row = true_codes[j]
                if int(row[0]) < 0:          # no true codes for this row: fall back to generated codes
                    with torch.inference_mode():
                        r = rows.index(i)
                        o = model.generate(input_ids=ids[r:r + 1], attention_mask=att[r:r + 1], max_new_tokens=LEVELS,
                                           do_sample=False, num_beams=1, pad_token_id=tokenizer.pad_token_id,
                                           prefix_allowed_tokens_fn=lambda b, seq: level_ranges[min(len(seq) - width, LEVELS - 1)])
                    fill.append(o[0, width:width + LEVELS].tolist())
                else:
                    fill.append(code_ids(base, row.tolist()))
            c = torch.tensor(fill, device="cuda")
        gen_codes[rows] = (c - base - torch.arange(LEVELS, device="cuda") * K).cpu()
        # ---- phase 2: separator + catalog-constrained beam over the item
        sep = torch.tensor(sep_ids, device="cuda").expand(len(rows), -1)
        ids2 = torch.cat([ids, c, sep], 1)
        att2 = torch.cat([att, torch.ones(len(rows), LEVELS + len(sep_ids), dtype=att.dtype, device="cuda")], 1)
        w2 = ids2.shape[1]
        with torch.inference_mode():
            gen = model.generate(input_ids=ids2, attention_mask=att2, num_beams=num_beams, num_return_sequences=num_beams,
                                 do_sample=False, max_new_tokens=128, length_penalty=1.0,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                 prefix_allowed_tokens_fn=lambda b, seq: allowed_catalog_tokens(trie, seq[w2:].tolist(), tokenizer.eos_token_id),
                                 return_dict_in_generate=True, output_scores=True)
        preds = tokenizer.batch_decode(gen.sequences[:, w2:], skip_special_tokens=True)
        scores = gen.sequences_scores.tolist()
        for o, i in enumerate(rows):
            lo, hi = o * num_beams, (o + 1) * num_beams
            valid = [(p.strip(' "\n'), s) for p, s in zip(preds[lo:hi], scores[lo:hi]) if p.strip(' "\n') in catalog]
            if not valid:
                raise RuntimeError(f"No catalog predictions for test row {i}")
            results[i]["predict"] = [t for t, _ in valid]
            results[i]["predict_score"] = [s for _, s in valid]
            results[i]["codes"] = gen_codes[i].tolist()
            results[i].pop("dedup", None)
        if start % (batch_size * 50) == 0:
            print(f"{start + len(rows)}/{n}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(result_json)), exist_ok=True)
    json.dump(results, open(result_json, "w"))
    metrics = calculate_ranking_metrics(results, info)
    has = true_codes[:n, 0] >= 0
    acc = {}
    for l in range(LEVELS):
        acc[f"code_acc_l{l + 1}"] = float((gen_codes[has, l] == true_codes[:n][has, l]).float().mean())
        acc[f"code_prefix_acc_1..{l + 1}"] = float((gen_codes[has, :l + 1] == true_codes[:n][has, :l + 1]).all(1).float().mean())
    print(json.dumps({"mode": mode, "rows": n, **metrics, **acc}))
    if wandb_run is not None:
        wandb_run.log({f"evaluation/{mode}/{k}": v for k, v in {**metrics, **acc}.items()}); wandb_run.finish()


if __name__ == "__main__":
    fire.Fire({"train": train, "evaluate": evaluate, "init_model": init_model})
