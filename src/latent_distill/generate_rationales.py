"""History-only rationales for latent distillation (document/latent_distillation.md).

For every row of a split CSV, an instruction-tuned LLM reads the user's purchase
history (titles only, never the target) and writes a short analysis of what the
user is doing and what they are likely to buy next.  The output is a JSON list of
strings aligned with the CSV rows, read by ``RationaleLatentRDataset``.

Usage:
    python src/latent_distill/generate_rationales.py --model Qwen/Qwen2.5-7B-Instruct \
        --csv data/Musical_Instruments/test/Musical_Instruments_5_2015-1-2018-11.csv \
        --out <dir>/test.json [--limit 64]
"""

import ast
import json
import os
import time

import fire
import pandas as pd

SYSTEM = ("You are an expert shopping assistant for musical instruments and accessories. "
          "You analyse a customer's purchase history to anticipate their next purchase.")

TASK = ("In 3 to 5 sentences: (1) infer what this customer is doing (instrument, skill level, setup or "
        "ongoing project), (2) explain what they are likely to need next and why, and (3) describe the "
        "kind of product they will most likely buy next: product type, key features, and brand if the "
        "history suggests one. Do not write a specific product title.")


def user_prompt(titles):
    lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
    return f"The customer bought these items, oldest first:\n{lines}\n\n{TASK}"


def clean(text):
    return " ".join(text.split())


def main(model: str, csv: str, out: str, limit: int = -1, max_tokens: int = 192,
         temperature: float = 0.0, seed: int = 0, gpu_memory_utilization: float = 0.85,
         max_model_len: int = 2048):
    from vllm import LLM, SamplingParams

    frame = pd.read_csv(csv, usecols=["history_item_title"])
    if limit > 0:
        frame = frame.iloc[:limit]
    histories = [list(ast.literal_eval(h)) for h in frame.history_item_title]

    llm = LLM(model=model, dtype="bfloat16", seed=seed, gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=max_model_len, enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()
    prompts = [tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
                                              {"role": "user", "content": user_prompt(h)}],
                                             tokenize=False, add_generation_prompt=True)
               for h in histories]
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed)
    start = time.time()
    outputs = llm.generate(prompts, params)
    rationales = [clean(o.outputs[0].text) for o in outputs]
    truncated = sum(o.outputs[0].finish_reason == "length" for o in outputs)
    print(f"{len(rationales)} rationales in {time.time() - start:.0f}s; truncated at {max_tokens} tokens: {truncated}")
    for h, r in list(zip(histories, rationales))[:3]:
        print("HISTORY:", h, "\nRATIONALE:", r, "\n")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as handle:
        json.dump({"model": model, "csv": os.path.abspath(csv), "max_tokens": max_tokens,
                   "temperature": temperature, "rationales": rationales}, handle)


if __name__ == "__main__":
    fire.Fire(main)
