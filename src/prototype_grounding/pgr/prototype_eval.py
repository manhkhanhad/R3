"""Prototype-aware variant of src/latent/latent_attention_eval.py.

Only two things differ from the original: (1) loads PrototypeGuidedLatentModel and
attaches the prototype bank, (2) that's it. The generate() call, output post-processing
(the [:, minLen-5:] slicing, "Response:" splitting), and constrained-decoding wiring
are all completely unchanged from the original -- deliberately, since
PrototypeGuidedLatentModel.prepare_inputs_for_generation makes prototype splicing
transparent to generate()'s own input_ids/logits-processor bookkeeping (see model.py).
There is no inputs_embeds-based generation here; generate() is called exactly as
latent_attention_eval.py always has, with real input_ids.

Run the same way as latent_attention_eval.py, with one extra required arg:
    python3 prototype_eval.py --base_model <stage3_checkpoint> --bank_path <bank.jsonl> \\
        --info_file ... --category ... --test_data_path ... --result_json_data ...
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "latent"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fire
import torch
from transformers import GenerationConfig, AutoTokenizer
from reasoning_dataset import LatentRDataset
from transformers import LogitsProcessorList, TemperatureLogitsWarper
from LogitProcesser import CFEnhancedLogitsProcessor, get_hash, get_prefix_data

from bank import load_prototype_bank
from model import PrototypeGuidedLatentModel

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

DEFAULT_BANK_PATH = Path(__file__).resolve().parent / "data" / (
    "CDs_and_Vinyl_5_2015-10-2018-11_train_evidence_bank.augmented.jsonl"
)


def main(
    base_model: str = "path/to/your/model",
    bank_path: str = str(DEFAULT_BANK_PATH),
    info_file: str = "data/info/Toys_and_Games_5_2017-1-2018-11.txt",
    category: str = "Toys_and_Games",
    test_data_path: str = "data/test/Toys_and_Games_5_2017-1-2018-11.csv",
    result_json_data: str = "temp.json",
    batch_size: int = 8,
    sample: int = 200,
    end_k: int = -1,
    seed: int = 0,
    temperature: float = 1.0,
    guidance_scale: float = 1.0,
    length_penalty: float = 1.0,
):
    category_dict = {"Office_Products": "office products", "Books": "books", "steam": "games", "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "STEAM": "games", "Movies": "movie",
                     "yelp": "resturant"}
    category = category_dict[category]

    bank = load_prototype_bank(bank_path)
    n_verified = sum(1 for t in bank.tiers if t == "verified")
    print(f"loaded bank: {len(bank)} prototypes ({n_verified} verified), dim={bank.dim}")

    model = PrototypeGuidedLatentModel.from_pretrained(base_model, torch_dtype=torch.bfloat16, use_flash_attention_2=False)
    model.attention.end_k = end_k
    model.set_prototype_bank(bank)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    hash_dict = get_prefix_data(info_file, tokenizer)

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    val_dataset = LatentRDataset(train_file=test_data_path, tokenizer=tokenizer, max_len=2560, category=category, test=True, K=4, seed=seed,
                                  sample=sample)

    encodings = [val_dataset.__getitem__(i) for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()

    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    model.eval()

    def evaluate(
            encodings,
            cf_logits,
            temperature=1.0,
            num_beams=10,
            max_new_tokens=128,
            guidance_scale=1.0,
            length_penalty=1.0,
            **kwargs,
    ):
        maxLen = max([len(_["input_ids"]) for _ in encodings])
        minLen = min([len(_["input_ids"]) for _ in encodings])

        padding_encodings = {"input_ids": [], "attention_mask": []}

        for _ in encodings:
            L = len(_["input_ids"])
            padding_encodings["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])
            padding_encodings["attention_mask"].append([0] * (maxLen - L) + _["attention_mask"])

        generation_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_beams,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        with torch.no_grad():
            ccc = CFEnhancedLogitsProcessor(
                guidance_scale=guidance_scale,
                cf_logits=cf_logits,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                cf_dict=None,
                unconditional_ids=None,
                model=model,
                tokenizer=tokenizer,
                num_beams=num_beams
            )
            logits_processor = LogitsProcessorList([TemperatureLogitsWarper(temperature=temperature), ccc])

            generation_output = model.generate(
                torch.tensor([_ for _ in padding_encodings["input_ids"]]).to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
                attention_mask=torch.tensor([_ for _ in padding_encodings["attention_mask"]]).to(device),
            )

        s = generation_output.sequences[:, minLen - 5:]
        scores = generation_output.sequences_scores.tolist()
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split("Response:")[-1] for _ in output]
        real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        real_scores = [scores[i * num_beams: (i + 1) * num_beams] for i in range(len(scores) // num_beams)]
        return real_outputs, real_scores, None

    model = model.to(device)

    from tqdm import tqdm
    outputs = []
    new_encodings = []
    BLOCK = (len(encodings) + batch_size - 1) // batch_size
    for i in range(BLOCK):
        new_encodings.append(encodings[i * batch_size: (i + 1) * batch_size])
    scores = []
    seq_scores = []
    for idx, encodings in enumerate(tqdm(new_encodings)):
        output, score, seq_score = evaluate(encodings, cf_logits=None, temperature=temperature, guidance_scale=guidance_scale, length_penalty=length_penalty)
        if idx == 0:
            print(output)
            print(score)
        outputs = outputs + output
        scores = scores + score
        seq_scores.append(seq_score)

    for i, test in enumerate(test_data):
        test["predict"] = outputs[i]
        test["predict_score"] = scores[i]

    for i in range(len(test_data)):
        if 'dedup' in test_data[i]:
            test_data[i].pop('dedup')

    import json
    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)


if __name__ == '__main__':
    fire.Fire(main)
