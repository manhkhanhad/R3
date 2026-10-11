"""R3 whose thought vector is pooled from history + a rationale the decoder never sees.

Latent distillation, stage 1 (document/latent_distillation.md).  An LLM writes a
history-only rationale per row (``src/latent_distill/generate_rationales.py``).
The teacher R3 pools ``r`` from a *pool sequence* (the R3 prompt with the
rationale appended to the input) and splices it into the ordinary R3 prompt the
decoder reads:

    pool sequence:    instruction + history + "### Analysis:" + rationale + <|Thought|>
    decoder sequence: instruction + history + <|Thought|> + target

so whatever the rationale adds can only reach the prediction through ``r``.  This
makes ``r`` non-redundant by construction (like the oracle side information of
decoder_insensitivity_to_latent.md), but with information that is available at
test time.  The teacher's ``r`` is the target latent for stage 2.

Dataset modes: ``latent`` (above), ``text`` (rationale written into the decoder
prompt; plain ``LatentModel``, a reference for how useful the rationale is at
all) and ``none`` (plain R3 prompt).
"""

import torch
from transformers import DataCollatorForSeq2Seq

from layers import LatentModel
from reasoning_dataset import LatentRDataset

THOUGHT = "<|Thought|>"
MODES = ("latent", "text", "none")


class RationaleLatentModel(LatentModel):
    def __init__(self, config):
        super().__init__(config)
        self._pool = None
        self.last_latent = None

    def set_pool(self, pool_input_ids, pool_attention_mask):
        """Pool ``r`` from these sequences in the next prompt pass (one row per decoder row,
        before any beam expansion)."""
        self._pool = (pool_input_ids, pool_attention_mask)

    def pool_latent(self, pool_input_ids, pool_attention_mask):
        thought = self.model.embed_tokens.num_embeddings - 1
        where = torch.nonzero(pool_input_ids == thought)
        if where.shape[0] != pool_input_ids.shape[0]:
            raise ValueError("every pool sequence needs exactly one thought token")
        hidden = self.model(input_ids=pool_input_ids, attention_mask=pool_attention_mask).last_hidden_state
        latent = self.attention(hidden_states=hidden, attention_mask=pool_attention_mask,
                                thought_id_idx=where[:, 1])
        latent = self.observe_reasoning_latent(latent)
        return self.project_reasoning_latent(latent)

    def generate_embs(self, input_ids, attention_mask):
        if self._pool is None:
            return super().generate_embs(input_ids, attention_mask)
        pool_ids, pool_mask = self._pool
        self._pool = None
        latent = self.pool_latent(pool_ids, pool_mask)
        if input_ids.shape[0] != latent.shape[0]:  # beam search repeats each row num_beams times
            latent = latent.repeat_interleave(input_ids.shape[0] // latent.shape[0], dim=0)
        thought = self.model.embed_tokens.num_embeddings - 1
        where = torch.nonzero(input_ids == thought)
        if where.shape[0] != input_ids.shape[0]:
            raise ValueError("every decoder sequence needs exactly one thought token")
        input_embs = self.model.embed_tokens(input_ids)
        input_embs[where[:, 0], where[:, 1]] = latent.to(input_embs.dtype)
        self.last_latent = latent.detach()
        return input_embs

    def forward(self, pool_input_ids=None, pool_attention_mask=None, **kwargs):
        if pool_input_ids is not None:
            self.set_pool(pool_input_ids, pool_attention_mask)
        try:
            return super().forward(**kwargs)
        finally:
            self._pool = None


class RationaleLatentRDataset(LatentRDataset):
    """``rationales``: list of strings aligned with the rows of the *full* split CSV."""

    def __init__(self, *args, rationales=None, mode="latent", **kwargs):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode != "none" and rationales is None:
            raise ValueError("rationales are required")
        self.rationales = rationales
        self.mode = mode
        super().__init__(*args, **kwargs)

    def row_id(self, idx):
        """Row of the full split: shards from utils/split.py keep it in ``Unnamed: 0``."""
        if "Unnamed: 0" in self.data.columns:
            return int(self.data.iloc[idx]["Unnamed: 0"])
        return int(self.data.index[idx])

    def prompt_tokens(self, idx, analysis=None):
        """The R3 prompt up to and including the thought token (``LatentRDataset.pre``),
        with ``analysis`` appended to the input when given."""
        instruction = ("Below is an instruction that describes a task, paired with an input that provides further "
                       "context. Write a response that appropriately completes the request. \n\n"
                       f"### Instruction:\n{self.instructs[0]}\n")
        history = self.get_history(self.data.iloc[idx])
        target = history["output"]
        text = history["input"]
        if analysis is not None:
            text = f"{text}\n\n### Analysis: \n{analysis}"
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        tokens = tokens + self.tokenizer.encode(self.generate_prompt({"input": text, "output": ""}), bos=False, eos=False)
        tokens = tokens + self.tokenizer.encode(THOUGHT, bos=False, eos=False)
        return tokens, target

    def pre(self, idx):
        if self.mode == "none":
            return super().pre(idx)
        rationale = self.rationales[self.row_id(idx)]
        if self.mode == "latent":
            example = super().pre(idx)
            pool, _ = self.prompt_tokens(idx, rationale)
            example["pool_input_ids"] = pool[-self.max_len:]
            example["pool_attention_mask"] = [1] * len(example["pool_input_ids"])
            return example
        tokens, target = self.prompt_tokens(idx, rationale)
        if self.test:
            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}
        prompt_len = len(tokens)
        tokens = tokens + self.tokenizer.encode(target, bos=False, eos=True)
        labels = [-100] * prompt_len + tokens[prompt_len:]
        return {"input_ids": tokens[-self.max_len:], "attention_mask": [1] * len(tokens[-self.max_len:]),
                "labels": labels[-self.max_len:]}


class RationaleCollator:
    """``DataCollatorForSeq2Seq`` for the decoder sequence; the pool sequence is padded on the
    same side, separately."""

    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.base = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=pad_to_multiple_of,
                                           return_tensors="pt", padding=True)
        self.pad_id = tokenizer.pad_token_id
        self.multiple = pad_to_multiple_of
        self.side = tokenizer.padding_side

    def pad(self, rows, value):
        length = max(len(r) for r in rows)
        length = -(-length // self.multiple) * self.multiple
        if self.side == "left":
            return torch.tensor([[value] * (length - len(r)) + list(r) for r in rows])
        return torch.tensor([list(r) + [value] * (length - len(r)) for r in rows])

    def __call__(self, features):
        pool = [f.pop("pool_input_ids") for f in features] if "pool_input_ids" in features[0] else None
        for f in features:
            f.pop("pool_attention_mask", None)
        batch = self.base(features)
        if pool is not None:
            batch["pool_input_ids"] = self.pad(pool, self.pad_id)
            batch["pool_attention_mask"] = self.pad([[1] * len(p) for p in pool], 0)
        return batch


def load_rationales(rationale_dir, csv_path):
    """``<rationale_dir>/<split>.json``, the split being the CSV's parent directory."""
    import json
    import os
    split = os.path.basename(os.path.dirname(os.path.abspath(csv_path)))
    with open(os.path.join(rationale_dir, f"{split}.json")) as handle:
        return json.load(handle)["rationales"]
