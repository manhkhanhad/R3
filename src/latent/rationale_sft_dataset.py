"""Latent-free R3 prompt with a text rationale where R3 puts its <|Thought|> latent.

    ... ### User Input: \n<history>\n\n### Response: \n<rational> {rationale} <end_rational> \n"<target title>"\n<eos>

The rationale block is input only (labels -100); the model is trained on the title alone.
The block ends in " \n", the same characters that precede a title in the plain prompt, so the
catalog trie of plain_sft_eval.py tokenizes titles identically.

Modes (the CSV row order defines "own"):
    off      no block at all (latent-free control, identical to PlainSFTDataset)
    own      the row's own rationale
    shuffle  another row's rationale (fixed derangement, seeded)
    empty    the block with "none"
"""

import random

from plain_sft_dataset import PlainSFTDataset
from rationale_text import NONE_TEXT

MODES = ("off", "own", "shuffle", "empty")
OPEN, CLOSE = "<rational>", "<end_rational>"


HINT_MODES = ("none", "gold", "shuffle")


def rationale_block(text, hint=None):
    """hint: ORACLE ablation only (the target's category), appended inside the block."""
    hint_text = f" Target category: {hint}." if hint else ""
    return f"{OPEN} {text or NONE_TEXT}{hint_text} {CLOSE} \n"


def derangement(n, seed):
    """A permutation with no fixed points (row i never keeps its own rationale)."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    perm = list(range(n))
    for k in range(n):
        # One cycle through the shuffled order: every row points to the next one.
        perm[order[k]] = order[(k + 1) % n]
    return perm


class RationaleSFTDataset(PlainSFTDataset):
    def __init__(self, *args, rationales=None, mode="own", shuffle_seed=0, hints=None, hint_mode="none", **kwargs):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if hint_mode not in HINT_MODES:
            raise ValueError(f"hint_mode must be one of {HINT_MODES}")
        if hint_mode != "none" and (hints is None or mode == "off"):
            raise ValueError("hint_mode needs hints and a rationale block (mode != off)")
        self.hints = hints
        self.hint_mode = hint_mode
        self._hint_perm = None
        if mode != "off" and rationales is None:
            raise ValueError(f"mode {mode!r} needs rationales")
        if kwargs.get("sample", -1) > 0 and mode != "off":
            # Sampling reorders rows; rationales are joined by CSV position, so keep it simple.
            raise ValueError("rationale modes require sample=-1 (full CSV in order)")
        self.rationales = rationales
        self.mode = mode
        self.shuffle_seed = shuffle_seed
        self._perm = None
        super().__init__(*args, **kwargs)

    def rationale_for(self, idx):
        if self.mode == "empty":
            return ""
        if self.mode == "shuffle":
            if self._perm is None:
                self._perm = derangement(len(self.rationales), self.shuffle_seed)
            return self.rationales[self._perm[idx]]
        return self.rationales[idx]

    def hint_for(self, idx):
        if self.hint_mode == "none":
            return None
        if self.hint_mode == "shuffle":
            if self._hint_perm is None:
                # independent of the rationale derangement
                self._hint_perm = derangement(len(self.hints), self.shuffle_seed + 1)
            return self.hints[self._hint_perm[idx]]
        return self.hints[idx]

    def pre(self, idx):
        if self.mode == "off":
            return super().pre(idx)
        if len(self.rationales) != len(self.data):
            raise ValueError("rationales are not aligned with the CSV rows")
        if self.hints is not None and len(self.hints) != len(self.data):
            raise ValueError("hints are not aligned with the CSV rows")
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
        tokens += self.tokenizer.encode(rationale_block(self.rationale_for(idx), self.hint_for(idx)),
                                        bos=False, eos=False)

        if self.test:
            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

        target_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        labels = [-100] * len(tokens) + target_tokens
        tokens += target_tokens
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": [1] * min(len(tokens), self.max_len),
            "labels": labels[-self.max_len:],
        }
