"""R3 dataset for next-item *category* prediction.

Same history prompt and <|Thought|> latent as the item task, but the response
is the category path of the target item (e.g. "Musical Instruments > ... > Picks")
instead of its title. Rows whose target has no valid category are dropped.
"""

import torch

from reasoning_dataset import LatentRDataset


def load_category_paths(category_file):
    """item_id -> category path, for items with a valid category."""
    data = torch.load(category_file, map_location="cpu", weights_only=True)
    return {i: p for i, p in enumerate(data["paths"]) if data["valid"][i]}


def write_category_info_file(category_paths, out_path):
    """Constrained-decoding catalog of the distinct category paths ("path\\tindex" per line)."""
    unique = sorted(set(category_paths.values()))
    with open(out_path, "w") as f:
        for i, path in enumerate(unique):
            f.write(f"{path}\t{i}\n")
    return unique


class CategoryRDataset(LatentRDataset):
    def __init__(self, *args, category_file=None, filter_invalid=True, **kwargs):
        """filter_invalid=False keeps every CSV row (prompt-only use, e.g. latent extraction)."""
        self.category_paths = load_category_paths(category_file)
        self.filter_invalid = filter_invalid
        super().__init__(*args, **kwargs)

    def get_inputs(self):
        keep = self.data["item_id"].map(lambda i: int(i) in self.category_paths)
        self.dropped_rows = int((~keep).sum()) if self.filter_invalid else 0
        if self.filter_invalid:
            self.data = self.data[keep].reset_index(drop=True)
        super().get_inputs()

    def get_history(self, row):
        out = super().get_history(row)
        out["output"] = "\"" + self.category_paths.get(int(row["item_id"]), "unknown") + "\"\n"
        return out

    def pre(self, idx):
        # identical to LatentRDataset.pre except for the instruction text
        category = self.category
        instruction = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Given a list of {category} the user recetenly enjoy, please predict the category of the next {category} that the user may bought
"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        history = self.get_history(self.data.iloc[idx])
        target = history["output"]
        history["output"] = ""
        tokens = tokens + self.tokenizer.encode(self.generate_prompt(history), bos=False, eos=False)
        tokens = tokens + self.tokenizer.encode("<|Thought|>", bos=False, eos=False)
        if self.test:
            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}
        prompt_len = len(tokens)
        tokens = tokens + self.tokenizer.encode(target, bos=False, eos=True)
        labels = [-100] * prompt_len + tokens[prompt_len:]
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": [1] * len(tokens[-self.max_len:]),
            "labels": labels[-self.max_len:],
        }
