"""Recommendation SFT examples using the existing prompt without latent tokens."""

from reasoning_dataset import LatentRDataset


class PlainSFTDataset(LatentRDataset):
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

        target_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        labels = [-100] * len(tokens) + target_tokens
        tokens += target_tokens
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": [1] * min(len(tokens), self.max_len),
            "labels": labels[-self.max_len:],
        }
