"""CPU repro: what does HF 4.48 beam search pass to logits processors step to step?"""
import sys
from pathlib import Path
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, GenerationConfig, LogitsProcessor, LogitsProcessorList

torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
model = Qwen2ForCausalLM(cfg).eval()

B, L, NB = 2, 7, 4
prompt = torch.randint(5, 60, (B, L))
mask = torch.ones(B, L, dtype=torch.long)


class Spy(LogitsProcessor):
    def __init__(self):
        self.seen = []

    def __call__(self, input_ids, scores):
        self.seen.append(input_ids.clone())
        if len(self.seen) > 1:
            prev, cur = self.seen[-2], self.seen[-1]
            print(f"step {len(self.seen)}: prev {tuple(prev.shape)} cur {tuple(cur.shape)}")
            if cur.size(1) != prev.size(1) + 1:
                print("  !! length jump", prev.size(1), "->", cur.size(1))
            else:
                n = cur.size(0)
                for g in range(n // NB):
                    sl = slice(g * NB, (g + 1) * NB)
                    eq = (cur[sl, :-1].unsqueeze(1) == prev[sl].unsqueeze(0)).all(-1)
                    if not bool(eq.any(1).all()):
                        print(f"  !! group {g}: some current prefix not among previous rows")
                        # check against the WHOLE previous batch
                        eq_all = (cur[sl, :-1].unsqueeze(1) == prev.unsqueeze(0)).all(-1)
                        print("     matches anywhere in batch:", eq_all.any(1).tolist())
        else:
            print(f"step 1: {tuple(input_ids.shape)} (N={input_ids.size(0)}, prompt L={L})")
        return scores


spy = Spy()
gen = GenerationConfig(num_beams=NB, num_return_sequences=NB, max_new_tokens=5,
                       pad_token_id=0, eos_token_id=1)
out = model.generate(prompt, attention_mask=mask, generation_config=gen,
                     logits_processor=LogitsProcessorList([spy]),
                     return_dict_in_generate=True, output_scores=True)
print("done; final sequences", tuple(out.sequences.shape))
