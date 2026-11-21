from transformers.generation import LogitsProcessor
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np
import torch

from transformers.utils import add_start_docstrings


def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)


def get_prefix_data(info_file, tokenizer, n_thoughts=1):
    with open(info_file, 'r') as f:
        info = f.readlines()
        info = ["\"" + _.split('\t')[0].strip(' ') + "\"\n" for _ in info]
        # 这个地方真是被坑惨了，千万别把下面这两行对齐，就这个写法，不然会多出空格
        info = [f'''### Response: 
{_}''' for _ in info]
    prefixID = [tokenizer(_).input_ids for _ in info]
    thought_id = len(tokenizer) - 1
    for i in range(len(prefixID)):
        prefixID[i][4:4] = [thought_id] * n_thoughts
    start = 4 + n_thoughts  # "### Response: \n" (4 tokens) + thought tokens
    hash_dict = dict()
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(start, len(ID)):
            if i == start:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[start:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])

    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])
    return hash_dict


def get_title_edges(info_file, tokenizer, n_thoughts=1):
    """Trie edges of every title, for per-row exclusion of titles (e.g. the user's history).

    Same tokenisation and hashing as get_prefix_data.  Returns (title -> list of (hash_key, token) edges,
    edge -> number of titles using it).  Removing the edges whose every title is excluded prunes exactly the
    excluded titles: no other title loses a path and no node is left without children."""
    from collections import Counter
    with open(info_file, 'r') as f:
        titles = [line.split('\t')[0].strip(' ') for line in f.readlines()]
    edges, count = {}, Counter()
    thought_id = len(tokenizer) - 1
    start = 4 + n_thoughts
    for title in titles:
        # byte-identical to get_prefix_data's string ("### Response: " + newline + quoted title + newline)
        ID = tokenizer("### Response: \n" + "\"" + title + "\"\n").input_ids
        ID[4:4] = [thought_id] * n_thoughts
        ID.append(tokenizer.eos_token_id)
        e = [(get_hash(ID[:i]) if i == start else get_hash(ID[start:i]), ID[i]) for i in range(start, len(ID))]
        if title in edges:          # duplicate title string in the info file: same path, count once per title
            continue
        edges[title] = e
        count.update(e)
    return edges, count


def excluded_edges(titles, edges, count):
    """Edges to remove so that exactly `titles` (those present in the trie) become unreachable."""
    from collections import Counter
    c = Counter()
    for t in set(titles):
        if t in edges:
            c.update(edges[t])
    return {e for e, k in c.items() if k >= count[e]}

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""

class PrefixConstrainedLogitsProcessor(LogitsProcessor):

    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
                if len(prefix_allowed_tokens) == 0:
                    raise ValueError(
                        f"`prefix_allowed_tokens_fn` returned an empty list for batch ID {batch_id}."
                        f"This means that the constraint is unsatisfiable. Please check your implementation"
                        f"of `prefix_allowed_tokens_fn` "
                    )
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        scores_processed = scores + mask
        return scores_processed
    

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

class CFEnhancedLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        tokenizer,
        model,
        cf_logits,
        cf_dict,
        guidance_scale: float,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        n_thoughts: int = 1,
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.n_thoughts = n_thoughts
        self.model = model
        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }
        self._num_beams = num_beams
        self.guidance_scale = guidance_scale
        self.tokenizer = tokenizer
        self.cf_logits = cf_logits
        self.cf_dict = cf_dict
        self.count=0

    
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, -1000000)
        cf_score = torch.full_like(scores, 1.0)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                if self.count == 0:
                    # hash_key = sent[-4:]
                    hash_key = sent[-(4 + self.n_thoughts):]
                else:
                    hash_key=sent[-self.count:]
                hash_key = hash_key.tolist()
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key)

                if len(prefix_allowed_tokens) == 0:
                    continue 
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

                temp = []
                if self.cf_logits is not None:
                    # print(self.cf_logits)
                    for allow_token in prefix_allowed_tokens:
                        if self.count == 0:
                            cf_key = [allow_token]
                        else:
                            cf_key = hash_key + [allow_token]
                        if get_hash(cf_key) in self.cf_dict:
                            hash_value = self.cf_dict[get_hash(cf_key)]
                        else:
                            continue
                        
                        sublogits = self.cf_logits[hash_value]
                        temp.append(sublogits.sum() + 1e-20) # max or sum
                    temp = torch.tensor(temp)
                    temp = temp / temp.sum()
                    cf_score[batch_id * self._num_beams + beam_id].scatter_(dim = -1, index=torch.tensor(prefix_allowed_tokens).to(cf_score.device), src=temp.to(cf_score.device))
        cf_score = torch.log(cf_score)
        cf_score = cf_score + mask
        self.count += 1

        if self.guidance_scale == 1:
            scores = scores + mask
            return scores
        
        scores = scores + mask
        out = self.guidance_scale * (scores - cf_score) + cf_score
    
        return out
