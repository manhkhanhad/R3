"""GRPO stage 2 with a ranking reward (attention-only, LLM frozen).

Stock R3 stage 2 (grpo_trainer.NoiseGRPORecTrainer) never samples: every group member is the
ground-truth title under a different Gaussian noise on r, and the reward is the target's
perplexity.  Here the group is the model's own ranked list:

  * for each prompt, constrained beam search (the evaluation decoder: same prefix-tree
    constraint, num_beams = num_return_sequences = G) returns the top-G titles, in rank order;
  * reward of the beam at rank k (1 = best):  +1 if it is the target title, otherwise
    -alpha / log2(k + 1)  (a wrong title ranked high is penalised more; MiniOneRec/ReRe-style
    ranking reward, so groups without a hit still give a signal);
  * advantage = (reward - group mean) / (group std + 1e-4)  (standard GRPO);
    with ``skip_nohit`` (default) a group whose beams all miss the target gets advantage 0:
    its rewards depend only on rank position, so the gradient would just push down whatever
    is ranked first and flatten every list (361059/361060 lowered their own hit rate);
  * the loss is the parent's: token-level clipped surrogate on log p(beam | prompt, r) plus
    beta * k3 KL to the reference.  With ``kl_anchor`` the reference scores the beams with
    its own frozen stage-1 r.  No noise is added to r (zeros are passed to the parent).

Only the pooling attention trains (the caller freezes the LLM), so every change of the ranked
list has to go through the latent r.
"""

import math
import re

import torch
from transformers import GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper
from trl.trainer.grpo_trainer import GRPOTrainer

from grpo_trainer import NoiseGRPORecTrainer
from latent_grpo_processor import CFEnhancedLogitsProcessor


def _strip(s):
    return s.strip().strip("\n").strip('"').strip()


def _tokens(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _brand(s):
    return (re.findall(r"[a-z0-9']+", s.lower()) + [""])[0]


def _jaccard(a, b):
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / max(1, len(ta | tb))


def list_penalties(titles, is_target, prompt_text, dup_jaccard=0.5):
    """C: per-beam list-quality flags (history item; redundant = same brand and title Jaccard >= dup_jaccard
    with a higher-ranked non-target beam).  The injected/hit target is exempt."""
    hist, dup = [], []
    for k, t in enumerate(titles):
        if is_target[k]:
            hist.append(0.0); dup.append(0.0); continue
        hist.append(float(f'"{t}"' in prompt_text))
        dup.append(float(any((not is_target[j]) and _brand(titles[j]) == _brand(t) and _jaccard(titles[j], t) >= dup_jaccard
                             for j in range(k))))
    return hist, dup


def rank_rewards(hits, alpha):
    """hits [G] bool in rank order -> rewards [G]: +1 for the target, -alpha/log2(k+1) otherwise."""
    return [1.0 if h else -alpha / math.log2(k + 2) for k, h in enumerate(hits)]


class RankGRPORecTrainer(NoiseGRPORecTrainer):
    def __init__(self, *args, rank_penalty=0.5, skip_nohit=True, gen_max_new_tokens=128,
                 inject_target=False, hist_penalty=0.0, dup_penalty=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        # A (inject_target): a group whose beams all miss gets the ground-truth title in place of its
        #   lowest-ranked beam (reward +1), so every row carries a signal.
        # C (hist_penalty / dup_penalty): a beam that is a history item / a redundant variant of a
        #   higher-ranked beam gets an extra -hist_penalty / -dup_penalty.
        self.inject_target = bool(inject_target)
        self.hist_penalty = float(hist_penalty)
        self.dup_penalty = float(dup_penalty)
        self.rank_penalty = float(rank_penalty)
        self.skip_nohit = bool(skip_nohit)
        self.gen_max_new_tokens = int(gen_max_new_tokens)
        print(f"rank GRPO: G={self.num_generations} beams, reward +1 hit / -{self.rank_penalty}/log2(k+1) miss, "
              f"beta={self.beta}, kl_anchor={self.kl_anchor}, groups without a hit: "
              f"{'advantage 0' if self.skip_nohit else 'rank penalty only'}; inject_target={self.inject_target} "
              f"hist_penalty={self.hist_penalty} dup_penalty={self.dup_penalty}", flush=True)

    @torch.no_grad()
    def _beam_search(self, model, prompts):
        """prompts: list of 1-D LongTensors (unpadded) -> sequences [len(prompts)*G, P+T], P."""
        tok = self.processing_class
        G = self.num_generations
        P = max(len(p) for p in prompts)
        pad = tok.pad_token_id
        ids = torch.stack([torch.cat([p.new_full((P - len(p),), pad), p]) for p in prompts])
        att = torch.stack([torch.cat([p.new_zeros(P - len(p)), torch.ones_like(p)]) for p in prompts])
        processor = CFEnhancedLogitsProcessor(
            guidance_scale=1.0, cf_logits=None, prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
            cf_dict=None, unconditional_ids=None, model=model, tokenizer=tok, num_beams=G)
        config = GenerationConfig(num_beams=G, num_return_sequences=G, length_penalty=1.0,
                                  max_new_tokens=self.gen_max_new_tokens, pad_token_id=pad,
                                  eos_token_id=tok.eos_token_id, use_cache=True, do_sample=False)
        was_training = model.training
        model.eval()
        try:
            out = model.generate(ids, attention_mask=att, generation_config=config,
                                 logits_processor=LogitsProcessorList([TemperatureLogitsWarper(1.0), processor]))
        finally:
            model.train(was_training)
        return out, P

    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        tok = self.processing_class
        G = self.num_generations
        batch = super(GRPOTrainer, self)._prepare_inputs(inputs)
        ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        B = ids.size(0)
        if B % G:
            raise ValueError(f"micro-batch {B} is not a multiple of num_generations {G}")
        model = self.accelerator.unwrap_model(self.model)

        # one prompt per group (TRL repeats each prompt G times consecutively)
        prompts, targets, batch_target_ids = [], [], []
        for g in range(B // G):
            r = g * G
            keep = mask[r].bool()
            t, l = ids[r][keep], labels[r][keep]
            prompts.append(t[l == -100])
            batch_target_ids.append(t[l != -100])
            targets.append(_strip(tok.decode(t[l != -100], skip_special_tokens=True)))

        out, P = self._beam_search(model, prompts)
        completion_ids = out[:, P:]
        eos = tok.eos_token_id
        is_eos = completion_ids == eos
        first = torch.where(is_eos.any(1), is_eos.int().argmax(1),
                            torch.full((B,), completion_ids.size(1) - 1, device=device, dtype=torch.long))
        completion_mask = (torch.arange(completion_ids.size(1), device=device)[None] <= first[:, None]).long()

        texts = tok.batch_decode(completion_ids, skip_special_tokens=True)
        prompt_texts = tok.batch_decode(torch.stack([torch.cat([p.new_full((P - len(p),), tok.pad_token_id), p])
                                                     for p in prompts]), skip_special_tokens=True)
        use_c = self.hist_penalty > 0 or self.dup_penalty > 0
        rewards, hit_any, hit_rank, advmask = [], [], [], []
        n_inj = n_hist = n_dup = n_slots = n_match = 0
        brands = []
        for g in range(B // G):
            titles = [_strip(texts[g * G + k]) for k in range(G)]
            is_t = [x == targets[g] for x in titles]
            natural_hit = any(is_t)
            if not natural_hit and self.inject_target:
                # replace the lowest-ranked beam by the target completion (labels: title + newline + eos)
                tt = batch_target_ids[g].to(device)
                if tt.numel() > completion_ids.size(1):
                    completion_ids = torch.nn.functional.pad(completion_ids, (0, tt.numel() - completion_ids.size(1)),
                                                             value=tok.pad_token_id)
                    completion_mask = torch.nn.functional.pad(completion_mask, (0, tt.numel() - completion_mask.size(1)))
                row = g * G + G - 1
                completion_ids[row] = tok.pad_token_id
                completion_ids[row, :tt.numel()] = tt
                completion_mask[row] = 0
                completion_mask[row, :tt.numel()] = 1
                titles[-1] = targets[g]; is_t[-1] = True; n_inj += 1
            hits = is_t
            r = rank_rewards(hits, self.rank_penalty)
            if use_c:
                hist, dup = list_penalties(titles, is_t, prompt_texts[g])
                r = [x - self.hist_penalty * h - self.dup_penalty * d for x, h, d in zip(r, hist, dup)]
                if not any(hits):  # no target to rank: only the list-quality penalties (no rank penalty that flattens)
                    r = [-self.hist_penalty * h - self.dup_penalty * d for h, d in zip(hist, dup)]
                n_hist += sum(hist); n_dup += sum(dup); n_slots += G
            rewards += r
            hit_any.append(float(natural_hit)); advmask.append(1.0 if (any(hits) or use_c or not self.skip_nohit) else 0.0)
            brands.append(len(set(_brand(t) for t in titles)))
            if natural_hit:
                hit_rank.append(is_t.index(True) + 1)
                tt = batch_target_ids[g].to(device); kk = is_t.index(True)
                n_match += int(torch.equal(completion_ids[g * G + kk, :tt.numel()], tt))  # label ids == beam ids
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        grouped = rewards.view(-1, G)
        advantages = (grouped - grouped.mean(1, keepdim=True)) / (grouped.std(1, keepdim=True) + 1e-4)
        if self.skip_nohit and not self.inject_target:
            advantages = advantages * torch.tensor(advmask, device=device).unsqueeze(1)
        advantages = advantages.view(-1)

        prompt_ids = out[:, :P]
        prompt_mask = (prompt_ids != tok.pad_token_id).long()
        # left padding only; the prompt itself never contains the pad (= eos) token
        input_ids = torch.cat([prompt_ids, completion_ids], 1)
        attention_mask = torch.cat([prompt_mask, completion_mask], 1)
        logits_to_keep = completion_ids.size(1)
        hidden = model.config.hidden_size
        noise = torch.zeros((B, hidden), device=device)

        ref_per_token_logps = None
        if self.beta != 0.0:
            with torch.no_grad():
                ref = self.accelerator.unwrap_model(self.ref_model)
                src = ref if self.kl_anchor else model
                embeds = src.generate_embs(input_ids, attention_mask)
                ref_per_token_logps = self.my_get_per_token_logps(ref, input_ids, embeds, attention_mask,
                                                                  logits_to_keep)

        m = self._metrics["train"]
        m["rank/reward_mean"].append(rewards.mean().item())
        m["rank/group_hit_rate"].append(sum(hit_any) / len(hit_any))
        if hit_rank:
            m["rank/hit_rank"].append(sum(hit_rank) / len(hit_rank))
        if self.inject_target:
            m["rank/injected_frac"].append(n_inj / (B // G))
        if use_c and n_slots:
            m["rank/hist_slot_frac"].append(n_hist / n_slots)
            m["rank/dup_slot_frac"].append(n_dup / n_slots)
        if hit_rank:
            m["rank/label_ids_match"].append(n_match / len(hit_rank))
        m["rank/distinct_brands"].append(sum(brands) / len(brands))
        m["rank/completion_len"].append(completion_mask.sum(1).float().mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": None,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "original_embeds": None,
            "noise": noise,
        }
