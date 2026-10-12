"""HiLaR stage 3: hierarchical reward-guided GRPO on the latent policy (document/hilar.md).

Re-implements Sec. 3.3 of HiLaR (Hu et al., arXiv 2607.27760) on a stage-2 checkpoint.
CHOICE marks details the paper leaves to its (unreleased) supplement.

Policy: each latent state is Gaussian, tau_k = mu_k + sigma_k eps_k, with
mu_k = head(state before slot k) conditioned on the history and tau_<k, and
sigma = (0.12, 0.08, 0.05, 0.025).  As in the paper only the latent module (the
reasoning head) and the alignment projection are trained; the backbone is frozen.

Per training user, G = 6 rollouts: sample the chain, decode a title by catalog-
constrained greedy search, and score
  final:   R_hit (exact title) + w_prefix * R_prefix (common prefix / target length)
           + w_f1 * R_f1 (title token F1) + w_cf * clip(z-scored SASRec score, -3, 3)
  process: R_align = mean_k cos(Proj(tau_k), e_{u,k})
           R_gain  = sum_k rho_k clip(LL_k - LL_{k-1}, -b, b), LL_k = mean target-token
                     log-likelihood with the first k latents (later slots left empty)
  R = R_final + lambda_proc (R_align + lambda_gain R_gain),  lambda_proc = 0.2
CHOICE: w_prefix = w_f1 = 0.5, w_cf = 0.1, rho_k = 1, b = 1, lambda_gain = 1.
Advantages are group-normalised.  One update per batch (the PPO ratio is 1 at the
update point, so the clipped objective reduces to the policy gradient):
  L = -mean_g A_g * sum_k log N(tau_k; mu_k, sigma_k^2) + beta * KL_latent,  beta = 0.05,
KL_latent = sum_k ||mu_k - mu_k^ref||^2 / (2 sigma_k^2) against the frozen stage-2 head.
The paper's text KL term has no gradient here: titles are decoded from fixed sampled
latents and the backbone is frozen.

--probe_only measures the within-group reward spread without updating: if rollouts of
the same user get the same reward, GRPO has nothing to learn from (as R3's own latent
RL, latent_distillation.md).
"""

import copy
import json
import math
import os
import re
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import fire
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GenerationConfig, LogitsProcessorList

from hilar_latent import HiLaRLatentModel
from larec_latent import LaRecLatentRDataset
from LogitProcesser import CFEnhancedLogitsProcessor, get_hash, get_prefix_data

SIGMA = (0.12, 0.08, 0.05, 0.025)


def norm_title(t):
    return str(t).strip().strip('"').strip()


def title_rewards(pred, target):
    p, t = norm_title(pred), norm_title(target)
    hit = float(p == t)
    common = 0
    for a, b in zip(p, t):
        if a != b:
            break
        common += 1
    prefix = common / max(len(t), 1)
    pw, tw = re.findall(r"\w+", p.lower()), re.findall(r"\w+", t.lower())
    overlap = sum(min(pw.count(w), tw.count(w)) for w in set(pw))
    f1 = 0.0 if not overlap else 2 * overlap / (len(pw) + len(tw))
    return hit, prefix, f1


class SASRecScorer:
    """z-scored dot product between a training row's SASRec sequence vector and an item."""

    def __init__(self, item_table, train_csv):
        from sasrec_pair_features import encode_rows, load_sasrec
        table = torch.load(item_table, weights_only=False)
        self.items = F.normalize(table["embeddings"].float(), dim=-1)
        self.valid = table["valid"].bool()
        frame = pd.read_csv(train_csv, usecols=["item_asins", "item_asin"])
        users, _ = encode_rows(load_sasrec(table["source"]), frame)
        self.users = torch.from_numpy(users)

    def __call__(self, row, item_id):
        if item_id is None or item_id < 0 or not bool(self.valid[item_id]):
            return -3.0
        scores = self.items[self.valid] @ self.users[row]
        s = float(self.items[item_id] @ self.users[row])
        return float(np.clip((s - float(scores.mean())) / float(scores.std() + 1e-6), -3, 3))


def left_pad(rows, value):
    width = max(len(r) for r in rows)
    return [[value] * (width - len(r)) + list(r) for r in rows]


def right_pad(rows, value):
    width = max(len(r) for r in rows)
    return [list(r) + [value] * (width - len(r)) for r in rows]


@torch.no_grad()
def target_loglik(model, prompts, target_ids, device, taus, keep):
    """Mean target-token log-likelihood with the first `keep` latents (replayed)."""
    seqs = [p + t for p, t in zip(prompts, target_ids)]
    ids = torch.tensor(right_pad(seqs, 0), device=device)
    mask = torch.tensor(right_pad([[1] * len(s) for s in seqs], 0), device=device)
    model.replay_tau, model.keep_levels = taus, keep
    try:
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
    finally:
        model.replay_tau, model.keep_levels = None, None
    logp = logits[:, :-1].log_softmax(-1)
    out = []
    for i, (p, t) in enumerate(zip(prompts, target_ids)):
        positions = torch.arange(len(p) - 1, len(p) - 1 + len(t), device=device)
        out.append(logp[i, positions, torch.tensor(t, device=device)].mean())
    return torch.stack(out)


def main(checkpoint: str, targets: str, train_csv: str, info_file: str, output_dir: str,
         sasrec_items: str = "data/Musical_Instruments/info/Musical_Instruments_sasrec_item.pt",
         steps: int = 2000, batch_users: int = 4, group: int = 6, lr: float = 5e-6, beta: float = 0.05,
         lambda_proc: float = 0.2, lambda_gain: float = 1.0, w_prefix: float = 0.5, w_f1: float = 0.5,
         w_cf: float = 0.1, gain_clip: float = 1.0, max_new_tokens: int = 64, probe_only: bool = False,
         log_every: int = 10, save_every: int = 500, seed: int = 0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cuda"
    model = HiLaRLatentModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
    table = torch.load(targets, weights_only=False)
    model.set_targets(table["e"])
    k = model.k
    if k != len(SIGMA):
        raise ValueError(f"sigma is defined for 4 levels, model has {k}")
    sigma = torch.tensor(SIGMA, device=device)
    for p in model.parameters():
        p.requires_grad = False
    trainable = list(model.head.parameters()) + list(model.align_proj_q.parameters())
    for p in trainable:
        p.requires_grad = True
    ref_head = copy.deepcopy(model.head).eval()
    for p in ref_head.parameters():
        p.requires_grad = False
    optim = torch.optim.AdamW(trainable, lr=lr)
    model.eval()  # no dropout; gradients still flow to the trainable parameters

    tok = AutoTokenizer.from_pretrained(checkpoint)
    tok.pad_token, tok.pad_token_id = tok.eos_token, tok.eos_token_id
    data = LaRecLatentRDataset(train_file=train_csv, tokenizer=tok, max_len=1024, sample=-1,
                               category="music instruments", test=True, k=k)
    frame = pd.read_csv(train_csv, usecols=["item_title", "item_id"])
    title_to_id = {}
    for line in open(info_file):
        t, i = line.rstrip("\n").rsplit("\t", 1)
        title_to_id[norm_title(t)] = int(i)
    scorer = SASRecScorer(sasrec_items, train_csv) if w_cf else None

    hash_dict = get_prefix_data(info_file, tok)
    thought = len(tok) - 1
    first = data[0]["input_ids"]
    first_key = get_hash(first[first.index(thought) - 4:first.index(thought) + 1])

    def allowed(batch_id, input_ids):
        key = get_hash(input_ids)
        if key not in hash_dict and len(input_ids) == 5 and all(int(t) == thought for t in input_ids):
            key = first_key
        return hash_dict.get(key, [])

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "probe.jsonl" if probe_only else "train.jsonl")
    log = open(log_path, "w")
    window, start_time = [], time.time()
    valid_rows = np.nonzero(table["valid"].numpy())[0]
    for step in range(1, steps + 1):
        rows = rng.choice(valid_rows, batch_users, replace=False)
        prompts = [data[int(r)]["input_ids"] for r in rows for _ in range(group)]
        target_titles = [frame.item_title[int(r)] for r in rows for _ in range(group)]
        target_ids = [tok.encode('"' + str(t) + '"\n', add_special_tokens=False) + [tok.eos_token_id]
                      for t in target_titles]  # as LatentRDataset.pre
        n = len(prompts)
        ids = torch.tensor(left_pad(prompts, tok.pad_token_id), device=device)
        mask = torch.tensor(left_pad([[1] * len(p) for p in prompts], 0), device=device)

        # 1. rollouts: sample latent chains, decode titles (constrained greedy)
        eps = torch.randn(n, k, model.config.hidden_size, device=device)
        model.rollout_sigma, model.rollout_eps = sigma, eps
        processor = CFEnhancedLogitsProcessor(guidance_scale=1.0, cf_logits=None, prefix_allowed_tokens_fn=allowed,
                                              cf_dict=None, unconditional_ids=None, model=model, tokenizer=tok,
                                              num_beams=1)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=mask, logits_processor=LogitsProcessorList([processor]),
                                 generation_config=GenerationConfig(num_beams=1, do_sample=False,
                                                                    max_new_tokens=max_new_tokens,
                                                                    pad_token_id=tok.pad_token_id,
                                                                    eos_token_id=tok.eos_token_id))
        taus = model.last_chain["heads"].detach()
        model.rollout_sigma, model.rollout_eps = None, None
        titles = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)

        # 2. rewards
        final, hits = [], []
        for i, (pred, tgt) in enumerate(zip(titles, target_titles)):
            hit, prefix, f1 = title_rewards(pred, tgt)
            cf = scorer(int(rows[i // group]), title_to_id.get(norm_title(pred))) if scorer else 0.0
            final.append(hit + w_prefix * prefix + w_f1 * f1 + w_cf * cf)
            hits.append(hit)
        final = torch.tensor(final, device=device)
        with torch.no_grad():
            e = model.targets[torch.as_tensor(np.repeat(rows, group))].to(device).float()
            proj = model.align_proj_q(taus.to(model.align_proj_q.weight.dtype)).float()
            r_align = F.cosine_similarity(proj, e, dim=-1).mean(1)
            ll = [target_loglik(model, prompts, target_ids, device, taus, keep) for keep in range(k + 1)]
            gains = torch.stack([ll[j + 1] - ll[j] for j in range(k)], 1).clamp(-gain_clip, gain_clip)
            r_gain = gains.sum(1)
        reward = final + lambda_proc * (r_align + lambda_gain * r_gain)
        grouped = reward.view(batch_users, group)
        adv = ((grouped - grouped.mean(1, keepdim=True)) / (grouped.std(1, keepdim=True) + 1e-4)).view(-1)

        record = {"step": step, "reward": float(reward.mean()), "final": float(final.mean()),
                  "hit": float(np.mean(hits)), "align": float(r_align.mean()), "gain": float(r_gain.mean()),
                  "group_std": float(grouped.std(1).mean()), "group_std_final": float(final.view(batch_users, group).std(1).mean()),
                  "distinct_titles_per_user": float(np.mean([len(set(titles[u * group:(u + 1) * group])) for u in range(batch_users)])),
                  "ll_full": float(ll[-1].mean()), "ll_none": float(ll[0].mean())}

        # 3. policy update on the head (and projection) only
        if not probe_only:
            model.replay_tau = taus
            try:
                model.generate_embs(ids, mask)  # states without grad; mu_k = head(state) with grad
                mus = model._trace["mus"].float()
                head_inputs = model._trace["head_inputs"].detach()
            finally:
                model.replay_tau = None
            with torch.no_grad():
                mus_ref = ref_head(head_inputs).float()
            s2 = sigma.view(1, k, 1) ** 2
            logp = -((taus.float() - mus) ** 2 / (2 * s2)).sum(-1).sum(-1)
            kl = ((mus - mus_ref) ** 2 / (2 * s2)).sum(-1).sum(-1)
            loss = -(adv.detach() * logp).mean() / model.config.hidden_size + beta * kl.mean() / model.config.hidden_size
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            record.update({"loss": float(loss), "kl": float(kl.mean())})
        window.append(record)
        log.write(json.dumps(record) + "\n")
        if step % log_every == 0:
            avg = {key: float(np.mean([r[key] for r in window])) for key in window[0] if key != "step"}
            print(f"step {step} ({time.time() - start_time:.0f}s) " + " ".join(f"{a}={b:.4f}" for a, b in avg.items()), flush=True)
            window = []
        if not probe_only and step % save_every == 0:
            model.save_pretrained(output_dir)
            tok.save_pretrained(output_dir)
    log.close()
    if not probe_only:
        model.save_pretrained(output_dir)
        tok.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(main)
