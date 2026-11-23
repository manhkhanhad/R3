"""HiLaR-style GRPO on a single-latent model (MW-R-H): Gaussian policy over the one thinking token r.

Port of the RL stage of HiLaR (arXiv 2607.27760, Sec. 3.3) to a model with ONE latent r = attention-pooled
state (no hierarchy, so no alignment / per-level gain terms).  The paper's code is unreleased; CHOICE marks
values we had to pick (see document notes in analysis/.../research_log.md, 2026-11-22/23).

Policy:   r_g = mu(H_u) + sigma * eps_g,  eps antithetic (+eps, -eps), G rollouts per user,
          sigma (per coordinate) = sigma_rel * D / sqrt(d), D = median distance between two users' mu
          (measured once on the start model; probe hilar_probe.py: sigma_rel 1.0 gives usable spread).
Rollout:  the decoder reads r_g in the thought slot; one title by catalog-constrained greedy decoding.
Reward:   R = hit + w_prefix * prefix + w_f1 * F1 + w_cf * clip(z-scored SASRec score of the decoded item, -3, 3)
              + dLL,  dLL = clip(mean target-token logp under r_g - under mu, -1, 1)   [teacher forced]
          CHOICE: w_prefix = w_f1 = 0.5, w_cf = 0.1 (as the earlier HiLaR reimplementation); no process terms.
Advantage: (R - mean_g R) / (std_g R + 1e-4).
Loss:     one update per batch (PPO ratio = 1): -mean_g A_g log N(r_g; mu_theta, sigma^2 I)
          + beta * ||mu_theta - mu_ref||^2 / (2 sigma^2)   (latent KL to the frozen start attention; the text KL has
          no gradient: the backbone is frozen and titles come from fixed sampled latents).
Only the pooling attention trains (fp32); the LLM is frozen.  Logs the antisymmetric / symmetric reward
components of the antithetic pairs: the antisymmetric part is the first-order (usable) policy-gradient signal.
"""
import copy, json, math, os, re, sys, time
from collections import Counter

import fire
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "latent"))
from transformers import AutoTokenizer, GenerationConfig, LogitsProcessorList
from model import LatentModel
from latent_grpo_dataset import D3Dataset, get_hash, get_prefix_data
from latent_grpo_processor import CFEnhancedLogitsProcessor
from rank_grpo import _strip
from hilar_probe import f1, prefix_score

CATEGORY_TEXT = {"Toys_and_Games": "toys and games", "CDs_and_Vinyl": "musics"}


class SasrecReward:
    """z-scored SASRec score of a decoded item among the catalog (frozen Cornac SASRec, CPU)."""

    def __init__(self, ckpt, csv_files):
        from sasrec_pair_features import load_sasrec
        self.rec = load_sasrec(ckpt)
        m = self.rec.model
        self.items = m.item_emb.weight.detach().float()[: self.rec.num_items]
        self.title2idx = {}
        for f in csv_files:
            df = pd.read_csv(f, usecols=["item_title", "item_asin"])
            for t, a in zip(df.item_title, df.item_asin):
                i = self.rec.iid_map.get(a, self.rec.num_items)
                if i < self.rec.num_items:
                    self.title2idx[_strip(str(t))] = i

    @torch.no_grad()
    def z_scores(self, asins_literal):
        import ast
        rec = self.rec
        hist = [rec.iid_map[a] for a in ast.literal_eval(asins_literal) if rec.iid_map.get(a, rec.num_items) < rec.num_items]
        hist = hist[-rec.max_len:]
        seq = torch.as_tensor([[rec.pad_idx] * (rec.max_len - len(hist)) + hist], dtype=torch.long)
        v = rec.model._encode(seq).float()[0]
        s = self.items @ v
        return ((s - s.mean()) / (s.std() + 1e-6)).clamp(-3, 3)

    def reward(self, z, titles):
        return np.array([float(z[self.title2idx[t]]) if t in self.title2idx else 0.0 for t in titles])


def main(
    base_model: str,
    train_file: str,
    info_file: str,
    output_dir: str,
    sasrec_ckpt: str,
    category: str = "Toys_and_Games",
    sigma_rel: float = 1.0,
    G: int = 12,
    batch_users: int = 8,
    lr: float = 5e-6,
    beta: float = 0.05,
    w_prefix: float = 0.5,
    w_f1: float = 0.5,
    w_cf: float = 0.1,
    warmup: int = 100,
    sample: int = -1,
    max_new_tokens: int = 128,
    dist_rows: int = 512,
    resume_every: int = 500,
    seed: int = 42,
):
    assert G % 2 == 0
    torch.manual_seed(seed)
    device = "cuda"
    os.makedirs(output_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.pad_token, tok.pad_token_id, tok.padding_side = tok.eos_token, tok.eos_token_id, "left"
    model = LatentModel.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device)
    model.attention.float()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.attention.parameters():
        p.requires_grad = True
    ref_attn = copy.deepcopy(model.attention).eval()
    for p in ref_attn.parameters():
        p.requires_grad = False
    model.eval()
    H = model.config.hidden_size
    thought_id = model.model.embed_tokens.num_embeddings - 1

    df = pd.read_csv(train_file)
    if sample > 0:
        df = df.sample(sample, random_state=seed)
    data_dir = os.path.dirname(os.path.dirname(train_file))
    csvs = [os.path.join(data_dir, s, f) for s in ("train", "valid", "test")
            for f in os.listdir(os.path.join(data_dir, s)) if f.endswith(".csv")]
    sas = SasrecReward(sasrec_ckpt, csvs)
    tmp_csv = os.path.join(output_dir, "rows_train.csv")
    df.to_csv(tmp_csv, index=False)
    rows = D3Dataset(train_file=tmp_csv, tokenizer=tok, max_len=2048, sample=-1, seed=seed,
                     category=CATEGORY_TEXT[category], test=False).get_inputs_list()
    asins = list(df["item_asins"])
    assert len(rows) == len(asins), (len(rows), len(asins))
    print(f"train rows {len(rows)}; SASRec catalog {sas.rec.num_items}, titles mapped {len(sas.title2idx)}", flush=True)

    hash_dict = get_prefix_data(info_file, tok)
    prefix_fn = lambda b, ids: hash_dict.get(get_hash(ids), [])
    state = {"noise": None}

    def hook(module, inp, out):
        if state["noise"] is not None:
            return out + state["noise"].to(out.dtype)
    model.attention.register_forward_hook(hook)

    def split(i):
        ids, lab = torch.tensor(rows[i]["input_ids"]), torch.tensor(rows[i]["labels"])
        return ids[lab == -100].to(device), ids[lab != -100].to(device), _strip(tok.decode(ids[lab != -100], skip_special_tokens=True))

    @torch.no_grad()
    def hidden(prompt):
        ids = prompt.unsqueeze(0)
        att = torch.ones_like(ids)
        hs = model.model(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states[-1]
        return hs.float(), att, torch.nonzero(ids == thought_id)[:, 1]

    @torch.no_grad()
    def decode(prompt, B):
        ids = prompt.unsqueeze(0).repeat(B, 1)
        proc = CFEnhancedLogitsProcessor(guidance_scale=1.0, cf_logits=None, prefix_allowed_tokens_fn=prefix_fn,
                                         cf_dict=None, unconditional_ids=None, model=model, tokenizer=tok, num_beams=1)
        cfg = GenerationConfig(num_beams=1, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                               eos_token_id=tok.eos_token_id, use_cache=True)
        out = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=cfg,
                             logits_processor=LogitsProcessorList([proc]))
        return [_strip(t) for t in tok.batch_decode(out[:, prompt.numel():], skip_special_tokens=True)]

    @torch.no_grad()
    def target_logp(prompt, tgt, B):
        full = torch.cat([prompt, tgt]).unsqueeze(0).repeat(B, 1)
        T = tgt.numel()
        # transformers 4.48 ignores logits_to_keep: slice the full logits ourselves
        lg = model(input_ids=full, attention_mask=torch.ones_like(full)).logits[:, -(T + 1):-1].float()
        lp = torch.log_softmax(lg, -1).gather(-1, tgt.view(1, -1, 1).expand(B, -1, 1)).squeeze(-1)
        return lp.mean(1).cpu().numpy()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))

    # sigma from the median distance between two users' mu on the start model
    with torch.no_grad():
        mus = []
        for i in order[:dist_rows]:
            prompt, _, _ = split(i)
            hs, att, w = hidden(prompt)
            mus.append(model.attention(hidden_states=hs, attention_mask=att, thought_id_idx=w).float()[0].cpu())
        M = torch.stack(mus)
        a, b = rng.integers(0, len(M), 4000), rng.integers(0, len(M), 4000)
        keep = a != b
        D = float((M[a[keep]] - M[b[keep]]).norm(dim=1).median())
    sigma = sigma_rel * D / math.sqrt(H)
    print(f"median inter-user ||mu_i - mu_j|| = {D:.2f}, mean ||mu|| = {float(M.norm(dim=1).mean()):.2f}; "
          f"sigma_rel {sigma_rel} -> per-coordinate sigma {sigma:.3f}", flush=True)

    params = [p for p in model.attention.parameters()]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    n_steps = len(rows) // batch_users
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    resume_path = os.path.join(output_dir, "resume.pt")
    step0 = 0
    if os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device)
        model.attention.load_state_dict(ck["attention"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); step0 = ck["step"]
        print(f"resumed at step {step0}", flush=True)

    log = open(os.path.join(output_dir, "train_log.jsonl"), "a")
    t0 = time.time()
    for step in range(step0, n_steps):
        opt.zero_grad(set_to_none=True)
        agg = {k: [] for k in ("R", "sd", "distinct", "hit", "dll", "cf", "anti", "sym", "dmu")}
        for i in order[step * batch_users:(step + 1) * batch_users]:
            prompt, tgt, target = split(i)
            state["noise"] = None
            hs, att, w = hidden(prompt)
            with torch.no_grad():
                mu_ref = ref_attn(hidden_states=hs, attention_mask=att, thought_id_idx=w).float()
            mu = model.attention(hidden_states=hs, attention_mask=att, thought_id_idx=w).float()  # [1,H], grad
            eps = torch.randn(G // 2, H, device=device)
            eps = torch.cat([eps, -eps])
            state["noise"] = sigma * eps
            titles = decode(prompt, G)
            ll = target_logp(prompt, tgt, G)
            state["noise"] = None
            ll0 = target_logp(prompt, tgt, 1)[0]
            dll = np.clip(ll - ll0, -1, 1)
            hit = np.array([t == target for t in titles], float)
            pre = np.array([prefix_score(t, target) for t in titles])
            f1s = np.array([f1(t, target) for t in titles])
            cf = sas.reward(sas.z_scores(asins[i]), titles)
            R = hit + w_prefix * pre + w_f1 * f1s + w_cf * cf + dll
            adv = torch.tensor((R - R.mean()) / (R.std() + 1e-4), device=device, dtype=torch.float32)
            diff = mu.detach() + sigma * eps - mu                       # [G,H]; value sigma*eps, grad wrt mu
            logp = -(diff ** 2).sum(1) / (2 * sigma ** 2)
            pg = -(adv * logp).mean()
            kl = ((mu - mu_ref) ** 2).sum() / (2 * sigma ** 2)
            ((pg + beta * kl) / batch_users).backward()
            half = G // 2
            agg["R"].append(R.mean()); agg["sd"].append(R.std()); agg["distinct"].append(len(set(titles)))
            agg["hit"].append(hit.mean()); agg["dll"].append(dll.mean()); agg["cf"].append(cf.mean())
            agg["anti"].append(((R[:half] - R[half:]) / 2).std()); agg["sym"].append(((R[:half] + R[half:]) / 2).std())
            agg["dmu"].append(float((mu.detach() - mu_ref).norm()))
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        rec = {"step": step, **{k: float(np.mean(v)) for k, v in agg.items()}, "grad_norm": float(gn),
               "lr": sched.get_last_lr()[0], "elapsed_h": (time.time() - t0) / 3600}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if step % 20 == 0:
            print(" ".join(f"{k}={v:.4g}" for k, v in rec.items()), f"/ {n_steps}", flush=True)
        if resume_every and (step + 1) % resume_every == 0:
            torch.save({"attention": model.attention.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step + 1}, resume_path + ".tmp")
            os.replace(resume_path + ".tmp", resume_path)
    model.attention.float()
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    if os.path.exists(resume_path):
        os.remove(resume_path)
    print("saved", output_dir, flush=True)


if __name__ == "__main__":
    fire.Fire(main)
