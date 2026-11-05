#!/usr/bin/env python3
"""Which operator removes the user-specific information of r?  (operator trace)

For test rows A (teacher-forced: prompt + <|Thought|> + target title) and a
donor row B of a different user, the decoder is run twice on A's input: with
A's own r and with B's r spliced into the thought slot (everything else equal).
The difference Δ = (A's r) − (B's r) is followed through every operator:

  pooling   r = Σ_j α_j W_V h_j            α: mass on the fixed instruction prefix vs user tokens;
                                           rel_r = ||r_A − r_B|| / ||r_A||
  per decoder layer l, at the thought position p (input of self_attn = after input_layernorm):
    norm    rel_in[l]  = ||h̃_A − h̃_B|| / ||h̃_A||       (does the difference survive RMSNorm / the
                                                          residual stream at p?)
    W_V     gain_V[l,g] = (||W_V Δh̃|| / ||Δh̃||) / (||W_V h̃_A|| / ||h̃_A||)   per KV group
            (< 1: the value projection suppresses the directions in which users differ)
    attn    a[l,h]     = attention weight from target positions t to p (mean over t), run A
            da[l,h]    = |a_A − a_B| (key path)
    W_O     gain_O[l,h] = same ratio for W_O^h applied to the head's value difference
    contrib share[l,h] = a·||W_O^h W_V Δh̃|| / ||head output at t||      (user-specific part of r in
                                                                         the head's output)
            total[l,h] = a·||W_O^h W_V h̃_A|| / ||head output at t||     (whole r, shared part included)
  residual  rel_res[l] = ||x_A − x_B|| / ||x_A|| at target positions after layer l
  sublayer  rel diff of the residual after attention vs after the MLP, per layer
  output    KL(p_A || p_B) over the target tokens (and ΔNLL of the true target), relative logit change
  control   r_A + a random direction with the same ||Δr|| (does the decoder ignore every change of r,
            or only the directions in which users differ?)

Reference: the same residual / KL numbers for dropping the most recent history
item (r recomputed), the decoder-insensitivity study's yardstick.

Usage: latent_operator_trace.py --checkpoint <R3 latent dir> --test_file <csv> --category musics
       --n 300 --output <json>
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "latent"), os.path.join(HERE, "..")]

from transformers import AutoTokenizer  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb  # noqa: E402

from layers import LatentModel  # noqa: E402
from reasoning_dataset import LatentRDataset  # noqa: E402


def pooling_stats(model, ids, mask, prefix_len):
    """r and the pooling's attention mass on the shared instruction prefix vs the user region."""
    out = LatentModel.__mro__[1].forward(model, input_ids=ids, attention_mask=mask, output_hidden_states=True)
    h = out.hidden_states[-1]
    att = model.attention
    t = (ids[0] == model.model.embed_tokens.num_embeddings - 1).nonzero()[0, 0]
    cos, sin = att.rope(h, position_ids=torch.arange(h.shape[1], device=h.device).unsqueeze(0))
    Q, K = att.query(h), att.key(h)
    Q, K = apply_rotary_pos_emb(Q, K, cos.squeeze(0), sin.squeeze(0), unsqueeze_dim=0)
    s = (Q[0, t - 1].float() @ K[0, :t].float().T) * att.scale_score
    w = F.softmax(s, -1)
    r = (w.to(h.dtype) @ att.value(h[0, :t])).float()
    return r, float(w[:prefix_len].sum()), float(w[prefix_len:].sum()), int(t)


def drop_last_history(row):
    row = row.copy()
    for c in ("history_item_title", "history_item_id"):
        v = eval(row[c])
        row[c] = str(v[:-1])
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_file", required=True)
    p.add_argument("--category", default="musics")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = LatentModel.from_pretrained(args.checkpoint, torch_dtype=dtype, attn_implementation="eager").to(device).eval()
    cfg = model.config
    L, H, G = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // H
    rep = H // G

    ds = LatentRDataset(train_file=args.test_file, tokenizer=tok, max_len=2560, test=False, category=args.category)
    frame = ds.data
    users = frame["user_id"].astype(str).to_numpy()
    rng = np.random.default_rng(args.seed)
    rows = rng.choice(len(frame), size=min(args.n, len(frame)), replace=False)
    prefix_len = len(ds.instruction_tokens())

    # inputs of self_attn (post input_layernorm) for every layer
    captured = {}

    def make_hook(l):
        def hook(module, a, kw):
            captured[l] = kw["hidden_states"][0].detach()
        return hook

    hooks = [layer.self_attn.register_forward_pre_hook(make_hook(l), with_kwargs=True)
             for l, layer in enumerate(model.model.layers)]
    post_attn = {}

    def make_post_hook(l):
        def hook(module, a):
            post_attn[l] = a[0][0].detach()
        return hook

    hooks += [layer.post_attention_layernorm.register_forward_pre_hook(make_post_hook(l))
              for l, layer in enumerate(model.model.layers)]

    def run(embeds, mask):
        captured.clear()
        post_attn.clear()
        out = LatentModel.__mro__[1].forward(model, inputs_embeds=embeds, attention_mask=mask,
                                             output_attentions=True, output_hidden_states=True)
        out.post_attn = {l: post_attn[l].clone() for l in range(L)}
        return out, {l: captured[l].clone() for l in range(L)}

    keys = ["rel_in", "gain_V", "a", "da", "gain_O", "share", "total"]
    acc = {k: [] for k in keys}
    acc_res, acc_res_attn, acc_res_rand = [], [], []
    kl_rand, dlogit, dlogit_rand = [], [], []
    pool_prefix, pool_user, rel_r, kl, kl_drop, dnll = [], [], [], [], [], []

    with torch.no_grad():
        for n_done, i in enumerate(rows):
            j = int(rng.integers(len(frame)))
            while users[j] == users[i]:
                j = int(rng.integers(len(frame)))
            item_a, item_b = ds[int(i)], ds[j]
            ids = torch.tensor([item_a["input_ids"]], device=device)
            mask = torch.ones_like(ids)
            labels = torch.tensor(item_a["labels"], device=device)
            ids_b = torch.tensor([item_b["input_ids"]], device=device)
            # r for A and B: pooling over each row's own prompt (prompt part only; causal, so the target
            # tokens after the thought do not affect it)
            r_a, m_pre, m_user, p = pooling_stats(model, ids, mask, prefix_len)
            r_b, _, _, _ = pooling_stats(model, ids_b, torch.ones_like(ids_b), prefix_len)
            pool_prefix.append(m_pre); pool_user.append(m_user)
            rel_r.append(float((r_a - r_b).norm() / r_a.norm()))
            emb = model.model.embed_tokens(ids)
            emb_a, emb_b = emb.clone(), emb.clone()
            emb_a[0, p] = r_a.to(emb.dtype); emb_b[0, p] = r_b.to(emb.dtype)
            out_a, cap_a = run(emb_a, mask)
            out_b, cap_b = run(emb_b, mask)
            tgt = torch.nonzero(labels != -100).squeeze(-1)   # target tokens
            qpos = tgt - 1                                    # positions that predict them
            lp_a = F.log_softmax(out_a.logits[0, qpos].float(), -1)
            lp_b = F.log_softmax(out_b.logits[0, qpos].float(), -1)
            kl.append(float((lp_a.exp() * (lp_a - lp_b)).sum(-1).mean()))
            dnll.append(float(-(lp_b.gather(1, ids[0, tgt][:, None]) - lp_a.gather(1, ids[0, tgt][:, None])).mean()))
            res = [float(((out_a.hidden_states[l + 1][0, qpos] - out_b.hidden_states[l + 1][0, qpos]).norm(dim=-1)
                          / out_a.hidden_states[l + 1][0, qpos].norm(dim=-1)).mean()) for l in range(L)]
            acc_res.append(res)
            acc_res_attn.append([float(((out_a.post_attn[l][qpos] - out_b.post_attn[l][qpos]).norm(dim=-1)
                                        / out_a.post_attn[l][qpos].norm(dim=-1)).mean()) for l in range(L)])
            dlogit.append(float(((out_a.logits[0, qpos] - out_b.logits[0, qpos]).float().norm(dim=-1)
                                 / out_a.logits[0, qpos].float().norm(dim=-1)).mean()))
            # control: random direction, same ||Δr|| as the swap
            g = torch.Generator().manual_seed(int(i))
            u = torch.randn(r_a.shape, generator=g).to(r_a.device)
            r_rand = r_a + u / u.norm() * (r_a - r_b).norm()
            emb_r = emb.clone(); emb_r[0, p] = r_rand.to(emb.dtype)
            out_r, _ = run(emb_r, mask)
            lp_r = F.log_softmax(out_r.logits[0, qpos].float(), -1)
            kl_rand.append(float((lp_a.exp() * (lp_a - lp_r)).sum(-1).mean()))
            dlogit_rand.append(float(((out_a.logits[0, qpos] - out_r.logits[0, qpos]).float().norm(dim=-1)
                                      / out_a.logits[0, qpos].float().norm(dim=-1)).mean()))
            acc_res_rand.append([float(((out_a.hidden_states[l + 1][0, qpos] - out_r.hidden_states[l + 1][0, qpos])
                                        .norm(dim=-1) / out_a.hidden_states[l + 1][0, qpos].norm(dim=-1)).mean())
                                 for l in range(L)])

            per = {k: np.zeros((L, H)) for k in keys}
            for l, layer in enumerate(model.model.layers):
                at = layer.self_attn
                ha, hb = cap_a[l][p].float(), cap_b[l][p].float()
                d = ha - hb
                per["rel_in"][l, :] = float(d.norm() / ha.norm())
                Wv = at.v_proj.weight.float().view(G, hd, -1)
                Wo = at.o_proj.weight.float().view(-1, H, hd)          # [hidden, H, hd]
                va = torch.einsum("gkd,d->gk", Wv, ha) + at.v_proj.bias.float().view(G, hd)
                dv = torch.einsum("gkd,d->gk", Wv, d)
                gv = (dv.norm(dim=-1) / d.norm()) / (va.norm(dim=-1) / ha.norm())
                # head outputs at target positions (run A): o_h,t = Σ_j a_tj v_j  (before W_O)
                x = cap_a[l].float()
                vall = (x @ at.v_proj.weight.float().T + at.v_proj.bias.float()).view(-1, G, hd)
                att_a = out_a.attentions[l][0].float()               # [H, T, T]
                att_b = out_b.attentions[l][0].float()
                for h in range(H):
                    g = h // rep
                    a = att_a[h, qpos, p]                             # [n_t]
                    o_h = att_a[h, qpos] @ vall[:, g]                 # [n_t, hd]
                    o_full = o_h @ Wo[:, h].T                         # [n_t, hidden]
                    wo_dv = Wo[:, h] @ dv[g]
                    wo_va = Wo[:, h] @ va[g]
                    per["gain_V"][l, h] = float(gv[g])
                    per["a"][l, h] = float(a.mean())
                    per["da"][l, h] = float((a - att_b[h, qpos, p]).abs().mean())
                    per["gain_O"][l, h] = float((wo_dv.norm() / dv[g].norm()) / (wo_va.norm() / va[g].norm()))
                    on = o_full.norm(dim=-1).clamp_min(1e-6)
                    per["share"][l, h] = float((a * wo_dv.norm() / on).mean())
                    per["total"][l, h] = float((a * wo_va.norm() / on).mean())
            for k in keys:
                acc[k].append(per[k])

            # reference: drop the most recent history item (r recomputed)
            row = drop_last_history(frame.iloc[int(i)])
            if len(eval(row["history_item_title"])) > 0:
                sub = LatentRDataset.__new__(LatentRDataset)
                sub.__dict__.update(ds.__dict__)
                sub.data = frame.iloc[[int(i)]].copy()
                sub.data.iloc[0] = row
                sub.get_inputs()
                item_d = sub[0]
                ids_d = torch.tensor([item_d["input_ids"]], device=device)
                lab_d = torch.tensor(item_d["labels"], device=device)
                out_d = model(input_ids=ids_d, attention_mask=torch.ones_like(ids_d))
                q_d = torch.nonzero(lab_d != -100).squeeze(-1) - 1
                lp_d = F.log_softmax(out_d.logits[0, q_d].float(), -1)
                if lp_d.shape == lp_a.shape:
                    kl_drop.append(float((lp_a.exp() * (lp_a - lp_d)).sum(-1).mean()))
            if (n_done + 1) % 25 == 0:
                print(f"{n_done + 1}/{len(rows)} rows; KL swap r {np.mean(kl):.5f}, KL drop item "
                      f"{np.mean(kl_drop) if kl_drop else float('nan'):.5f}", flush=True)
    for h_ in hooks:
        h_.remove()

    mean = {k: np.mean(acc[k], axis=0) for k in keys}
    res = np.mean(acc_res, axis=0)
    res_attn = np.mean(acc_res_attn, axis=0)
    res_rand = np.mean(acc_res_rand, axis=0)
    summary = {
        "checkpoint": args.checkpoint, "n": len(rows),
        "pooling_mass_prefix": float(np.mean(pool_prefix)), "pooling_mass_user": float(np.mean(pool_user)),
        "rel_r": float(np.mean(rel_r)),
        "kl_swap_r": float(np.mean(kl)), "dnll_swap_r": float(np.mean(dnll)),
        "kl_drop_last_item": float(np.mean(kl_drop)) if kl_drop else None,
        "kl_random_r": float(np.mean(kl_rand)), "rel_dlogit_swap": float(np.mean(dlogit)),
        "rel_dlogit_random": float(np.mean(dlogit_rand)),
        "per_layer": [],
    }
    print(f"\npooling mass prefix {summary['pooling_mass_prefix']:.4f} / user {summary['pooling_mass_user']:.4f}; "
          f"||r_A - r_B|| / ||r_A|| = {summary['rel_r']:.4f}")
    print(f"KL swap r {summary['kl_swap_r']:.5f} (dNLL {summary['dnll_swap_r']:+.5f}); "
          f"KL drop last item {summary['kl_drop_last_item']}; KL random-direction r {summary['kl_random_r']:.5f}")
    print(f"relative logit change at target positions: swap {summary['rel_dlogit_swap']:.5f}, "
          f"random {summary['rel_dlogit_random']:.5f}")
    print("layer | rel_in at p | gain_V | gain_O | attn a (mean / max head) | |da| | share of head out: user-specific "
          "(mean / max) | whole r (mean / max) | rel diff residual at targets")
    for l in range(L):
        row = {"layer": l, "rel_in": float(mean["rel_in"][l, 0]),
               "gain_V": float(mean["gain_V"][l].mean()), "gain_O": float(mean["gain_O"][l].mean()),
               "a_mean": float(mean["a"][l].mean()), "a_max": float(mean["a"][l].max()),
               "da_mean": float(mean["da"][l].mean()),
               "share_mean": float(mean["share"][l].mean()), "share_max": float(mean["share"][l].max()),
               "total_mean": float(mean["total"][l].mean()), "total_max": float(mean["total"][l].max()),
               "rel_res": float(res[l]), "rel_res_attn": float(res_attn[l]), "rel_res_random": float(res_rand[l]),
               "heads": {k: mean[k][l].tolist() for k in ("a", "share", "total", "gain_O")}}
        summary["per_layer"].append(row)
        print(f"{l:2d} | {row['rel_in']:.4f} | {row['gain_V']:.3f} | {row['gain_O']:.3f} | "
              f"{row['a_mean']:.3f} / {row['a_max']:.3f} | {row['da_mean']:.4f} | "
              f"{row['share_mean']:.4f} / {row['share_max']:.4f} | {row['total_mean']:.3f} / {row['total_max']:.3f} | "
              f"{row['rel_res_attn']:.5f} / {row['rel_res']:.5f} / {row['rel_res_random']:.5f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    json.dump(summary, open(args.output, "w"), indent=1)


if __name__ == "__main__":
    main()
