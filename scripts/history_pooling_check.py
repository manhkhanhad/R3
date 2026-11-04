"""Check for history-only pooling (history_pooling.py) on a trained checkpoint:
pooling weight on non-title tokens must be 0 (and, for reference, how much of plain
pooling would sit on the template). Prints per-row sums over N test rows."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "latent"))
import torch
from transformers import AutoTokenizer
from layers import LatentModel, SelfAttentionLayer, apply_rotary_pos_emb
from reasoning_dataset import LatentRDataset
from history_pooling import history_title_mask

CATS = {"Toys_and_Games": "toys and games", "Musical_Instruments": "music instruments", "CDs_and_Vinyl": "musics"}


def weights(layer, h, mask, pos):
    m = SelfAttentionLayer.mask_to_weights(mask.clone(), pos, layer.end_k)
    cos, sin = layer.rope(h, position_ids=torch.arange(h.size(1), device=h.device)[None])
    q, k = apply_rotary_pos_emb(layer.query(h), layer.key(h), cos.squeeze(0), sin.squeeze(0), unsqueeze_dim=0)
    qs = q[torch.arange(h.size(0)), pos - 1][:, None]
    return torch.softmax((qs @ k.transpose(-2, -1)) * layer.scale_score + m[:, None], -1)[:, 0]


ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint"); ap.add_argument("--test_file"); ap.add_argument("--category")
ap.add_argument("--rows", type=int, default=64)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.checkpoint)
model = LatentModel.from_pretrained(a.checkpoint, torch_dtype=torch.float32).cuda().eval()
spec = model.config.latent_pool_history
assert spec, "checkpoint has no latent_pool_history"
d = LatentRDataset(a.test_file, tok, sample=a.rows, category=CATS[a.category], test=True)
tid = model.model.embed_tokens.num_embeddings - 1
off, on = [], []
with torch.no_grad():
    for i in range(len(d)):
        ids = torch.tensor(d[i]["input_ids"])[None].cuda(); full = torch.ones_like(ids)
        h = model.model(input_ids=ids, attention_mask=full).last_hidden_state.float()
        pos = (ids[0] == tid).nonzero()[0]
        hm = history_title_mask(ids, full, spec, tid)
        w = weights(model.attention, h, model.pooling_mask(ids, full), pos)[0]
        w_plain = weights(model.attention, h, full, pos)[0]
        off.append(float(w[~hm[0]].sum())); on.append(float(w_plain[~hm[0]].sum()))
print(f"rows {len(off)}: max pooling weight on non-title tokens (history-only pooling) {max(off):.3e}")
print(f"same checkpoint with the plain mask: mean weight on non-title tokens {sum(on)/len(on):.3f}")
assert max(off) < 1e-6, "pooling leaks onto non-title tokens"
print("history pooling check passed")
