"""R3's own representation of every catalog item title (candidate_pooling.py).

For each item in the info file: the R3 backbone's last-layer hidden states
(the space R3's pooling reads) over the quoted title tokens, as the title
appears in the prompt's history, averaged over the tokens.

Output (.pt): ``vectors`` [n_items, H] fp32 (row = item id), ``mean_norm``.
Usage: candidate_title_vectors.py --base_model <R3 checkpoint> --info_file <info txt> --output <.pt>
"""

import argparse
import os

import torch
from transformers import AutoTokenizer, Qwen2Model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--info_file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base_model)
    backbone = Qwen2Model.from_pretrained(args.base_model, torch_dtype=torch.bfloat16).to(device).eval()
    titles = {}
    for line in open(args.info_file):
        title, i = line.rstrip("\n").rsplit("\t", 1)
        titles[int(i)] = title.strip()
    n = max(titles) + 1
    vectors = torch.zeros(n, backbone.config.hidden_size)
    ids = sorted(titles)
    for s in range(0, len(ids), args.batch_size):
        chunk = ids[s:s + args.batch_size]
        enc = [tok.encode(f"\"{titles[i]}\"", add_special_tokens=False) for i in chunk]
        width = max(map(len, enc))
        x = torch.tensor([e + [tok.eos_token_id] * (width - len(e)) for e in enc], device=device)
        m = torch.tensor([[1] * len(e) + [0] * (width - len(e)) for e in enc], device=device)
        h = backbone(input_ids=x, attention_mask=m).last_hidden_state.float()
        vectors[chunk] = ((h * m[..., None]).sum(1) / m.sum(1, keepdim=True)).cpu()
    mean_norm = float(vectors[ids].norm(dim=-1).mean())
    print(f"{len(ids)} titles, mean norm {mean_norm:.2f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({"vectors": vectors, "mean_norm": mean_norm, "source": os.path.abspath(args.base_model)}, args.output)


if __name__ == "__main__":
    main()
