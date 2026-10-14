"""Embed target-conditioned teacher explanations, one row per train-CSV row.

Input: the SCoTER-style explanation file (``candidates.jsonl``), one record per
train interaction with ``row_index`` into the R3 train CSV and ``cot.steps``
(stage, text).  The teacher saw the observed next item privately, so these
embeddings are training-only targets for latent distillation; there are none
for valid/test.

Three views per row:

* ``full``           every step, ``"<stage>: <text>"`` joined by newlines;
* ``post_evidence``  steps 3..end (drops "Analyze the purchase pattern" and
                     "Extract description evidence", which restate the history);
* ``feature``        the final "Feature prediction" step only, the part that
                     carries the teacher's view of the next item.

Output (``torch.save``): ``{view: float16 [n_rows, dim]}`` with zero rows where
no explanation exists, ``mask`` (bool [n_rows]), and provenance.  Rows are
aligned to the train CSV and verified by comparing the explanation's history
titles with the CSV history.

Embedding model defaults to Qwen3-Embedding-8B truncated to 1536 dims and
L2-normalized with last-token pooling (the tokenizer appends <|endoftext|>), the
model recorded in ``<category>_qwen3emb_1536.pt``.  It needs transformers >= 4.51
(the ``racer`` env), not the ``rrec`` training env.
"""

import argparse
import ast
import hashlib
import json
import os

import pandas as pd
import torch

VIEWS = ("full", "post_evidence", "feature")
EVIDENCE_STAGES = {"analyze the purchase pattern", "extract description evidence"}


def view_texts(steps):
    steps = [s for s in steps if s.get("text", "").strip()]
    if not steps:
        return None
    fmt = lambda s: f"{s['stage'].strip()}: {s['text'].strip()}"
    post = [s for s in steps if s["stage"].strip().lower() not in EVIDENCE_STAGES]
    return {
        "full": "\n".join(fmt(s) for s in steps),
        "post_evidence": "\n".join(fmt(s) for s in (post or steps)),
        "feature": steps[-1]["text"].strip(),
    }


def csv_history(value):
    return ast.literal_eval(value) if isinstance(value, str) else list(value)


def load_explanations(path, train, window):
    """Return {row_index: texts} after checking each record against the CSV row."""
    out, mismatched, skipped = {}, [], 0
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("split", "train") != "train":
                skipped += 1
                continue
            index = int(record["row_index"])
            texts = view_texts(record["cot"]["steps"])
            if texts is None:
                skipped += 1
                continue
            payload = [h["title"] for h in record["history_payload"]["history"]]
            expected = csv_history(train.iloc[index]["history_item_title"])[-window:]
            if [t.strip() for t in payload] != [t.strip() for t in expected]:
                mismatched.append(index)
                continue
            out[index] = texts
    return out, mismatched, skipped


class Encoder:
    """Plain-transformers text encoder: pool, truncate to ``dim`` (Matryoshka), L2-normalize.

    ``last`` pooling (Qwen3-Embedding) reads the final position under left padding.
    ``mean`` pooling is for small sentence-transformers models in CPU smoke tests.
    """

    def __init__(self, path, dim, pooling, max_length):
        from transformers import AutoModel, AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        self.model = AutoModel.from_pretrained(path, dtype=dtype).to(self.device).eval()
        limits = [getattr(self.model.config, "max_position_embeddings", max_length),
                  self.tokenizer.model_max_length]
        self.max_length = min([max_length] + [x for x in limits if x and x < 10**6])
        self.dim, self.pooling = dim, pooling

    @torch.no_grad()
    def __call__(self, texts, batch_size):
        from tqdm import tqdm
        out = []
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-bucketed batches
        for start in tqdm(range(0, len(order), batch_size), desc="encode", mininterval=30):
            index = order[start:start + batch_size]
            batch = self.tokenizer([texts[i] for i in index], padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt").to(self.device)
            hidden = self.model(**batch).last_hidden_state
            if self.pooling == "last":
                pooled = hidden[:, -1]
            else:
                weight = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * weight).sum(1) / weight.sum(1)
            pooled = torch.nn.functional.normalize(pooled.float()[:, :self.dim], dim=-1)
            out.append((index, pooled.cpu()))
        result = torch.empty(len(texts), min(self.dim, out[0][1].size(1)))
        for index, vectors in out:
            result[index] = vectors
        return result


def encode(model, texts, batch_size):
    return model(texts, batch_size)


def item_probe(model, embeddings, train, item_table, item_titles, batch_size, n=256, seed=0):
    """Cosine of the feature view to the true target item vs a random item."""
    table = torch.load(item_table, map_location="cpu", weights_only=False)
    items = torch.nn.functional.normalize(table["embeddings"].float(), dim=-1)
    rows = torch.nonzero(embeddings["mask"]).flatten()
    feature = embeddings["feature"][rows].float()
    target = torch.as_tensor(train["item_id"].to_numpy()[rows.numpy()])
    true_cos = (feature * items[target]).sum(-1)
    generator = torch.Generator().manual_seed(seed)
    random_cos = (feature * items[torch.randint(len(items), (len(rows),), generator=generator)]).sum(-1)
    rank = ((feature @ items.T) > true_cos[:, None]).sum(-1) + 1
    report = {
        "feature_target_cos": true_cos.mean().item(),
        "feature_random_item_cos": random_cos.mean().item(),
        "target_rank_median": rank.median().item(),
        "target_hit@10": (rank <= 10).float().mean().item(),
        "target_hit@100": (rank <= 100).float().mean().item(),
    }
    # Same-space check: re-encode some titles and compare with the stored table.
    sample = torch.randperm(len(item_titles), generator=generator)[:n]
    fresh = encode(model, [item_titles[i] for i in sample], batch_size)
    report["item_table_recode_cos"] = (fresh * items[sample]).sum(-1).mean().item()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explanations", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--dim", type=int, default=1536)
    parser.add_argument("--window", type=int, default=10, help="history window used by the generator")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--pooling", choices=("last", "mean"), default="last",
                        help="last = Qwen3-Embedding; mean = sentence-transformers smoke models")
    parser.add_argument("--limit", type=int, default=0, help="embed only the first N explained rows (smoke test)")
    parser.add_argument("--item_table", default="", help="<category>_qwen3emb_<dim>.pt, for the target probe")
    parser.add_argument("--item_info", default="", help="<category>_*.txt, item titles for the same-space check")
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    explained, mismatched, skipped = load_explanations(args.explanations, train, args.window)
    print(f"train rows {len(train)}; explained {len(explained)}; skipped {skipped}; "
          f"history mismatch {len(mismatched)} (first: {mismatched[:5]})")
    if mismatched:
        raise SystemExit("explanations are not aligned with the train CSV; refusing to write")
    rows = sorted(explained)[: args.limit or None]

    model = Encoder(args.model, args.dim, args.pooling, args.max_seq_length)

    mask = torch.zeros(len(train), dtype=torch.bool)
    mask[rows] = True
    result = {"mask": mask}
    for view in VIEWS:
        # Explanations are documents: Qwen3-Embedding uses no instruction prompt for them.
        vectors = encode(model, [explained[i][view] for i in rows], args.batch_size)
        table = torch.zeros(len(train), vectors.size(1), dtype=torch.float16)
        table[rows] = vectors.half()
        result[view] = table
        print(f"{view}: {tuple(vectors.shape)}")

    with open(args.explanations, "rb") as handle:
        source_sha = hashlib.sha256(handle.read()).hexdigest()
    result.update({
        "views": list(VIEWS), "model": args.model, "dim": args.dim, "normalized": True,
        "explanations": os.path.abspath(args.explanations), "explanations_sha256": source_sha,
        "train_csv": os.path.abspath(args.train_csv), "n_rows": len(train), "n_explained": len(rows),
    })
    if args.item_table and args.item_info:
        titles = [line.split("\t")[0].strip() for line in open(args.item_info)]
        result["probe"] = item_probe(model, result, train, args.item_table, titles, args.batch_size)
        print("probe:", json.dumps(result["probe"], indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(result, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
