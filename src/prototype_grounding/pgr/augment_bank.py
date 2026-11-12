"""Re-embed every rationale in the RALR bank -- all 514, not just the 350 that are
missing an embedding -- so the whole bank comes from one uniform, known pipeline.

Embedding model: Alibaba-NLP/gte-large-en-v1.5 (1024-dim), chosen deliberately --
*not* RALR's own `DEFAULT_EMBED_MODEL` (sentence-transformers/all-mpnet-base-v2,
768-dim, in research/LLMReasoningPrototype/pipeline/embedding_client.py). We don't
need to match RALR's original clustering-time embedding space, only to be internally
self-consistent for our own retrieval's cosine comparisons -- so the model choice here
is a free one, gte-large-en-v1.5 was picked as (presumably) a stronger embedder, not
for continuity with RALR. `bank.py` derives `bank_dim` from the data rather than
hardcoding 768 or 1024 anywhere, so this choice doesn't need to propagate as a
constant through the rest of the pipeline -- but if you change EMBED_MODEL again,
regenerate this file from scratch rather than trying to merge with an older one:
mixing rows from two different models in the same JSONL is exactly the silent
cross-space mismatch this whole exercise exists to avoid (`bank.py`'s loader now
raises if it ever sees more than one embedding dimension in a single file).

Why re-embed all 514 rather than just filling in the 350 gaps: whatever originally
produced the 164 existing mpnet embeddings, we're replacing them anyway by switching
models, so there's no "existing embeddings to preserve" question left -- every row
comes from this one run, uniformly.

All 514 rows have real rationale text (the 350 discarded ones failed RALR's own
quality_score gate -- checked: 0.0-0.53, mean 0.337, vs. the >=0.6 threshold "ok" rows
pass -- not missing text), so every row gets embedded and tagged with `tier`
("verified" / "unverified") based on its original `status`. Folding the unverified tier
into the retrievable pool is a real trust shift, not a free lunch: RALR's gate was a
tested claim (held-out ranking accuracy in `verification`); the quality-weighted MMR
and the quality_score feature fed into proj_up (Mechanism step 3) that now have to do
that suppression job instead are untested. `tier` is kept on every row precisely so
retrieval diagnostics can always tell which side of that line a given prototype came
from -- see bank.py's PrototypeBank.tiers.

Run (CPU is fine for 514 short rationales, but --device cuda works if one's free):
    python3 augment_bank.py
    python3 augment_bank.py --device cuda --batch-size 64
    python3 augment_bank.py --raw-bank /path/to/other_bank.jsonl --out /path/to/out.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_RAW_BANK = Path(
    "/common/home/users/m/mk.ngohuu.2024/research/LLMReasoningPrototype/data/evidence_bank/"
    "CDs_and_Vinyl_5_2015-10-2018-11_train_evidence_bank.jsonl"
)
# Local to this package, not the shared analysis/ cache -- keeps the augmented artifact
# colocated with the code that produces/consumes it and avoids writing through a
# symlink into the shared checkout from inside a worktree-isolated session.
DEFAULT_OUT_BANK = Path(__file__).resolve().parent / "data" / (
    "CDs_and_Vinyl_5_2015-10-2018-11_train_evidence_bank.augmented.jsonl"
)
EMBED_MODEL = "Alibaba-NLP/gte-large-en-v1.5"  # 1024-dim; deliberately not RALR's mpnet-base-v2 -- see module docstring


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-bank", type=Path, default=DEFAULT_RAW_BANK,
                    help="Source JSONL with status=ok/discarded_low_quality rows (default: CDs_and_Vinyl bank)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_BANK,
                    help="Where to write the augmented JSONL (all rows embedded + tagged with tier)")
    p.add_argument("--device", default=None,
                    help="Passed to SentenceTransformer, e.g. 'cuda', 'cuda:0', 'cpu'. "
                         "Default: let sentence-transformers auto-detect.")
    p.add_argument("--batch-size", type=int, default=32,
                    help="Encoding batch size (350 rows total for the default bank, so this barely matters "
                         "at CPU scale; bump it up if running on GPU with a much larger bank).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rows = []
    with args.raw_bank.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n_had_embedding = sum(1 for r in rows if r.get("embedding") is not None)
    print(f"{len(rows)} total rows, {n_had_embedding} had a pre-existing embedding "
          f"(discarded regardless -- re-embedding all {len(rows)} from {EMBED_MODEL} for consistency)")

    from sentence_transformers import SentenceTransformer
    import torch
    print(f"torch cuda available: {torch.cuda.is_available()}, requested device: {args.device!r}")
    model = SentenceTransformer(EMBED_MODEL, device=args.device, trust_remote_code=True)
    texts = [r["rationale"] for r in rows]
    vectors = model.encode(
        texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True,
    )
    expected_dim = model.get_sentence_embedding_dimension()
    assert vectors.shape[1] == expected_dim, (
        f"{EMBED_MODEL} reports dim={expected_dim} but encode() returned shape {vectors.shape} "
        f"-- something is wrong with the model/library, do not write this file"
    )
    print(f"embedding dim: {expected_dim}")
    for row, vec in zip(rows, vectors):
        row["embedding"] = vec.tolist()

    for r in rows:
        r["tier"] = "verified" if r.get("status") == "ok" else "unverified"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_verified = sum(1 for r in rows if r["tier"] == "verified")
    print(f"wrote {len(rows)} rows ({n_verified} verified, {len(rows) - n_verified} "
          f"unverified), all with embeddings, to {args.out}")


if __name__ == "__main__":
    main()
