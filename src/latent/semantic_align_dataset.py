"""LatentRDataset subclass that also exposes user/target identity per example.

Additive only: does not modify reasoning_dataset.py. Needed by
semantic_alignment_attention.py's memory-bank alignment loss (Attempt 8),
which needs to know which other training rows share a user_id or target
item_id. user_id is a string (ASIN-like); item_id is already a small integer,
used directly. Both survive DataCollatorForSeq2Seq/HFDataset unchanged as
plain int64 columns (verified: extra scalar columns pass through the
collator as [batch] tensors without special handling needed).
"""

import hashlib

from reasoning_dataset import LatentRDataset


def _stable_hash(s: str) -> int:
    # deterministic across processes/runs, unlike Python's built-in hash()
    return int(hashlib.md5(str(s).encode()).hexdigest()[:15], 16)


class SemanticAlignLatentRDataset(LatentRDataset):
    def pre(self, idx):
        item = super().pre(idx)
        if not self.test:
            row = self.data.iloc[idx]
            item["user_hash"] = _stable_hash(row["user_id"])
            item["target_hash"] = int(row["item_id"])
        return item
