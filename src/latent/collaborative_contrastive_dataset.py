"""Plain-R3 dataset with stable user, target-item, and sample identifiers."""

import hashlib

from reasoning_dataset import LatentRDataset


def stable_hash(value) -> int:
    """Return a deterministic positive int64-compatible identifier."""
    return int(hashlib.md5(str(value).encode("utf-8")).hexdigest()[:15], 16)


class CollaborativeContrastiveDataset(LatentRDataset):
    def pre(self, idx):
        item = super().pre(idx)
        if not self.test:
            row = self.data.iloc[idx]
            item["user_hash"] = stable_hash(row["user_id"])
            item["target_hash"] = stable_hash(row["item_id"])
            # Exclude this sample's stale queue entries on later epochs.
            item["sample_hash"] = int(idx)
        return item
