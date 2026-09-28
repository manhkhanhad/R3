"""Plain-R3 dataset with stable user, target-item, and sample identifiers."""

import hashlib

from reasoning_dataset import LatentRDataset


def stable_hash(value) -> int:
    """Return a deterministic positive int64-compatible identifier."""
    return int(hashlib.md5(str(value).encode("utf-8")).hexdigest()[:15], 16)


class CollaborativeContrastiveDataset(LatentRDataset):
    def __init__(self, *args, item_to_index=None, title_vectors=None, **kwargs):
        self.item_to_index = item_to_index
        self.title_vectors = title_vectors
        super().__init__(*args, **kwargs)

    def pre(self, idx):
        item = super().pre(idx)
        if not self.test:
            row = self.data.iloc[idx]
            item["user_hash"] = stable_hash(row["user_id"])
            item["target_hash"] = stable_hash(row["item_id"])
            # Exclude this sample's stale queue entries on later epochs.
            item["sample_hash"] = int(idx)
            if self.title_vectors is not None:
                item_index = self.item_to_index.get(int(row["item_id"]), -1)
                item["target_item_index"] = item_index
                item["target_title_features"] = (
                    self.title_vectors[item_index].tolist()
                    if item_index >= 0 else [0.0] * self.title_vectors.shape[1]
                )
        return item
