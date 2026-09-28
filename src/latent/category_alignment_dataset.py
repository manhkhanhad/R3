"""Expose each target item's ID to the category alignment model."""

from reasoning_dataset import LatentRDataset


class CategoryAlignedLatentRDataset(LatentRDataset):
    def pre(self, idx):
        example = super().pre(idx)
        example["category_item_id"] = int(self.data.iloc[idx]["item_id"])
        return example
