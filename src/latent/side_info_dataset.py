"""Datasets for the decoder-insensitivity side-information tests."""

from reasoning_dataset import LatentRDataset

CATEGORY_TEXT_LINE = "The category of the item to recommend is: {path}"


class SideInfoLatentRDataset(LatentRDataset):
    """Adds ``side_index``: the target item id, or a row index plus an offset."""

    def __init__(self, *args, side_key="item_id", row_offset=0, **kwargs):
        self.side_key = side_key
        self.row_offset = row_offset
        super().__init__(*args, **kwargs)

    def pre(self, idx):
        example = super().pre(idx)
        if self.side_key == "item_id":
            example["side_index"] = int(self.data.iloc[idx]["item_id"])
        else:
            example["side_index"] = self.row_offset + idx
        return example


class CategoryTextLatentRDataset(LatentRDataset):
    """Text positive control: the target's category path is written into the prompt."""

    def __init__(self, *args, category_paths=None, **kwargs):
        self.category_paths = category_paths
        super().__init__(*args, **kwargs)

    def get_history(self, row):
        out = super().get_history(row)
        path = self.category_paths.get(int(row["item_id"]), "unknown")
        out["input"] += "\n" + CATEGORY_TEXT_LINE.format(path=path)
        return out
