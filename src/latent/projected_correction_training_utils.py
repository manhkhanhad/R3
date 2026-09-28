"""Training helpers owned by the projected Attempt-10 correction experiment."""

import os

import pandas as pd
import torch
import torch.nn.functional as F
import transformers

from projected_neighbor_refinement import ProjectedNeighborRefinementLayer
from reasoning_dataset import LatentRDataset


CATEGORY_TEXT = {
    "CDs_and_Vinyl": "musics",
    "Toys_and_Games": "toys and games",
    "Musical_Instruments": "music instruments",
}


class TimedDataset(LatentRDataset):
    def pre(self, idx):
        item = super().pre(idx)
        if not self.test:
            item["query_time"] = int(self.data.iloc[idx]["timestamp"])
        return item


class NeighborLogCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None or model is None or not isinstance(model.attention, ProjectedNeighborRefinementLayer):
            return
        for attr in ("last_attention_entropy", "last_refinement_ratio", "last_valid_fraction"):
            value = getattr(model.attention, attr, None)
            if value is not None:
                logs[attr.replace("last_", "neighbor_")] = value.item()


class TwoRateTrainer(transformers.Trainer):
    def __init__(self, *args, module_lr=5e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.module_lr = module_lr

    def create_optimizer(self):
        if self.optimizer is None:
            fresh = {id(p) for name, p in self.model.named_parameters()
                     if name.startswith("attention.") and not name.startswith("attention.base_attention.")}
            parameters = [p for p in self.model.parameters() if p.requires_grad]
            groups = [{"params": [p for p in parameters if id(p) not in fresh], "lr": self.args.learning_rate}]
            groups.append({"params": [p for p in parameters if id(p) in fresh], "lr": self.module_lr})
            self.optimizer = torch.optim.AdamW(groups, weight_decay=self.args.weight_decay)
        return self.optimizer


def build_bank(layer: ProjectedNeighborRefinementLayer, bank_dir: str,
               train_file: str, item_emb_file: str) -> None:
    r = torch.load(os.path.join(bank_dir, "thoughts.pt"), map_location="cpu", weights_only=True).float()
    meta = pd.read_csv(os.path.join(bank_dir, "metadata.csv"))
    frame = pd.read_csv(train_file)
    if len(frame) != len(r) or (meta.target_item.values != frame.item_id.values).any():
        raise ValueError("bank thoughts are not aligned with the train CSV")
    mean = r.mean(0)
    item_emb = torch.load(item_emb_file, map_location="cpu", weights_only=False)["embeddings"]
    device = layer.bank_key.device
    layer.bank_mean.copy_(mean.to(device))
    layer.bank_key.copy_(F.normalize(r - mean, dim=-1).half().to(device))
    layer.bank_item.copy_(torch.as_tensor(frame.item_id.values, dtype=torch.int64).to(device))
    layer.bank_time.copy_(torch.as_tensor(frame.timestamp.values, dtype=torch.int64).to(device))
    layer.item_emb.copy_(item_emb.half().to(device))
    print(f"bank: {len(r)} rows, items={len(item_emb)}, "
          f"time range={frame.timestamp.min()}..{frame.timestamp.max()}", flush=True)
