"""Plain-R3 dataset carrying frozen SASRec sequence/target features (Attempt 11)."""

from collaborative_contrastive_dataset import CollaborativeContrastiveDataset


class SASRecCollaborativeDataset(CollaborativeContrastiveDataset):
    def __init__(self, *args, sasrec_sequence_vectors=None, sasrec_targets=None, **kwargs):
        # Both arrays are aligned to the rows of the full CSV, so lookups use
        # the frame's original row label and remain valid after ``sample``.
        self.sasrec_sequence_vectors = sasrec_sequence_vectors
        self.sasrec_targets = sasrec_targets
        super().__init__(*args, **kwargs)

    def pre(self, idx):
        item = super().pre(idx)
        if not self.test:
            row = self.data.index[idx]
            item["sasrec_sequence"] = self.sasrec_sequence_vectors[row].tolist()
            item["sasrec_item"] = int(self.sasrec_targets[row])
        return item
