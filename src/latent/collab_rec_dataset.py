"""Supports for CollabRec on R3 (see collab_rec_latent.py).

The support artifacts come from the IntuRec repo (``<IntuRec>/data/<dataset>/collab``);
R3's and IntuRec's train/valid/test CSVs for CDs_and_Vinyl and Toys_and_Games are
byte-identical, so row indices carry over.

Support sources (``support_source``):
  peer           train rows' LLM rationales (``rationales_facet_train_<tag>.jsonl``), rows from
                 ``stage2/support_rows_<split>.npy`` (latent-hybrid retrieval, target-balanced).
                 emb mode uses ``stage2/rationale_emb_train.npy`` instead of the text.
  motive         motive prototypes predicted by the motive retriever
                 (``motive_corpus/stage2/support_texts.json`` + ``support_rows_<split>.npy``).
  oracle_motive  analysis only: each row's own 1-2 ground-truth motive prototypes (k = 2, one
                 motive repeated, rows without motives -> "No specific motive."). Uses the target.
  intuition      IntuRec stage 2 (stage 1 bypassed): each row's 15 "intuition" items, the stage-1
                 beam outputs stored in ``<IntuRec>/data/<dataset>/<split>_extra/extra.json``
                 (``intuition_list_15``). The table is the item catalogue (info file), every entry
                 the quoted title ``"<title>"``, so intuitions, random items and the target all
                 index the same table (for IntuRec's intuition loss).
  none           no supports (k = 0).
"""

import glob
import json
import os

import numpy as np

from reasoning_dataset import LatentRDataset, Tokenizer

INTUREC_ROOT = "/common/home/users/m/mk.ngohuu.2024/research/IntuRec"
SOURCES = ("peer", "motive", "oracle_motive", "intuition", "none")
TARGET_FORMATS = ("r3", "end_predict")


def default_collab_dir(dataset):
    return os.path.join(INTUREC_ROOT, "data", dataset, "collab")


def tokenize_texts(tokenizer, texts, max_len):
    """Left-padded token ids and mask, [len(texts), max_len]."""
    tok = Tokenizer(tokenizer)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = np.full((len(texts), max_len), pad, dtype=np.int64)
    mask = np.zeros((len(texts), max_len), dtype=bool)
    for i, text in enumerate(texts):
        t = tok.encode(text, bos=False, eos=False)[:max_len]
        ids[i, max_len - len(t):] = t
        mask[i, max_len - len(t):] = True
    return ids, mask


def load_rationales(path):
    reasons = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "error" not in rec:
                reasons[rec["row"]] = rec["reason"]
    assert len(reasons) == max(reasons) + 1, f"{path} must cover every row"
    return [reasons[r] for r in range(len(reasons))]


def motive_texts(collab_dir):
    with open(os.path.join(collab_dir, "motive_corpus", "prototypes.json"), encoding="utf-8") as f:
        return [f"{p['motive']}: {p['reason']}" for p in json.load(f)] + ["No specific motive."]


def inturec_data_dir(collab_dir):
    return os.path.dirname(os.path.normpath(collab_dir))  # <IntuRec>/data/<dataset>


def item_titles(collab_dir):
    """Catalogue titles in info-file order, quoted as IntuRec writes them ('"title"')."""
    (info,) = glob.glob(os.path.join(inturec_data_dir(collab_dir), "info", "*.txt"))
    with open(info, encoding="utf-8") as f:
        return ['"' + line.split("\t")[0].strip('"').strip(" ").strip("\n").strip('"') + '"' for line in f]


def title_index(collab_dir):
    index = {}
    for i, t in enumerate(item_titles(collab_dir)):
        index.setdefault(t, i)  # duplicate titles: first id (same text either way)
    return index


def load_extra(collab_dir, split):
    with open(os.path.join(inturec_data_dir(collab_dir), f"{split}_extra", "extra.json"), encoding="utf-8") as f:
        return json.load(f)


def intuition_aux_rows(spec, split, seed=0):
    """IntuRec's intuition-loss inputs: random item ids [n, k] (fixed per row) and target ids [n]."""
    extra = load_extra(spec["collab_dir"], split)
    k = int(spec["support_k"])
    n_items, rng = len(item_titles(spec["collab_dir"])), np.random.default_rng(seed)
    random_ids = np.stack([rng.choice(n_items, k, replace=False) for _ in range(len(extra))])
    # info line i is item id i (one title contains a tab, so match by id, not by title)
    target_ids = np.array([int(r["item_id"]) for r in extra], dtype=np.int64)
    assert target_ids.max() < n_items
    return random_ids.astype(np.int64), target_ids


def build_support_table(spec, tokenizer=None):
    """Table the support ids index into: (tokens, mask) for text mode, (vectors, None) for emb."""
    source, mode, collab_dir = spec["support_source"], spec["support_mode"], spec["collab_dir"]
    if source == "none":
        return None, None
    if mode == "emb":
        if source != "peer":
            raise ValueError("emb mode exists only for peer rationales")
        return np.load(os.path.join(collab_dir, "stage2", "rationale_emb_train.npy")).astype(np.float32), None
    if source == "peer":
        texts = load_rationales(os.path.join(collab_dir, f"rationales_facet_train_{spec['rationale_tag']}.jsonl"))
    elif source == "motive":
        with open(os.path.join(collab_dir, "motive_corpus", "stage2", "support_texts.json"), encoding="utf-8") as f:
            texts = json.load(f)
    elif source == "intuition":
        texts = item_titles(collab_dir)
    else:  # oracle_motive
        texts = motive_texts(collab_dir)
    return tokenize_texts(tokenizer, texts, int(spec["max_support_len"]))


def load_support_rows(spec, split):
    """[n_rows_of_split, k] support ids (indices into the support table), or None."""
    source, collab_dir, k = spec["support_source"], spec["collab_dir"], int(spec["support_k"])
    if source == "none" or k == 0:
        return None
    if source == "peer":
        return np.load(os.path.join(collab_dir, "stage2", f"support_rows_{split}.npy"))[:, :k]
    if source == "motive":
        return np.load(os.path.join(collab_dir, "motive_corpus", "stage2", f"support_rows_{split}.npy"))[:, :k]
    if source == "intuition":
        index = title_index(collab_dir)
        lists = [r["intuition_list_15"] for r in load_extra(collab_dir, split)]
        return np.array([[index[t] for t in lst[:k]] for lst in lists], dtype=np.int64)
    no_motive = len(motive_texts(collab_dir)) - 1
    with open(os.path.join(collab_dir, "motive_corpus", f"row_motives_{split}.json"), encoding="utf-8") as f:
        motives = json.load(f)
    return np.array([(m + m)[:2] if m else [no_motive] * 2 for m in motives], dtype=np.int64)


class CollabLatentRDataset(LatentRDataset):
    """R3's dataset plus ``support_ids`` per example.

    Rows are matched to the support files by their index in the original split CSV:
    the ``Unnamed: 0`` column written by src/utils/split.py when present, else the
    DataFrame index (which ``sample`` preserves).
    """

    def __init__(self, *args, support_rows=None, target_format="r3", own_rows=False, intuition_aux=None, **kwargs):
        if target_format not in TARGET_FORMATS:
            raise ValueError(f"target_format must be one of {TARGET_FORMATS}")
        self.support_rows = support_rows
        self.target_format = target_format
        self.own_rows = own_rows  # add "own_row" (original CSV row) for the rationale InfoNCE
        self.intuition_aux = intuition_aux  # (random_ids [n, k], target_ids [n]) for IntuRec's intuition loss
        super().__init__(*args, **kwargs)

    def get_inputs(self):
        # end_predict (IntuRec): the target is '"title"' + <|EndPredict|> instead of '"title"\n' + <|endoftext|>;
        # LatentRDataset appends Tokenizer.eos_id after the target, so point it at <|EndPredict|>.
        if self.target_format == "end_predict":
            self.tokenizer.eos_id = self.tokenizer.tokenizer.convert_tokens_to_ids("<|EndPredict|>")
        super().get_inputs()

    def get_history(self, row):
        out = super().get_history(row)
        if self.target_format == "end_predict":
            out["output"] = out["output"].rstrip("\n")
        return out

    def original_row(self, idx):
        row = self.data.iloc[idx]
        return int(row["Unnamed: 0"]) if "Unnamed: 0" in self.data.columns else int(self.data.index[idx])

    def pre(self, idx):
        example = super().pre(idx)
        if self.support_rows is not None:
            example["support_ids"] = [int(x) for x in self.support_rows[self.original_row(idx)]]
        if self.own_rows:
            example["own_row"] = self.original_row(idx)
        if self.intuition_aux is not None:
            random_ids, target_ids = self.intuition_aux
            example["random_ids"] = [int(x) for x in random_ids[self.original_row(idx)]]
            example["target_id"] = int(target_ids[self.original_row(idx)])
        return example
