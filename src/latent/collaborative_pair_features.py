"""Training-only title and collaborative signals for contrastive pairs."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def build_pair_features(
    train_file,
    title_dim=64,
    cf_topk=32,
    cf_min_common=2,
    cf_min_cosine=0.1,
    seed=0,
):
    frame = pd.read_csv(train_file, usecols=["user_id", "item_id", "item_title"])
    items = frame.drop_duplicates("item_id").sort_values("item_id").reset_index(drop=True)
    item_to_index = {int(item_id): index for index, item_id in enumerate(items.item_id)}

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, max_features=20_000, stop_words="english"
    ).fit_transform(items.item_title.fillna("").astype(str))
    dimensions = min(title_dim, min(tfidf.shape) - 1)
    if dimensions < 1:
        raise ValueError("Not enough title vocabulary for semantic pair features")
    title_vectors = normalize(
        TruncatedSVD(n_components=dimensions, random_state=seed).fit_transform(tfidf)
    ).astype(np.float32)

    pairs = frame[["user_id", "item_id"]].drop_duplicates()
    users = pd.Index(pairs.user_id.unique())
    row = pairs.item_id.map(item_to_index).to_numpy()
    column = users.get_indexer(pairs.user_id)
    interactions = csr_matrix(
        (np.ones(len(pairs), dtype=np.float32), (row, column)),
        shape=(len(items), len(users)),
    )
    frequency = np.asarray(interactions.sum(axis=1)).ravel()
    cooccurrence = (interactions @ interactions.T).tocsr()
    neighbors = np.full((len(items), cf_topk), -1, dtype=np.int64)
    for item_index in range(len(items)):
        start, end = cooccurrence.indptr[item_index:item_index + 2]
        candidates = cooccurrence.indices[start:end]
        common = cooccurrence.data[start:end]
        similarity = common / np.sqrt(frequency[item_index] * frequency[candidates])
        eligible = (
            (candidates != item_index)
            & (common >= cf_min_common)
            & (similarity >= cf_min_cosine)
        )
        candidates = candidates[eligible]
        similarity = similarity[eligible]
        ordered = np.lexsort((candidates, -similarity))[:cf_topk]
        neighbors[item_index, :len(ordered)] = candidates[ordered]

    report = {
        "train_rows": len(frame),
        "items": len(items),
        "users": len(users),
        "title_feature_dim": dimensions,
        "cf_items_with_neighbors": int((neighbors[:, 0] >= 0).sum()),
        "cf_directed_edges": int((neighbors >= 0).sum()),
        "cf_topk": cf_topk,
        "cf_min_common": cf_min_common,
        "cf_min_cosine": cf_min_cosine,
    }
    return item_to_index, title_vectors, neighbors, report
