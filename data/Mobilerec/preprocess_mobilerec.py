"""Convert MobileRec (recmeapp/mobilerec) into the R3 / Amazon-Toys data format.

Mirrors data/process.py (used for Toys_and_Games_5_2017-1-2018-11):

    <out>/train/<name>.csv      80% of next-item rows, global chronological split
    <out>/valid/<name>.csv      next 10%
    <out>/test/<name>.csv       last 10%
    <out>/info/<name>.txt       "<title>\t<id>" per item; line number == item_id
    <out>/info/<name>.item_meta.json   {app_package: {"title", "description"}}

CSV columns (lists are python reprs, eval()-able, like the Toys CSVs):
    user_id, item_asins, item_asin, history_item_id, item_id,
    history_item_title, item_title, history_rating, rating,
    history_timestamp, timestamp

MobileRec specifics:
  * 19.3M interactions / 700k users, far bigger than Toys, so a time window
    and a seeded random user sample are applied before iterative K-core.
    Defaults (2021-01..2022-01, 10k users, K=5) give ~3.4k users,
    ~4.9k items, ~64k rows, about the size of Toys.
  * item "asin" = app_package, title = app_name. Evaluation matches
    predictions by title, so duplicate app names among the retained items
    are disambiguated with " - <developer>", then " (<package>)".
  * Repeated (user, app) reviews (0.8%) are dropped by default (keep first);
    pass --dedup=False to keep them as Amazon processing does.
  * Timestamps have day resolution. Same-day order follows the raw file,
    which is chronologically sorted.

Usage (from repo root, rrec env):
    python data/Mobilerec/preprocess_mobilerec.py
    python data/Mobilerec/preprocess_mobilerec.py --n_users 20000 --st_year 2020
"""

import json
import os
import random
import re

import fire
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_COLUMNS = [
    "user_id", "item_asins", "item_asin", "history_item_id", "item_id",
    "history_item_title", "item_title", "history_rating", "rating",
    "history_timestamp", "timestamp",
]


def clean_text(text):
    text = "" if pd.isna(text) else str(text)
    text = text.replace("&quot;", '"').replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip(' "')


def kcore(df, K):
    while K > 0:
        u = df.groupby("uid")["uid"].transform("size")
        i = df.groupby("app_package")["app_package"].transform("size")
        keep = (u >= K) & (i >= K)
        if keep.all():
            break
        df = df[keep]
    return df


def unique_titles(meta):
    """app_package -> title, unique across the given (retained) items."""
    meta = meta.copy()
    meta["title"] = meta["app_name"].map(clean_text)
    dup = meta["title"].duplicated(keep=False)
    meta.loc[dup, "title"] = meta.loc[dup, "title"] + " - " + meta.loc[dup, "developer_name"].map(clean_text)
    dup = meta["title"].duplicated(keep=False)
    meta.loc[dup, "title"] = meta.loc[dup, "title"] + " (" + meta.loc[dup, "app_package"] + ")"
    assert not meta["title"].duplicated().any()
    return dict(zip(meta["app_package"], meta["title"]))


def main(
    K: int = 5,
    st_year: int = 2021,
    st_month: int = 1,
    ed_year: int = 2022,
    ed_month: int = 1,
    n_users: int = 10000,
    window_size: int = 10,
    dedup: bool = True,
    seed: int = 42,
    raw_dir: str = HERE,
    output_dir: str = HERE,
    name: str = "",
):
    """Window [st, ed) -> sample n_users (0 = all) -> K-core -> splits."""
    start = pd.Timestamp(year=st_year, month=st_month, day=1).timestamp()
    end = pd.Timestamp(year=ed_year, month=ed_month, day=1).timestamp()
    name = name or f"Mobilerec_{K}_{st_year}-{st_month}-{ed_year}-{ed_month}"

    df = pd.read_csv(
        os.path.join(raw_dir, "interactions", "mobilerec_final.csv"),
        usecols=["uid", "app_package", "rating", "unix_timestamp"],
        engine="pyarrow",
    )
    print(f"raw: {len(df)} interactions, {df.uid.nunique()} users, {df.app_package.nunique()} apps")
    df["order"] = np.arange(len(df))
    df["timestamp"] = df["unix_timestamp"].astype("int64")
    df = df[(df.timestamp >= start) & (df.timestamp < end)]
    if dedup:
        df = df.drop_duplicates(["uid", "app_package"], keep="first")

    meta = pd.read_csv(os.path.join(raw_dir, "app_meta", "app_meta.csv"))
    meta = meta[meta["app_name"].map(clean_text).str.len() > 1]
    df = df[df.app_package.isin(set(meta.app_package))]
    print(f"window: {len(df)} interactions, {df.uid.nunique()} users, {df.app_package.nunique()} apps")

    if n_users:
        users = np.sort(df.uid.unique())
        rng = np.random.default_rng(seed)
        sampled = set(rng.choice(users, min(n_users, len(users)), replace=False))
        df = df[df.uid.isin(sampled)]
    df = kcore(df, K)
    n_u, n_i = df.uid.nunique(), df.app_package.nunique()
    print(f"after sample + {K}-core: users={n_u}, items={n_i}, interactions={len(df)}, "
          f"density={len(df) / (n_u * n_i):.5f}")

    # item ids: shuffled with seed, as in process.py; info line number == id
    meta = meta[meta.app_package.isin(set(df.app_package))]
    asin2title = unique_titles(meta)
    items = sorted(asin2title)
    random.seed(seed)
    random.shuffle(items)
    asin2id = {a: i for i, a in enumerate(items)}

    os.makedirs(os.path.join(output_dir, "info"), exist_ok=True)
    with open(os.path.join(output_dir, "info", f"{name}.txt"), "w") as f:
        for a in items:
            f.write(f"{asin2title[a]}\t{asin2id[a]}\n")
    meta_idx = meta.set_index("app_package")
    item_meta = {}
    for a in items:
        m = meta_idx.loc[a]
        desc = [f"Category: {clean_text(m.app_category)}. Developer: {clean_text(m.developer_name)}."]
        if clean_text(m.description):
            desc.append(str(m.description).strip())
        item_meta[a] = {"title": asin2title[a], "description": desc}
    with open(os.path.join(output_dir, "info", f"{name}.item_meta.json"), "w") as f:
        json.dump(item_meta, f)

    # next-item rows per user (users in first-appearance order, like process.py)
    df = df.sort_values(["timestamp", "order"], kind="stable")
    rows = []
    for user, g in df.groupby("uid", sort=False):
        a = g.app_package.tolist()
        ids = [asin2id[x] for x in a]
        titles = [asin2title[x] for x in a]
        r = [float(x) for x in g.rating]
        t = g.timestamp.tolist()
        for i in range(1, len(a)):
            s = max(i - window_size, 0)
            rows.append([user, a[s:i], a[i], ids[s:i], ids[i], titles[s:i], titles[i],
                         r[s:i], r[i], t[s:i], t[i]])
    rows.sort(key=lambda x: int(x[-1]))  # stable global chronological order

    n = len(rows)
    splits = {"train": rows[: int(n * 0.8)], "valid": rows[int(n * 0.8): int(n * 0.9)],
              "test": rows[int(n * 0.9):]}
    for split, part in splits.items():
        out = pd.DataFrame(part, columns=CSV_COLUMNS)
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
        path = os.path.join(output_dir, split, f"{name}.csv")
        out.to_csv(path, index=False)  # list cells -> python repr, eval()-able
        print(f"[{split}] rows={len(out)} users={out.user_id.nunique()} "
              f"target_items={out.item_id.nunique()} -> {path}")
    print(f"items: {len(items)} -> {os.path.join(output_dir, 'info', name + '.txt')}")


if __name__ == "__main__":
    fire.Fire(main)
