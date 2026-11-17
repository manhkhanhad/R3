"""Convert Goodreads (UCSD Book Graph) into the R3 / Amazon-Toys data format.

Mirrors data/process.py and data/Mobilerec/preprocess_mobilerec.py:

    <out>/train/<name>.csv      80% of next-item rows, global chronological split
    <out>/valid/<name>.csv      next 10%
    <out>/test/<name>.csv       last 10%
    <out>/info/<name>.txt       "<title>\t<id>" per item; line number == item_id
    <out>/info/<name>.item_meta.json   {work_id: {"title", "description"}}

Goodreads specifics:
  * Interactions = rows of goodreads_interactions.csv with is_reviewed == 1
    (16.2M), with ids mapped through user_id_map.csv / book_id_map.csv. The
    CSV has no time, so the interaction time is the review time, joined on
    (user_id, book_id) from goodreads_reviews_dedup.json.gz ("date_added",
    all 15.7M reviews) if that file is present, otherwise from
    goodreads_reviews_spoiler.json.gz ("timestamp", day resolution; only
    1.38M reviews). Reviewed interactions with no review time are dropped.
  * Items are works, not editions: book_id -> work_id, so reviews of
    different editions of one book become one item (item_asin = work_id).
    Duplicate (user, work) pairs keep only the first interaction in time,
    deduplicated over the full history before windowing.
  * Window + seeded user sample + iterative K-core, as for MobileRec.
    Defaults (2017-01..2017-12, 26k users, K=5) with the full review file
    give ~6.4k users, ~8k works, ~95k rows (~1.5x Toys).
  * rating 0 means "reviewed without a star rating"; kept as 0.0.
  * Ties at the same time keep the review file's order.
  * Each work is represented by its most-interacted edition in the window.
    Titles come from that edition in goodreads_books.json.gz ("title", which
    includes the series tag, e.g. "... (Harry Potter, #1)"). Remaining
    duplicate titles (distinct works) get " (<year>)", then " [<work_id>]".
    Author names are not in the download (no goodreads_book_authors file).

Usage (from repo root, rrec env):
    python data/Goodreads/preprocess_goodreads.py
    python data/Goodreads/preprocess_goodreads.py --n_users 10000 --st_year 2016
"""

import gzip
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
BOOK_ID_RE = re.compile(r'"book_id": "(\d+)"')


def clean_text(text):
    text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    text = text.replace("&quot;", '"').replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip(' "')


def kcore(df, K):
    while K > 0:
        u = df.groupby("user_id")["user_id"].transform("size")
        i = df.groupby("book_id")["book_id"].transform("size")
        keep = (u >= K) & (i >= K)
        if keep.all():
            break
        df = df[keep]
    return df


def load_reviewed_interactions(raw_dir):
    import pyarrow.compute as pc
    import pyarrow.csv as pcsv
    types = {k: "int32" for k in ("user_id", "book_id")} | {k: "int8" for k in ("is_read", "rating", "is_reviewed")}
    tb = pcsv.read_csv(os.path.join(raw_dir, "goodreads_interactions.csv"),
                       convert_options=pcsv.ConvertOptions(column_types=types))
    df = tb.filter(pc.equal(tb["is_reviewed"], 1)).select(["user_id", "book_id", "rating"]).to_pandas()
    um = pd.read_csv(os.path.join(raw_dir, "user_id_map.csv"), dtype={"user_id": str})
    bm = pd.read_csv(os.path.join(raw_dir, "book_id_map.csv"), dtype={"book_id": str})
    df["user_id"] = df["user_id"].map(dict(zip(um.user_id_csv, um.user_id)))
    df["book_id"] = df["book_id"].map(dict(zip(bm.book_id_csv, bm.book_id)))
    df["rating"] = df["rating"].astype(float)
    return df


REVIEW_RE = {k: re.compile(rf'"{k}": "([^"]*)"') for k in ("user_id", "book_id", "date_added", "timestamp")}


def load_review_times(raw_dir):
    """(user_id, book_id) -> review time in unix seconds, plus file order."""
    full = os.path.join(raw_dir, "goodreads_reviews_dedup.json.gz")
    if os.path.isfile(full):
        path, key, fmt = full, "date_added", "%a %b %d %H:%M:%S %z %Y"
    else:
        path, key, fmt = os.path.join(raw_dir, "goodreads_reviews_spoiler.json.gz"), "timestamp", "%Y-%m-%d"
    print(f"review times from {os.path.basename(path)} ({key})")
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:  # regex, not json: review text is large; escaped quotes cannot match
            rows.append(tuple(REVIEW_RE[k].search(line).group(1) for k in ("user_id", "book_id", key)))
    df = pd.DataFrame(rows, columns=["user_id", "book_id", "date"])
    ts = pd.to_datetime(df["date"], format=fmt, utc=True, errors="coerce")  # a few dates are corrupt (e.g. year 1092)
    df["order"] = np.arange(len(df))
    df = df[ts.notna()].copy()
    df["timestamp"] = ts[ts.notna()].astype("int64") // 10**9
    print(f"dropped {int(ts.isna().sum())} reviews with unparseable dates")
    return df.drop(columns="date").drop_duplicates(["user_id", "book_id"], keep="first")


def load_books(path, wanted):
    books = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            m = BOOK_ID_RE.search(line)
            if not m or m.group(1) not in wanted:
                continue
            d = json.loads(line)
            d["popular_shelves"] = d.get("popular_shelves", [])[:15]
            books[d["book_id"]] = {k: d.get(k) for k in (
                "work_id", "title", "title_without_series", "publication_year", "publisher",
                "num_pages", "format", "popular_shelves", "description")}
    return books


def unique_titles(books):
    """work id -> title, unique across the given works."""
    df = pd.DataFrame({
        "book_id": list(books),
        "title": [clean_text(b.get("title") or b.get("title_without_series")) for b in books.values()],
        "year": [clean_text(b.get("publication_year")) for b in books.values()],
    })
    dup = df["title"].duplicated(keep=False) & (df["year"] != "")
    df.loc[dup, "title"] = df.loc[dup, "title"] + " (" + df.loc[dup, "year"] + ")"
    dup = df["title"].duplicated(keep=False)
    df.loc[dup, "title"] = df.loc[dup, "title"] + " [" + df.loc[dup, "book_id"] + "]"
    assert not df["title"].duplicated().any()
    return dict(zip(df["book_id"], df["title"]))


def describe(book):
    parts = []
    shelves = [s["name"] for s in book.get("popular_shelves", [])
               if s["name"] not in {"to-read", "currently-reading", "favorites", "owned", "books-i-own"}][:8]
    if shelves:
        parts.append("Shelves: " + ", ".join(shelves) + ".")
    info = [f"{k.replace('_', ' ')}: {clean_text(book.get(k))}"
            for k in ("publisher", "publication_year", "num_pages", "format")
            if clean_text(book.get(k))]
    if info:
        parts.append("; ".join(info) + ".")
    desc = (book.get("description") or "").strip()
    if desc:
        parts.append(desc)
    return parts


def main(
    K: int = 5,
    st_year: int = 2017,
    st_month: int = 1,
    ed_year: int = 2017,
    ed_month: int = 12,
    n_users: int = 26000,
    window_size: int = 10,
    max_title_words: int = 20,
    seed: int = 42,
    raw_dir: str = HERE,
    output_dir: str = HERE,
    name: str = "",
):
    """Reviewed interactions + review time -> works, dedup -> window [st, ed) -> sample n_users (0 = all) -> K-core -> splits."""
    start = pd.Timestamp(year=st_year, month=st_month, day=1).timestamp()
    end = pd.Timestamp(year=ed_year, month=ed_month, day=1).timestamp()
    name = name or f"Goodreads_{K}_{st_year}-{st_month}-{ed_year}-{ed_month}"

    df = load_reviewed_interactions(raw_dir)
    print(f"reviewed interactions: {len(df)}, {df.user_id.nunique()} users, {df.book_id.nunique()} books")
    times = load_review_times(raw_dir)
    df = df.merge(times, on=["user_id", "book_id"], how="inner")
    print(f"with review time: {len(df)}, {df.user_id.nunique()} users, {df.book_id.nunique()} books")

    books = load_books(os.path.join(raw_dir, "goodreads_books.json.gz"), set(df.book_id))
    titled = {b for b, d in books.items()
              if 1 < len(clean_text(d.get("title"))) and len(clean_text(d.get("title")).split()) <= max_title_words}
    df = df[df.book_id.isin(titled)].copy()
    # editions -> works; duplicate (user, work): keep the first interaction in time
    df["edition"] = df["book_id"]
    df["book_id"] = df["edition"].map(lambda b: books[b].get("work_id") or "e" + b)
    df = df.sort_values(["timestamp", "order"], kind="stable")
    n_before = len(df)
    df = df.drop_duplicates(["user_id", "book_id"], keep="first")
    print(f"removed {n_before - len(df)} duplicate (user, work) interactions")
    df = df[(df.timestamp >= start) & (df.timestamp < end)]
    # representative edition = most-interacted one in the window
    rep = df.groupby(["book_id", "edition"]).size().reset_index(name="n")
    rep = rep.sort_values(["book_id", "n", "edition"], ascending=[True, False, True])
    work2edition = dict(rep.drop_duplicates("book_id")[["book_id", "edition"]].values)
    books = {w: books[e] for w, e in work2edition.items()}
    print(f"window: {len(df)} interactions, {df.user_id.nunique()} users, {df.book_id.nunique()} books")

    if n_users:
        users = np.sort(df.user_id.unique())
        rng = np.random.default_rng(seed)
        sampled = set(rng.choice(users, min(n_users, len(users)), replace=False))
        df = df[df.user_id.isin(sampled)]
    df = kcore(df, K)
    n_u, n_i = df.user_id.nunique(), df.book_id.nunique()
    print(f"after sample + {K}-core: users={n_u}, items={n_i}, interactions={len(df)}, "
          f"density={len(df) / (n_u * n_i):.5f}")

    kept = {b: books[b] for b in sorted(set(df.book_id))}
    asin2title = unique_titles(kept)
    items = sorted(asin2title)
    random.seed(seed)
    random.shuffle(items)
    asin2id = {a: i for i, a in enumerate(items)}

    os.makedirs(os.path.join(output_dir, "info"), exist_ok=True)
    with open(os.path.join(output_dir, "info", f"{name}.txt"), "w") as f:
        for a in items:
            f.write(f"{asin2title[a]}\t{asin2id[a]}\n")
    with open(os.path.join(output_dir, "info", f"{name}.item_meta.json"), "w") as f:
        json.dump({a: {"title": asin2title[a], "description": describe(kept[a])} for a in items}, f)

    df = df.sort_values(["timestamp", "order"], kind="stable")
    rows = []
    for user, g in df.groupby("user_id", sort=False):
        a = g.book_id.tolist()
        ids = [asin2id[x] for x in a]
        titles = [asin2title[x] for x in a]
        r = g.rating.tolist()
        t = g.timestamp.tolist()
        for i in range(1, len(a)):
            s = max(i - window_size, 0)
            rows.append([user, a[s:i], a[i], ids[s:i], ids[i], titles[s:i], titles[i],
                         r[s:i], r[i], t[s:i], t[i]])
    rows.sort(key=lambda x: int(x[-1]))

    n = len(rows)
    splits = {"train": rows[: int(n * 0.8)], "valid": rows[int(n * 0.8): int(n * 0.9)],
              "test": rows[int(n * 0.9):]}
    for split, part in splits.items():
        out = pd.DataFrame(part, columns=CSV_COLUMNS)
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
        path = os.path.join(output_dir, split, f"{name}.csv")
        out.to_csv(path, index=False)
        print(f"[{split}] rows={len(out)} users={out.user_id.nunique()} "
              f"target_items={out.item_id.nunique()} -> {path}")
    print(f"items: {len(items)} -> {os.path.join(output_dir, 'info', name + '.txt')}")


if __name__ == "__main__":
    fire.Fire(main)
