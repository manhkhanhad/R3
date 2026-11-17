"""Convert the preprocessed Yelp splits (data/Yelp, made by preprocess_yelp.py
for the RRec / Verified-Rec format) into the R3 / Amazon-Toys format.

data/Yelp is left untouched (other runs use it). Output:

    data/Yelp_R3/{train,valid,test}/<name>.csv
    data/Yelp_R3/info/<name>.txt              "<title>\t<id>", line number == item_id
    data/Yelp_R3/info/<name>.item_meta.json   {business_id: {"title", "description"}}

Changes from data/Yelp:
  * Same rows and chronological 80/10/10 split (cold-start valid/test rows were
    already removed there). neg_item_ids and interaction_id are dropped.
  * Item ids are remapped to 0..N-1 (pad item removed); order follows data/Yelp.
  * History truncated to the last `window_size` (default 10, as in process.py).
  * Timestamps converted from milliseconds to seconds.
  * Duplicate business names (chains, 1,785 rows) are made unique, since
    evaluation matches by title: " (<street address>)", then ", <city>",
    then " [<business_id>]".
  * List columns are written as python reprs (eval()-able), like the Toys CSVs.

Usage (from repo root):  python data/Yelp_R3/convert_yelp_for_r3.py
"""

import json
import os

import fire
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_COLUMNS = [
    "user_id", "item_asins", "item_asin", "history_item_id", "item_id",
    "history_item_title", "item_title", "history_rating", "rating",
    "history_timestamp", "timestamp",
]


def unique_titles(info):
    title = info["title"].astype(str).str.strip()
    loc = info["subtitle"].fillna("").astype(str).str.split(", ")
    street = loc.map(lambda p: p[0] if p and p[0] else "")
    city = loc.map(lambda p: p[-3] if len(p) >= 3 else "")
    dup = title.duplicated(keep=False) & (street != "")
    title[dup] = title[dup] + " (" + street[dup] + ")"
    dup = title.duplicated(keep=False) & (city != "")
    title[dup] = title[dup].str[:-1] + ", " + city[dup] + ")"
    dup = title.duplicated(keep=False)
    title[dup] = title[dup] + " [" + info.loc[dup, "parent_asin"] + "]"
    assert not title.duplicated().any() and not title.str.contains("\t").any()
    return title


def main(src_dir: str = os.path.join(os.path.dirname(HERE), "Yelp"),
         output_dir: str = HERE,
         name: str = "Yelp_5_2021-1-2022-2",
         window_size: int = 10):
    info = pd.read_csv(os.path.join(src_dir, "item_info", f"{name}.csv"))
    info = info[info["parent_asin"] != "pad_asin"].sort_values("item_id").reset_index(drop=True)
    old2new = {int(o): n for n, o in enumerate(info["item_id"])}
    info["title"] = unique_titles(info)
    titles = info["title"].tolist()
    print(f"items: {len(info)}, renamed duplicates: "
          f"{(info['title'] != pd.read_csv(os.path.join(src_dir, 'item_info', f'{name}.csv')).set_index('parent_asin').loc[info['parent_asin'], 'title'].values).sum()}")

    os.makedirs(os.path.join(output_dir, "info"), exist_ok=True)
    with open(os.path.join(output_dir, "info", f"{name}.txt"), "w") as f:
        for i, t in enumerate(titles):
            f.write(f"{t}\t{i}\n")
    meta = {a: {"title": t, "description": json.loads(d) if isinstance(d, str) else []}
            for a, t, d in zip(info["parent_asin"], titles, info["description"])}
    with open(os.path.join(output_dir, "info", f"{name}.item_meta.json"), "w") as f:
        json.dump(meta, f)

    for split in ("train", "valid", "test"):
        df = pd.read_csv(os.path.join(src_dir, split, f"{name}.csv"))
        out = pd.DataFrame({"user_id": df["user_id"], "item_asin": df["item_asin"]})
        hist_ids = df["history_item_id"].map(lambda s: [old2new[i] for i in json.loads(s)][-window_size:])
        out["item_asins"] = df["item_asins"].map(lambda s: json.loads(s)[-window_size:])
        out["history_item_id"] = hist_ids
        out["item_id"] = df["item_id"].map(old2new)
        out["history_item_title"] = hist_ids.map(lambda h: [titles[i] for i in h])
        out["item_title"] = out["item_id"].map(lambda i: titles[i])
        out["history_rating"] = df["history_rating"].map(lambda s: [float(x) for x in json.loads(s)][-window_size:])
        out["rating"] = df["rating"].astype(float)
        out["history_timestamp"] = df["history_timestamp"].map(lambda s: [int(x) // 1000 for x in json.loads(s)][-window_size:])
        out["timestamp"] = df["timestamp"].astype("int64") // 1000
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
        path = os.path.join(output_dir, split, f"{name}.csv")
        out[CSV_COLUMNS].to_csv(path, index=False)
        print(f"[{split}] rows={len(out)} users={out.user_id.nunique()} -> {path}")


if __name__ == "__main__":
    fire.Fire(main)
