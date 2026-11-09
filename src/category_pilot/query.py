"""Ask an OpenAI-compatible vLLM server to predict the next item's category (standard library only).

Variants: A = history + taxonomy; B = A + neighbour statistics; B_think = B with thinking enabled.
Resumable: rows already in <out>/<variant>.jsonl with a parsed answer are skipped.
"""

import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

SYSTEM = (
    "You are an expert in e-commerce shopping behaviour for toys and games. Given a customer's purchase "
    "history (oldest first), predict the product CATEGORY of the very next item they will buy. The next "
    "item is often from a different category than the history, so reason about what such customers "
    "typically buy next (complements, the next stage of the same interest, gifts for the same child, "
    "compatible brands), not only about repeating the last category. Choose ONLY from the supplied list "
    "of category paths and copy each path exactly."
)


def user_prompt(row, taxonomy, with_stats):
    # Taxonomy first: identical across requests, so vLLM's prefix cache reuses it.
    lines = [f"{i + 1}. {h['title']}  [category: {h['category']}]" for i, h in enumerate(row["history"])]
    parts = ["Allowed category paths:", *taxonomy, "", "Purchase history (oldest first):", *lines, ""]
    if with_stats:
        stats = row["neighbour_next_categories"]
        if stats:
            parts += ["Customers with similar purchase histories next bought items in these categories "
                      "(share of similar customers):",
                      *[f"- {s['category']}: {s['share']:.0%}" for s in stats], ""]
    parts += ['Choose only from the allowed category paths above. Return JSON only: {"predictions": [{"category": "<exact path>", "confidence": <0-1>, '
              '"reason": "<one short sentence>"}, ... exactly 3 items, most likely first]}']
    return "\n".join(parts)


def call(args, messages, think, seed):
    body = {"model": args.model, "messages": messages, "temperature": 0.0 if not think else 0.6,
            "top_p": 0.95, "max_tokens": 6000 if think else 600, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": think}}
    if not think:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=args.timeout) as response:
        return json.load(response)


def parse(text):
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    start, end = text.find("{"), text.rfind("}")
    if start < 0:
        return None
    try:
        preds = json.loads(text[start:end + 1]).get("predictions")
    except json.JSONDecodeError:
        return None
    if not isinstance(preds, list):
        return None
    return [{"category": str(p.get("category", "")).strip(), "confidence": p.get("confidence"),
             "reason": p.get("reason")} for p in preds[:3] if isinstance(p, dict)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", required=True)
    ap.add_argument("--variants", default="A,B,B_think")
    ap.add_argument("--base-url", default="http://127.0.0.1:18431/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()
    taxonomy = json.load(open(os.path.join(args.pilot_dir, "taxonomy.json")))
    rows = [json.loads(l) for l in open(os.path.join(args.pilot_dir, "inputs.jsonl"))]
    for variant in args.variants.split(","):
        out = os.path.join(args.pilot_dir, f"{variant}.jsonl")
        done = set()
        if os.path.exists(out):
            for l in open(out):
                rec = json.loads(l)
                if rec.get("predictions"):
                    done.add(rec["row"])
        todo = [r for r in rows if r["row"] not in done]
        think, with_stats = variant.endswith("_think"), variant.startswith("B")

        def work(row):
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user_prompt(row, taxonomy, with_stats)}]
            last = None
            for attempt in range(args.retries + 1):
                try:
                    resp = call(args, messages, think, seed=row["row"] + attempt)
                    text = resp["choices"][0]["message"].get("content") or ""
                    preds = parse(text)
                    if preds:
                        return {"row": row["row"], "variant": variant, "predictions": preds,
                                "completion_tokens": resp.get("usage", {}).get("completion_tokens")}
                    last = text[-500:]
                except Exception as exc:  # network/timeouts: retry
                    last = repr(exc)
            return {"row": row["row"], "variant": variant, "predictions": None, "error": last}

        with open(out, "a") as handle, ThreadPoolExecutor(args.concurrency) as pool:
            futures = [pool.submit(work, r) for r in todo]
            for i, f in enumerate(as_completed(futures), 1):
                handle.write(json.dumps(f.result()) + "\n")
                handle.flush()
                if i % 50 == 0 or i == len(futures):
                    print(f"{variant}: {i}/{len(futures)}", flush=True)


if __name__ == "__main__":
    main()
