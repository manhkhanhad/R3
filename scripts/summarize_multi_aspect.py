#!/usr/bin/env python3
"""Print the key numbers of multi-aspect evaluation summaries.
Usage: python scripts/summarize_multi_aspect.py confidence mean"""
import json
import sys

for t in sys.argv[1:]:
    s = json.load(open(f"analysis/results/multi_aspect/full_{t}/summary.json"))
    print(f"##### {t}  gate_scale={s['gate_scale']:.3f} bias={[round(x, 3) for x in s['gate_bias']]}")
    for a, p in s["probes"].items():
        print(f"  probe {a:9s} head={p['head_top1']:.3f} last={p['last_history_item']:.3f} major={p['history_majority']:.3f} "
              f"gate={p['mean_gate']:.3f} conf={p['mean_confidence']:.3f} g_ok={p['gate_when_head_correct']} g_bad={p['gate_when_head_wrong']}")
    print("  ratio", {k: round(v, 4) for k, v in s["sensitivity"]["ratio_to_drop_last"].items()})
    r = s["rerank"]
    f = lambda m: f"HR@10={m['HR@10']:.4f} NDCG@10={m['NDCG@10']:.4f}"
    print("  lambda", r["chosen_lambda"], "recall@20", round(r["candidate_recall@beams"], 4))
    print("  valid grid", {k: round(v["NDCG@10"], 4) for k, v in r["valid_grid"].items()})
    print("  test decoder-only", f(r["test_decoder_only"]), "| rescored", f(r["test_rescored"]))
    print("  test grid", {k: (round(v["HR@10"], 4), round(v["NDCG@10"], 4)) for k, v in r["test_grid"].items()})
    print("  single aspect", {k: (round(v["HR@10"], 4), round(v["NDCG@10"], 4)) for k, v in r["test_rescored_single_aspect"].items()})
