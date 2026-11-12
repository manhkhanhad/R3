#!/usr/bin/env python3
"""Progress check for the side-information tests.

Prints, per variant: SLURM state of its latest job, last training epoch / eval
loss, and the key results once analysis/results/side_info/full_<variant>/summary.json
exists. Run from the repository root:  python scripts/side_info_progress.py
"""

import glob
import json
import os
import re
import subprocess

VARIANTS = ["oracle_raw", "oracle_token", "null_raw", "text_control", "sasrec_raw", "plain_token", "oracle_cfitem", "oracle_cfcluster", "oracle_cfcluster150"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# checkpoints live in scratch, or in the repo output_dir for runs moved there when scratch was full
OUTPUT_ROOTS = ["/common/home/users/m/mk.ngohuu.2024/scratchDirectory/R3/output_dir", os.path.join(ROOT, "output_dir")]


def job_state(job_id):
    out = subprocess.run(["sacct", "-j", job_id, "-X", "-n", "-o", "State,Elapsed"], capture_output=True, text=True).stdout.split()
    return " ".join(out[:2]) if out else "unknown"


def latest_log(variant):
    logs = glob.glob(os.path.join(ROOT, "run_logs", f"R3_side_{variant}.*.out"))
    return max(logs, key=lambda p: int(p.rsplit(".", 2)[1])) if logs else None


def train_progress(variant):
    states = []
    for root in OUTPUT_ROOTS:
        out = os.path.join(root, f"Musical_Instruments_R3_side_{variant}", "latent")
        states += glob.glob(os.path.join(out, "checkpoint-*", "trainer_state.json"))
    states = sorted(states, key=lambda p: int(re.findall(r"checkpoint-(\d+)", p)[0]))
    if not states:
        return "no checkpoint yet"
    hist = json.load(open(states[-1]))["log_history"]
    evals = [f"{h['eval_loss']:.3f}" for h in hist if "eval_loss" in h]
    return f"epochs done {len(evals)}; eval_loss {' -> '.join(evals)}"


def main():
    print(f"{'variant':14s} {'job':>8s} {'state':22s} training")
    for v in VARIANTS:
        log = latest_log(v)
        jid = log.rsplit(".", 2)[1] if log else "-"
        print(f"{v:14s} {jid:>8s} {job_state(jid) if log else '-':22s} {train_progress(v)}")
    print()
    for v in VARIANTS:
        path = os.path.join(ROOT, "analysis/results/side_info", f"full_{v}", "summary.json")
        if not os.path.exists(path):
            continue
        s = json.load(open(path))
        ratio = s.get("ratio_to_drop_last", {})
        line = "  ".join(f"{k}={ratio[k]:.4f}" for k in ("swap_side", "swap_r", "swap_all", "mask") if k in ratio)
        print(f"{v:14s} ratio-to-drop-last: {line}")
        for key in ("generation", "full_ranking"):
            g = s.get(key)
            if not g:
                continue
            for cond in ("true", "swap_side"):
                c = g.get(cond, g if key == "full_ranking" else None)
                if c:
                    extra = "  ".join(f"{m}={c[m]:.4f}" for m in ("HR@10", "NDCG@10", "top10_in_own_category", "top10_in_donor_category", "donor_target_hit@10") if m in c)
                    print(f"{'':14s} {key}/{cond if key == 'generation' else 'true'}: {extra}")
                if key == "full_ranking":
                    break


if __name__ == "__main__":
    main()
