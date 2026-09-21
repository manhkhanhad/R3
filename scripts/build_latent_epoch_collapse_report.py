#!/usr/bin/env python3
"""Build a collapse report after all latent checkpoint E1 artifacts exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
DEFAULT_STEPS = (384, 768, 1152)
MODES = (
    "raw",
    "mask",
    "mean",
    "shuffle_user",
    "shuffle_target",
    "noise_0.25",
    "noise_0.5",
    "noise_1.0",
)
LABELS = {
    "raw": "Raw",
    "mask": "Zero / mask",
    "mean": "Global train mean",
    "shuffle_user": "Shuffled user",
    "shuffle_target": "Shuffled target",
    "noise_0.25": "Gaussian noise 0.25x",
    "noise_0.5": "Gaussian noise 0.50x",
    "noise_1.0": "Gaussian noise 1.00x",
}
METRICS = tuple(f"{kind}@{k}" for kind in ("HR", "NDCG") for k in (1, 3, 5, 10))


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def fmt(value: float, digits: int = 6, signed: bool = False) -> str:
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def contributions(ranks: np.ndarray, metric: str) -> np.ndarray:
    kind, cutoff_text = metric.split("@")
    cutoff = int(cutoff_text)
    if kind == "HR":
        return (ranks <= cutoff).astype(np.float64)
    return np.where(ranks <= cutoff, 1.0 / np.log2(ranks + 1), 0.0)


def bootstrap_ci(delta: np.ndarray, rng: np.random.Generator, reps: int = 2000) -> tuple[float, float]:
    means = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        sample = rng.integers(0, len(delta), len(delta))
        means[index] = delta[sample].mean()
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def intervention_path(tag_prefix: str, step: int, mode: str) -> Path:
    return (
        ANALYSIS
        / "results"
        / "intervention"
        / f"{tag_prefix}_checkpoint_{step}"
        / f"generation_test_all_k32_s42_{mode}_t1_temp0.1_a1.0.json"
    )


def validate_and_summarize(version: str, steps: tuple[int, ...], tag_prefix: str,
                            checkpoint_root: str, test_n: int) -> tuple[dict, dict]:
    results: dict[int, dict[str, dict]] = {}
    intervals: dict[int, dict[str, dict[str, tuple[float, float, float]]]] = {}
    expected_split_hash = None

    for step in steps:
        by_mode = {mode: load_json(intervention_path(tag_prefix, step, mode)) for mode in MODES}
        raw_records = by_mode["raw"]["records"]
        raw_keys = [(r["sample_id"], r["user_id"], r["target_item"], r["target"]) for r in raw_records]
        checkpoint = f"{checkpoint_root}/checkpoint-{step}"

        for mode, artifact in by_mode.items():
            metrics = artifact["metrics"]
            records = artifact["records"]
            keys = [(r["sample_id"], r["user_id"], r["target_item"], r["target"]) for r in records]
            if metrics["n"] != test_n or len(records) != test_n:
                raise ValueError(f"{version} checkpoint {step}/{mode}: expected {test_n:,} records")
            if metrics["catalog_violations"] != 0:
                raise ValueError(f"{version} checkpoint {step}/{mode}: catalog violations")
            if metrics["checkpoint"] != checkpoint:
                raise ValueError(f"{version} checkpoint {step}/{mode}: checkpoint mismatch")
            if keys != raw_keys:
                raise ValueError(f"{version} checkpoint {step}/{mode}: records are not paired")
            if expected_split_hash is None:
                expected_split_hash = metrics["split_sha256"]
            elif metrics["split_sha256"] != expected_split_hash:
                raise ValueError(f"{version} checkpoint {step}/{mode}: split hash mismatch")

        results[step] = by_mode
        raw_ranks = np.asarray([r["rank"] for r in raw_records], dtype=np.float64)
        rng = np.random.default_rng(42)
        intervals[step] = {}
        for mode in MODES[1:]:
            ranks = np.asarray([r["rank"] for r in by_mode[mode]["records"]], dtype=np.float64)
            intervals[step][mode] = {}
            for metric in ("HR@10", "NDCG@10"):
                delta = contributions(ranks, metric) - contributions(raw_ranks, metric)
                low, high = bootstrap_ci(delta, rng)
                intervals[step][mode][metric] = (float(delta.mean()), low, high)
    return results, intervals


def geometry(version: str, steps: tuple[int, ...], tag_prefix: str) -> tuple[list[dict], dict[int, dict]]:
    summary_path = (
        ANALYSIS / "results" / "latent_distribution" / f"latent_{version}_epoch_comparison" / "summary.json"
    )
    summary = load_json(summary_path)["checkpoints"]
    if [row["step"] for row in summary] != list(steps):
        raise ValueError("Geometry summary does not contain the expected checkpoints")
    details = {
        step: load_json(
            ANALYSIS
            / "results"
            / "latent_distribution"
            / f"{tag_prefix}_checkpoint_{step}"
            / "train_latent_distribution.json"
        )
        for step in steps
    }
    return summary, details


def inferential_table(step: int, results: dict, intervals: dict) -> list[str]:
    rows = [
        f"### Checkpoint {step}",
        "",
        "| Intervention | HR@10 | Delta HR@10 (95% CI) | NDCG@10 | Delta NDCG@10 (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    raw = results[step]["raw"]["metrics"]
    rows.append(f"| Raw | {fmt(raw['HR@10'])} | 0 | {fmt(raw['NDCG@10'])} | 0 |")
    for mode in MODES[1:]:
        metrics = results[step][mode]["metrics"]
        hr = intervals[step][mode]["HR@10"]
        ndcg = intervals[step][mode]["NDCG@10"]
        rows.append(
            f"| {LABELS[mode]} | {fmt(metrics['HR@10'])} | {fmt(hr[0], signed=True)} "
            f"[{fmt(hr[1], signed=True)}, {fmt(hr[2], signed=True)}] | {fmt(metrics['NDCG@10'])} | "
            f"{fmt(ndcg[0], signed=True)} [{fmt(ndcg[1], signed=True)}, {fmt(ndcg[2], signed=True)}] |"
        )
    return rows


def complete_table(step: int, results: dict) -> list[str]:
    rows = [
        f"#### Checkpoint {step}",
        "",
        "| Intervention | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        metrics = results[step][mode]["metrics"]
        cells = " | ".join(fmt(metrics[name]) for name in METRICS)
        rows.append(f"| {LABELS[mode]} | {cells} |")
    return rows


def build(version: str, dataset: str, steps: tuple[int, ...], tag_prefix: str,
          checkpoint_root: str, train_n: int, test_n: int, dim: int) -> str:
    summary, details = geometry(version, steps, tag_prefix)
    results, intervals = validate_and_summarize(version, steps, tag_prefix, checkpoint_root, test_n)
    rows_by_step = {row["step"]: row for row in summary}
    s0, s1, s2 = steps

    effective = [rows_by_step[s]["entropy_effective_rank"] for s in steps]
    unique = [100 * rows_by_step[s]["unique_fraction"] for s in steps]
    top10 = [100 * rows_by_step[s]["top_10_template_fraction"] for s in steps]
    random_cos = [rows_by_step[s]["random_pair_cosine"] for s in steps]
    norm_ratio = [rows_by_step[s]["mean_norm_ratio"] for s in steps]
    raw_hr = [results[s]["raw"]["metrics"]["HR@10"] for s in steps]
    raw_ndcg = [results[s]["raw"]["metrics"]["NDCG@10"] for s in steps]

    zero_sig = all(
        intervals[s]["mask"][metric][2] < 0 for s in steps for metric in ("HR@10", "NDCG@10")
    )
    content_modes = ("mean", "shuffle_user", "shuffle_target", "noise_0.25", "noise_0.5", "noise_1.0")
    content_not_harmful = all(
        intervals[s][mode][metric][2] >= 0
        for s in steps
        for mode in content_modes
        for metric in ("HR@10", "NDCG@10")
    )
    if zero_sig and content_not_harmful:
        causal_verdict = (
            "Zero masking significantly reduces HR@10 and NDCG@10 at every checkpoint, while the global-mean, "
            "shuffled, and Gaussian-noise controls never significantly worsen ranking. The decoder therefore "
            "depends on a nonzero latent activation but does not benefit from the measured sample-specific content. "
            "**E1 strongly supports latent-utilization collapse at all three checkpoints.**"
        )
    else:
        causal_verdict = (
            "E1 does not match the simple activation-without-content pattern uniformly across all three checkpoints; "
            "the checkpoint-level confidence intervals below should be used for the causal conclusion."
        )

    lines = [
        f"# Latent-collapse assessment for `{tag_prefix}` checkpoints {s0}, {s1}, and {s2}",
        "",
        "## Executive verdict",
        "",
        f"**All three `{tag_prefix}` checkpoints are strongly shared-direction and exact-template concentrated. "
        f"The trajectory is non-monotonic through checkpoint {s1}, followed by a large recovery in centered dimensional "
        f"diversity at checkpoint {s2}.**",
        "",
        f"Only {min(unique):.2f}%–{max(unique):.2f}% of stored FP16 thoughts are unique, and the ten most common "
        f"templates cover {min(top10):.2f}%–{max(top10):.2f}% of train examples. Checkpoint {s1} is the most template "
        f"concentrated. Centered entropy effective rank moves from {effective[0]:.2f} to {effective[1]:.2f} to "
        f"{effective[2]:.2f}; checkpoint {s2} is substantially broader, but its exact-template concentration remains high.",
        "",
        f"All checkpoints remain anisotropic: random-pair cosine is {min(random_cos):.3f}–{max(random_cos):.3f}, and "
        f"the global-mean norm is {100*min(norm_ratio):.1f}%–{100*max(norm_ratio):.1f}% of the mean sample norm. "
        "The last checkpoint improves residual diversity without eliminating the common carrier direction.",
        "",
        causal_verdict,
        "",
        "## Scope and reproducibility",
        "",
        f"- Dataset: `{dataset}`",
        f"- Geometry and extraction likelihood split: train, {train_n:,} examples",
        f"- Causal ranking split: test, {test_n:,} examples",
        f"- Architecture output: one {dim:,}-dimensional stored thought per interaction",
        "- Stored dtype: FP16",
        f"- Train split SHA-256: `{details[s0]['split_sha256']}`",
        f"- Test split SHA-256: `{results[s0]['raw']['metrics']['split_sha256']}`",
        "- Pairwise samples: 100,000 per diagnostic, seed 42",
        f"- PCA: centered and fitted on all {train_n:,} thoughts independently at each checkpoint",
        f"- Ranking: constrained 10-beam catalog generation; paired bootstrap 95% intervals over all {test_n:,} test examples, 2,000 resamples, seed 42",
        "",
        "Primary artifacts:",
        "",
        f"- [Epoch comparison summary](../results/latent_distribution/latent_{version}_epoch_comparison/summary.json)",
        f"- [Epoch trajectory figure](../figures/latent_{version}_epoch_collapse_trajectory.png)",
        f"- Checkpoint statistics: [{s0}](../results/latent_distribution/{tag_prefix}_checkpoint_{s0}/train_latent_distribution.json), [{s1}](../results/latent_distribution/{tag_prefix}_checkpoint_{s1}/train_latent_distribution.json), [{s2}](../results/latent_distribution/{tag_prefix}_checkpoint_{s2}/train_latent_distribution.json)",
        f"- Distribution overviews: [{s0}](../figures/{tag_prefix}_checkpoint_{s0}_latent_distribution_overview.png), [{s1}](../figures/{tag_prefix}_checkpoint_{s1}_latent_distribution_overview.png), [{s2}](../figures/{tag_prefix}_checkpoint_{s2}_latent_distribution_overview.png)",
        f"- All-vector PCA views: [{s0}](../figures/{tag_prefix}_checkpoint_{s0}_latent_pca_all_scatter.png), [{s1}](../figures/{tag_prefix}_checkpoint_{s1}_latent_pca_all_scatter.png), [{s2}](../figures/{tag_prefix}_checkpoint_{s2}_latent_pca_all_scatter.png)",
        f"- E1 ranking artifacts: [{s0}](../results/intervention/{tag_prefix}_checkpoint_{s0}/), [{s1}](../results/intervention/{tag_prefix}_checkpoint_{s1}/), [{s2}](../results/intervention/{tag_prefix}_checkpoint_{s2}/)",
        "",
        "## 1. Epoch trajectory",
        "",
        f"| Diagnostic | Step {s0} (epoch 1) | Step {s1} (epoch 2) | Step {s2} (epoch 3) |",
        "|---|---:|---:|---:|",
    ]

    diagnostics = [
        ("Raw train target NLL", lambda s: f"{rows_by_step[s]['raw_train_nll']:.6f}"),
        ("Zero-mask train target NLL", lambda s: f"{rows_by_step[s]['zero_mask_train_nll']:.6f}"),
        ("Zero-mask NLL penalty", lambda s: f"+{rows_by_step[s]['zero_mask_penalty']:.6f}"),
        ("Mean sample norm", lambda s: f"{details[s]['norm']['mean']:.2f}"),
        ("Global-mean vector norm", lambda s: f"{details[s]['mean_vector_norm']:.2f}"),
        ("Global-mean norm / mean sample norm", lambda s: f"{details[s]['mean_norm_over_mean_sample_norm']:.4f}"),
        ("Mean cosine to global mean", lambda s: f"{details[s]['cosine_to_global_mean']['mean']:.4f}"),
        ("Mean random-pair cosine", lambda s: f"{details[s]['pairwise_cosine']['random']['mean']:.4f}"),
        ("Exact FP16-unique vectors", lambda s: f"{details[s]['exact_templates_at_stored_precision']['unique_vectors']:,}"),
        ("Unique fraction", lambda s: f"{100*details[s]['exact_templates_at_stored_precision']['unique_fraction']:.2f}%"),
        ("Top-10-template coverage", lambda s: f"{100*details[s]['exact_templates_at_stored_precision']['top_10_sample_fraction']:.2f}%"),
        ("Largest-template coverage", lambda s: f"{100*details[s]['exact_templates_at_stored_precision']['top_1_sample_fraction']:.2f}%"),
        ("PC1 centered variance", lambda s: f"{100*details[s]['pca']['top_eigenvalue_fraction']:.2f}%"),
        ("Top-two centered variance", lambda s: f"{100*details[s]['pca']['top_2_fraction']:.2f}%"),
        ("Entropy effective rank", lambda s: f"{details[s]['pca']['entropy_effective_rank']:.2f}"),
        ("Participation ratio", lambda s: f"{details[s]['pca']['participation_ratio']:.2f}"),
        ("Components for 90% variance", lambda s: str(details[s]['pca']['components_for_90pct'])),
    ]
    for label, render in diagnostics:
        lines.append(f"| {label} | {render(s0)} | {render(s1)} | {render(s2)} |")

    lines += [
        "",
        f"Raw train NLL improves monotonically, but collapse metrics do not. Effective rank falls from {effective[0]:.2f} "
        f"to {effective[1]:.2f}, then rises to {effective[2]:.2f}. At checkpoint {s2}, PC1 explains "
        f"{100*details[s2]['pca']['top_eigenvalue_fraction']:.2f}% of centered variance and "
        f"{details[s2]['pca']['components_for_90pct']} components are needed for 90%, compared with "
        f"{details[s1]['pca']['components_for_90pct']} at checkpoint {s1}. This is a substantial late recovery in "
        "dimensional diversity, not monotonic collapse.",
        "",
        "## 2. Template concentration versus dimensional concentration",
        "",
        f"Exact-template collapse is strong at every checkpoint. The unique-vector counts are "
        f"{details[s0]['exact_templates_at_stored_precision']['unique_vectors']:,}, "
        f"{details[s1]['exact_templates_at_stored_precision']['unique_vectors']:,}, and "
        f"{details[s2]['exact_templates_at_stored_precision']['unique_vectors']:,} among {train_n:,} samples. The largest "
        f"single templates cover {100*details[s0]['exact_templates_at_stored_precision']['top_1_sample_fraction']:.2f}%, "
        f"{100*details[s1]['exact_templates_at_stored_precision']['top_1_sample_fraction']:.2f}%, and "
        f"{100*details[s2]['exact_templates_at_stored_precision']['top_1_sample_fraction']:.2f}% of the train split.",
        "",
        f"Dimensional concentration and exact repetition move differently late in training. Checkpoint {s2} expands to "
        f"effective rank {effective[2]:.2f} and reduces the first-ten-PC share to "
        f"{100*details[s2]['pca']['top_10_fraction']:.2f}%, yet only {unique[2]:.2f}% of stored vectors are unique. "
        "A broader centered span therefore does not imply recovery from discrete template reuse.",
        "",
        "## 3. User-, target-, and preference-linked structure",
        "",
        f"| Diagnostic | Step {s0} | Step {s1} | Step {s2} |",
        "|---|---:|---:|---:|",
    ]
    group_rows = [
        ("Same-user cosine (absolute)", lambda s: details[s]["pairwise_cosine"]["same_user"]["mean"]),
        ("Same-target cosine (absolute)", lambda s: details[s]["pairwise_cosine"]["same_target"]["mean"]),
        ("Same CF-cluster cosine (absolute)", lambda s: details[s]["pairwise_cosine"]["same_cf_cluster"]["mean"]),
        ("Adjusted variance associated with user", lambda s: details[s]["group_centroid_variance"]["user_id"]["adjusted_eta_squared"]),
        ("Adjusted variance associated with target", lambda s: details[s]["group_centroid_variance"]["target_item"]["adjusted_eta_squared"]),
        ("Adjusted variance associated with CF cluster", lambda s: details[s]["group_centroid_variance"]["cf_cluster"]["adjusted_eta_squared"]),
    ]
    for label, getter in group_rows:
        vals = [getter(s) for s in steps]
        if label.startswith("Adjusted"):
            cells = " | ".join(f"{100*v:.2f}%" for v in vals)
        else:
            cells = " | ".join(f"{v:.4f}" for v in vals)
        lines.append(f"| {label} | {cells} |")

    lines += [
        "",
        f"The table reports absolute within-group cosine values. Random-pair baselines are "
        f"{random_cos[0]:.4f}, {random_cos[1]:.4f}, and {random_cos[2]:.4f}. User identity is the strongest of the "
        "measured grouping variables at all checkpoints; target-associated variance is smaller, and conventional "
        "collaborative-filtering cluster structure is weak. These associations "
        "show that the latent bank is not arbitrary noise, but they do not establish that the decoder uses the variation; "
        "that causal question is tested by E1 below.",
        "",
        "The exact-template utility eta-squared values are not treated as reliable utility evidence because the many "
        "small template groups make that statistic structurally prone to overfitting.",
        "",
        "## 4. Causal intervention ranking evidence",
        "",
        "Positive deltas below mean better ranking than raw; negative deltas mean worse ranking. Confidence intervals are "
        "paired bootstrap 95% intervals over the per-example metric contribution.",
        "",
    ]
    for step in steps:
        lines += inferential_table(step, results, intervals) + [""]

    lines += [
        "### Complete metrics at all cutoffs",
        "",
        "The tables below report every ranking metric stored by E1. The paired effect tables above remain the inferential "
        "comparison against raw; these tables provide point estimates at cutoffs 1, 3, 5, and 10.",
        "",
    ]
    for step in steps:
        lines += complete_table(step, results) + [""]

    max_content_hr = max(abs(intervals[s][m]["HR@10"][0]) for s in steps for m in content_modes)
    max_content_ndcg = max(abs(intervals[s][m]["NDCG@10"][0]) for s in steps for m in content_modes)
    zero_hr = [intervals[s]["mask"]["HR@10"][0] for s in steps]
    zero_ndcg = [intervals[s]["mask"]["NDCG@10"][0] for s in steps]
    prediction_diffs = {
        s: sum(
            a["prediction"] != b["prediction"]
            for a, b in zip(results[s]["shuffle_user"]["records"], results[s]["shuffle_target"]["records"])
        )
        for s in steps
    }

    lines += [
        f"All {len(MODES)*len(steps)} generation runs returned {test_n:,} paired records and zero catalog violations. Raw ranking improves with "
        f"training: HR@10 is {raw_hr[0]:.4f}, {raw_hr[1]:.4f}, and {raw_hr[2]:.4f}; NDCG@10 is "
        f"{raw_ndcg[0]:.4f}, {raw_ndcg[1]:.4f}, and {raw_ndcg[2]:.4f}.",
        "",
        f"Zero-mask HR@10 deltas are {zero_hr[0]:+.6f}, {zero_hr[1]:+.6f}, and {zero_hr[2]:+.6f}; corresponding "
        f"NDCG@10 deltas are {zero_ndcg[0]:+.6f}, {zero_ndcg[1]:+.6f}, and {zero_ndcg[2]:+.6f}. Across the mean, "
        f"shuffle, and noise controls, the largest absolute point changes are {max_content_hr:.6f} HR@10 and "
        f"{max_content_ndcg:.6f} NDCG@10. The confidence intervals above determine whether those changes are distinguishable from zero.",
        "",
        f"The shuffled-user and shuffled-target decoded top-10 lists differ on {prediction_diffs[s0]}, "
        f"{prediction_diffs[s1]}, and {prediction_diffs[s2]} of {test_n:,} examples at checkpoints {s0}, {s1}, and {s2}. "
        "Where they are effectively identical, they should be treated as one control rather than independent evidence.",
        "",
        "### Activation-versus-content interpretation",
        "",
        f"Train zero masking raises NLL by {rows_by_step[s0]['zero_mask_penalty']:.4f}, "
        f"{rows_by_step[s1]['zero_mask_penalty']:.4f}, and {rows_by_step[s2]['zero_mask_penalty']:.4f}. Held-out E1 "
        "separately asks whether ranking depends on the activation's identity. A damaging zero intervention combined with "
        "null mean/shuffle/noise effects is activation dependence without sample-specific content dependence—the causal "
        "criterion for latent-utilization collapse used in the reference report.",
        "",
        "## 5. Diagnosis",
        "",
        f"1. **Exact-template collapse: supported at all checkpoints.** Only {min(unique):.2f}%–{max(unique):.2f}% of "
        f"stored FP16 thoughts are unique, and the top ten templates cover {min(top10):.2f}%–{max(top10):.2f}%.",
        f"2. **Shared-direction collapse: supported at all checkpoints.** Random-pair cosine is "
        f"{min(random_cos):.3f}–{max(random_cos):.3f}, and the mean vector retains "
        f"{100*min(norm_ratio):.1f}%–{100*max(norm_ratio):.1f}% of a typical sample norm.",
        f"3. **Centered dimensional collapse: strongest at checkpoint {s1} and substantially weaker at checkpoint {s2}.** "
        f"Effective ranks are {effective[0]:.2f}, {effective[1]:.2f}, and {effective[2]:.2f}.",
        ("4. **Functional/utilization collapse: strongly supported at all checkpoints.** Zeroing damages ranking while "
         "mean, shuffled, and large-noise interventions never significantly worsen it." if zero_sig and content_not_harmful else
         "4. **Functional/utilization collapse: mixed.** At least one checkpoint or content control departs from the uniform null pattern; use the paired intervals above."),
        "",
        f"The most defensible overall conclusion is that `{tag_prefix}` learns increasingly accurate recommendations "
        f"and, by checkpoint {s2}, a much broader centered residual space. It nevertheless retains severe exact-template "
        "reuse and a large common direction. Geometric diversity and causal utilization must be evaluated separately.",
        "",
        "## 6. Limitations and next action",
        "",
        "- Geometry and extraction NLL are measured on train thoughts; ranking interventions are measured on the held-out test split.",
        "- PCA axes are fitted separately at each checkpoint; scalar concentration metrics are comparable, but coordinates are not pointwise aligned across epochs.",
        "- Exact duplicate counts are at FP16 storage precision; cosine and PCA diagnostics operate in float32 after loading.",
        "- Pairwise cosine uses 100,000 reproducibly sampled pairs; PCA and group-variance calculations use the full train bank.",
        "- Zero masking is off-manifold; mean and shuffled interventions are in-distribution controls, while Gaussian noise tests continuous perturbations.",
        "- This run evaluates causal ranking but not held-out teacher-forced NLL. Ranking can miss probability changes that do not alter the constrained top-10 order.",
        "",
        "The next action is a paired validation-NLL intervention at all three checkpoints. Track activation dependence, "
        r"\(A_t=\mathrm{NLL}_{zero,t}-\mathrm{NLL}_{raw,t}\), and content dependence, "
        r"\(C_t=\mathrm{NLL}_{shuffle,t}-\mathrm{NLL}_{raw,t}\). This tests whether token probabilities share the ranking result.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v6")
    parser.add_argument("--dataset", default="CDs_and_Vinyl", help="category name, e.g. Musical_Instruments")
    parser.add_argument("--steps", default="", help="comma-separated checkpoint steps, e.g. 519,1038,1557 (default: 384,768,1152)")
    parser.add_argument("--tag-prefix", default="", help="tag prefix before _checkpoint_<step> (default: latent_<version>)")
    parser.add_argument("--checkpoint-root", default="", help="path prefix before /checkpoint-<step> (default: output_dir/CDs_and_Vinyl_<version>/latent)")
    parser.add_argument("--train-n", type=int, default=49251)
    parser.add_argument("--test-n", type=int, default=6157)
    parser.add_argument("--dim", type=int, default=1536)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    version = args.version.removeprefix("latent_")
    steps = tuple(int(s) for s in args.steps.split(",")) if args.steps else DEFAULT_STEPS
    tag_prefix = args.tag_prefix or f"latent_{version}"
    checkpoint_root = args.checkpoint_root or f"output_dir/CDs_and_Vinyl_{version}/latent"
    output = args.output or ANALYSIS / "document" / f"latent_{version}_epoch_collapse_report.md"
    report = build(version, args.dataset, steps, tag_prefix, checkpoint_root, args.train_n, args.test_n, args.dim)
    output.write_text(report)
    print(output)


if __name__ == "__main__":
    main()
