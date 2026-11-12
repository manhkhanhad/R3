# Research log: collaborative / behaviour-grounded latent reasoning

Thesis framing: user-behaviour grounding for LLM recommendation. Working hypothesis for this line: **other samples' reasoning, paired with the behaviour it led to, can ground the current latent reasoning and improve recommendation.**

Code branch: `R3LatentCollaborativeReasoning` (worktree `.claude/worktrees/R3LatentCollaborativeReasoning`). Background: [Attempt 10](../../avoid_reasoning_collapse_and_shortcut_learning.md#attempt-10), [decoder insensitivity](../../decoder_insensitivity_to_latent.md), [outcome memory pilot](../../outcome_grounded_latent_memory.md), [grounded reasoning design](../../grounded_latent_reasoning.md).

## Status board

| ID | Experiment | Dataset | Job | Status | Output |
|---|---|---|---|---|---|
| L1 | Two-phase: margin (epoch 1) → margin + collaborative InfoNCE (epochs 2–3) | Musical Instruments | 322167 | **Done** | `output_dir/Musical_Instruments_R3_collab_margin/` |
| L2 | Same as L1 | CDs and Vinyl | 322174 | **Done** | `output_dir/CDs_and_Vinyl_R3_collab_margin/` |
| L3 | Behaviour-grounded two-step reasoning, smoke test | Musical Instruments (512 rows) | 323030 | **Done (pipeline check only)** | `output_dir/Musical_Instruments_R3_grounded_smoke/` |
| L4 | Behaviour-grounded two-step reasoning, full run + 5 donor controls | Musical Instruments | 323041 | **Done** 2026-10-15 01:36 (14h27m) | `output_dir/Musical_Instruments_R3_grounded/` |
| L5 | Same as L4 | CDs and Vinyl | 323042 | **Done** 2026-10-14 (7h21m), results below | `output_dir/CDs_and_Vinyl_R3_grounded/` |
| L6-smoke | Disentangled-interest reasoning, pipeline checks (margin on) | Musical Instruments (512 rows) | 323542, 323551 | **Done (found and fixed 2 design issues)** | `output_dir/Musical_Instruments_R3_intent_smoke{,2}/` |
| L6a | Disentangled-interest reasoning, K = 4, no margin | Musical Instruments | 323553 | **Done** 2026-10-15 (code `cc94b77`, bf16-frozen scales) | `output_dir/Musical_Instruments_R3_intent/` |
| L6b | Same as L6a + across-intent decoder margin | Musical Instruments | 323554 | **Done** 2026-10-15 15:51 (code `cc94b77`, bf16-frozen scales) | `output_dir/Musical_Instruments_R3_intent_margin/` |
| L6c | L6a + literature collapse fixes: MIMA Hungarian (3 later items), Re4 re-contrast (0.1), REMI routing reg (1.0) | Musical Instruments | 323697 | **Done** 2026-10-15 15:25 (code `cd96bd6`, bf16-frozen scales) | `output_dir/Musical_Instruments_R3_intent_fixes/` |
| L6d-valid | Phase 2 on L6a: DivReason path aggregation (router + redundancy suppression), router trained through frozen decoder on the **validation** split | Musical Instruments | 324862 | **Done** 2026-10-15 13:40 (2h12m) | `output_dir/Musical_Instruments_R3_intent_router_valid/` |
| L6d-train | Same, router fit on 8,312 **training** rows (memorised; same size as validation) | Musical Instruments | 324863 | **Done** 2026-10-15 13:39 (2h12m) | `output_dir/Musical_Instruments_R3_intent_router_train/` |
| L7-smoke | Latent-only bottleneck (RGLT-style), pipeline check | Musical Instruments (512 rows) | 324125 | **Done** 2026-10-14 (6 min; mask check passed) | `output_dir/Musical_Instruments_R3_latent_only_smoke/` |
| L7 | Latent-only bottleneck: target tokens see only the fixed instruction prefix, `r` and earlier target tokens | Musical Instruments | 325712 | **Done** 2026-10-16 09:41 (first attempt 324126 preempted; code `1aedd7e`) | `output_dir/Musical_Instruments_R3_latent_only/` |
| L8a | Attempt 10 + relative NLL queue gate (best 25% of last 1,024 samples; no token-accuracy gate) | Musical Instruments | 325727 | **Done** 2026-10-16 07:56 (code `413b4c0`) | `output_dir/Musical_Instruments_R3_collab_queue_gate/` |
| L8b | L8a + diversity-aware queue: per-user cap 2, per-target cap 4, same-user near-duplicate (cos ≥ 0.9) refresh | Musical Instruments | 325728 | **Done** 2026-10-16 01:53 (code `413b4c0`) | `output_dir/Musical_Instruments_R3_collab_queue_diverse/` |
| L9B | R3 stage 2 (noise-GRPO) on L8b + collaborative InfoNCE (L8b queue), λ auto-calibrated | Musical Instruments | 328106 (smoke 328105) | **Done** 2026-10-17 13:52 (15h57m; code `a9c1efa`) | `output_dir/Musical_Instruments_R3_grpo_musical_l9b_collab/` |
| L9C | Control: stock R3 stage 2 on plain R3 v1 stage-1 | Musical Instruments | 328084 (smoke 328082) | **Preempted** 2026-10-17 04:12 at step 1757/2078 (85%), not requeued; `checkpoint-1600` saved; needs resume (code `9e7f11d`) | `output_dir/Musical_Instruments_R3_grpo_musical_l9c_stock/` |
| L9B-lr1e-4 | L9B at lr 1e-4 (ranking only) | Musical Instruments | 330291 | **Done** 2026-10-18 14:37: **collapsed** (HR@10 0.0577) (code `5c56bf0`) | `output_dir/Musical_Instruments_R3_grpo_musical_l9b_collab_lr1e-4/` |
| L9B-lr5e-4 | L9B at lr 5e-4 (ranking only) | Musical Instruments | 330292 | **Done** 2026-10-18 07:22: **collapsed** (HR@10 0.0482) (code `5c56bf0`) | `output_dir/Musical_Instruments_R3_grpo_musical_l9b_collab_lr5e-4/` |
| L9C-r2 | L9C restarted from scratch, lr 1e-5 (ranking only) | Musical Instruments | 330293 | **Done** 2026-10-18 07:06: no-op vs R3 v1 (code `5c56bf0`) | `output_dir/Musical_Instruments_R3_grpo_musical_l9c_stock_lr1e-5/` |
| L10 | Behaviour-grounded reasoning (L4/L5 method) + 5 donor controls | Toys and Games | 332484 (after item-embedding job 332483) | **Done** 2026-10-20 01:53 (9h34m; grounded code `4671046`; embedding code `93f819c`) | `output_dir/Toys_and_Games_R3_grounded/` |
| A10-Toys-r2 | Attempt 10 rerun on Toys (collaborative InfoNCE, token-accuracy gate 1.0, 10k FIFO), ranking only | Toys and Games | 332796 | **Cancelled** 2026-10-19 23:20 after 39 min (`scancel`, SIGNAL Terminated; not by this session) (code `93f819c`) | `output_dir/Toys_and_Games_R3_collaborative_infonce_rerun_20261019/` |
| A10-Toys-acc05 | Attempt 10 on Toys with token-accuracy gate 0.5, ranking only | Toys and Games | 332797 | **Done** 2026-10-20 04:56 (code `93f819c`) | `output_dir/Toys_and_Games_R3_collaborative_infonce_acc05_20261019/` |
| A10-CDs-repro | Attempt 10 reproduction on CDs, original settings (gate 1.0, 10k FIFO), ranking only | CDs and Vinyl | 335891 | **Submitted** 2026-10-23 (code `93f819c`) | `scratchDirectory/R3/output_dir/CDs_and_Vinyl_R3_collaborative_infonce_repro_20261023/` |
| D1 | Queue-drift measurement: Attempt 10 gate 0.5, 2 epochs, checkpoint every 100 steps, drift analysis | Toys and Games | 338330 | **Done** 2026-10-25 21:41 (3h37m, 3090) (code `fbf53ca`) | `output_dir/Toys_and_Games_R3_collab_drift_toys_acc05/`, `analysis/results/drift/toys_acc05.json` |

Pending analyses: causal intervention suite (raw / mask / mean / shuffle) on L1 and L2 checkpoints.

## Results: all completed evaluations (updated 2026-10-14 21:15)

Computed with the repo's `src/utils/ranking_metrics.calculate_ranking_metrics` from the saved per-row predictions (Musical n = 8,313; CDs n = 6,157). Generation returns 10 beams, so **@20 = @10** for every run (no item ranks 11–20).

### Musical Instruments (test n = 8,313)

| run | mode | HR@1 | HR@3 | HR@5 | HR@10 | HR@20 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | two-phase (final) | 0.0766 | 0.0944 | 0.1044 | 0.1223 | 0.1223 | 0.0766 | 0.0869 | 0.0910 | 0.0968 | 0.0968 |
| L4 | grounded, knn | 0.0737 | 0.0901 | 0.1009 | 0.1203 | 0.1203 | 0.0737 | 0.0832 | 0.0876 | 0.0938 | 0.0938 |
| L4 | grounded, off | 0.0724 | 0.0885 | 0.0992 | 0.1205 | 0.1205 | 0.0724 | 0.0817 | 0.0861 | 0.0930 | 0.0930 |
| L4 | grounded, shuffle_outcome | 0.0734 | 0.0895 | 0.0996 | 0.1204 | 0.1204 | 0.0734 | 0.0827 | 0.0868 | 0.0936 | 0.0936 |
| L4 | grounded, latent_only | 0.0736 | 0.0895 | 0.1003 | 0.1195 | 0.1195 | 0.0736 | 0.0828 | 0.0872 | 0.0934 | 0.0934 |
| L4 | grounded, random | 0.0736 | 0.0894 | 0.0997 | 0.1198 | 0.1198 | 0.0736 | 0.0827 | 0.0870 | 0.0934 | 0.0934 |
| L6a | intent, mixture | 0.0724 | 0.0869 | 0.0953 | 0.1120 | 0.1120 | 0.0724 | 0.0807 | 0.0842 | 0.0895 | 0.0895 |
| L6a | intent, map | 0.0718 | 0.0854 | 0.0942 | 0.1101 | 0.1101 | 0.0718 | 0.0797 | 0.0832 | 0.0884 | 0.0884 |
| L6a | intent, oracle (uses true item; analysis only) | 0.0864 | 0.1173 | 0.1401 | 0.1778 | 0.1778 | 0.0864 | 0.1043 | 0.1137 | 0.1259 | 0.1259 |
| L6a | intent, random | 0.0463 | 0.0639 | 0.0722 | 0.0876 | 0.0876 | 0.0463 | 0.0564 | 0.0598 | 0.0648 | 0.0648 |
| L6a | intent, r0 (no intent) | 0.0573 | 0.0817 | 0.0912 | 0.1063 | 0.1063 | 0.0573 | 0.0720 | 0.0759 | 0.0808 | 0.0808 |
| L6b | intent + margin, mixture | 0.0669 | 0.0802 | 0.0900 | 0.1050 | 0.1050 | 0.0669 | 0.0745 | 0.0785 | 0.0834 | 0.0834 |
| L6b | intent + margin, map | 0.0668 | 0.0805 | 0.0899 | 0.1025 | 0.1025 | 0.0668 | 0.0746 | 0.0784 | 0.0825 | 0.0825 |
| L6b | intent + margin, oracle | 0.0828 | 0.1150 | 0.1399 | 0.1796 | 0.1796 | 0.0828 | 0.1012 | 0.1114 | 0.1243 | 0.1243 |
| L6b | intent + margin, random | 0.0208 | 0.0279 | 0.0338 | 0.0461 | 0.0461 | 0.0208 | 0.0248 | 0.0273 | 0.0312 | 0.0312 |
| L6b | intent + margin, r0 (no intent) | 0.0617 | 0.0733 | 0.0819 | 0.0936 | 0.0936 | 0.0617 | 0.0684 | 0.0720 | 0.0757 | 0.0757 |
| L6c | intent + literature fixes, mixture | 0.0695 | 0.0844 | 0.0938 | 0.1143 | 0.1143 | 0.0695 | 0.0781 | 0.0820 | 0.0885 | 0.0885 |
| L6c | intent + literature fixes, map | 0.0690 | 0.0834 | 0.0921 | 0.1072 | 0.1072 | 0.0690 | 0.0773 | 0.0809 | 0.0857 | 0.0857 |
| L6c | intent + literature fixes, oracle | 0.0829 | 0.1121 | 0.1341 | 0.1737 | 0.1737 | 0.0829 | 0.0997 | 0.1087 | 0.1216 | 0.1216 |
| L6c | intent + literature fixes, random | 0.0545 | 0.0719 | 0.0835 | 0.0977 | 0.0977 | 0.0545 | 0.0648 | 0.0695 | 0.0741 | 0.0741 |
| L6c | intent + literature fixes, r0 (no intent) | 0.0595 | 0.0840 | 0.0939 | 0.1140 | 0.1140 | 0.0595 | 0.0742 | 0.0783 | 0.0848 | 0.0848 |
| L6d-valid | router (fit on validation), aggregate | 0.0686 | 0.0812 | 0.0899 | 0.1051 | 0.1051 | 0.0686 | 0.0757 | 0.0792 | 0.0842 | 0.0842 |
| L6d-train | router (fit on train rows), aggregate | 0.0718 | 0.0849 | 0.0933 | 0.1096 | 0.1096 | 0.0718 | 0.0794 | 0.0829 | 0.0881 | 0.0881 |
| L6d | prior_blend (L6a paths, prior weights; no router) | 0.0709 | 0.0840 | 0.0923 | 0.1090 | 0.1090 | 0.0709 | 0.0783 | 0.0817 | 0.0871 | 0.0871 |
| L6d | aggregate_uniform (L6a paths, equal weights) | 0.0563 | 0.0734 | 0.0911 | 0.1072 | 0.1072 | 0.0563 | 0.0661 | 0.0735 | 0.0787 | 0.0787 |
| L7 | latent-only bottleneck (standard eval) | 0.0636 | 0.0788 | 0.0909 | 0.1085 | 0.1085 | 0.0636 | 0.0721 | 0.0770 | 0.0827 | 0.0827 |
| L8a | Attempt 10 + NLL queue gate | 0.0721 | 0.0895 | 0.1000 | 0.1217 | 0.1217 | 0.0721 | 0.0820 | 0.0863 | 0.0933 | 0.0933 |
| L8b | L8a + diversity-aware queue | 0.0749 | 0.0902 | 0.1026 | 0.1210 | 0.1210 | 0.0749 | 0.0838 | 0.0889 | 0.0949 | 0.0949 |

Paired bootstrap (2,000 resamples, seed 42), Δ HR@10 [95% CI] / Δ NDCG@10 [95% CI]:
- L4 knn − off: −0.0002 [−0.0025, +0.0020] / +0.0008 [−0.0004, +0.0020], n.s.
- L4 knn − shuffle_outcome: −0.0001 [−0.0013, +0.0010] / +0.0002 [−0.0003, +0.0008], n.s.
- L4 knn − latent_only: +0.0008 [−0.0006, +0.0023] / +0.0004 [−0.0003, +0.0012], n.s.
- L4 knn − random: +0.0005 [−0.0008, +0.0019] / +0.0004 [−0.0002, +0.0010], n.s.
- L6a mixture − L1: **−0.0103 [−0.0153, −0.0052] / −0.0072 [−0.0102, −0.0042]**, significant.
- L6a mixture − L4 off: −0.0085 [−0.0134, −0.0036] / −0.0035 [−0.0063, −0.0005], significant.
- L6a mixture − map: +0.0019 [+0.0005, +0.0034] / +0.0012 [+0.0004, +0.0020], significant.
- L6a mixture − random: +0.0244 [+0.0195, +0.0298] / +0.0247 [+0.0210, +0.0287], significant.

### CDs and Vinyl (test n = 6,157)

| run | mode | HR@1 | HR@3 | HR@5 | HR@10 | HR@20 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L2 | two-phase (final) | 0.0546 | 0.0794 | 0.0916 | 0.1156 | 0.1156 | 0.0546 | 0.0690 | 0.0740 | 0.0817 | 0.0817 |
| L5 | grounded, knn | 0.0542 | 0.0807 | 0.0960 | **0.1223** | 0.1223 | 0.0542 | 0.0697 | 0.0759 | **0.0843** | 0.0843 |
| L5 | grounded, off | 0.0538 | 0.0797 | 0.0957 | 0.1200 | 0.1200 | 0.0538 | 0.0690 | 0.0755 | 0.0833 | 0.0833 |
| L5 | grounded, shuffle_outcome | 0.0551 | 0.0799 | 0.0944 | 0.1200 | 0.1200 | 0.0551 | 0.0695 | 0.0754 | 0.0837 | 0.0837 |
| L5 | grounded, latent_only | 0.0542 | 0.0806 | 0.0950 | 0.1210 | 0.1210 | 0.0542 | 0.0695 | 0.0754 | 0.0838 | 0.0838 |
| L5 | grounded, random | 0.0542 | 0.0802 | 0.0952 | 0.1215 | 0.1215 | 0.0542 | 0.0693 | 0.0754 | 0.0839 | 0.0839 |

L4's other controls (off, shuffle_outcome, latent_only, random) are still running; L6a/L6b/L6c have no results yet.

Reading across cutoffs:
- **L5 (CDs):** knn's gain over `off` and `shuffle_outcome` appears at @3–@10 (HR@10 +0.0023; significance at @10 in the L5 section). At @1 it is absent: knn HR@1 0.0542 is slightly *below* `shuffle_outcome` (0.0551). Grounding on observed outcomes moves the target into the top 5–10, not to rank 1.
- **L4 (Musical, knn only):** below L1 at every cutoff (HR@1 0.0737 vs 0.0766; NDCG@10 0.0938 vs 0.0968) and at or below the Musical plain-R3 references (v0 HR@10 0.1209 / NDCG@10 0.0941; v1 0.1226 / 0.0950).
- **L1 vs L2:** both at baseline level at every cutoff; no cutoff shows a gain from the two-phase objective.

---

## 2026-10-13: L1, L2: two-phase margin → collaborative InfoNCE

**Question.** Does teaching the decoder to depend on the correct latent first (Attempt-2 margin) let Attempt-10's collaborative InfoNCE produce a latent the decoder actually uses?

**Setup.** Plain R3, Qwen2.5-1.5B-Instruct, 3 epochs, batch 128, both jobs on a 3090 (micro-batch 2).
- Epoch 1: CE + margin hinge (m = 0.2, weight 1.0). The FIFO queue fills, gated on correct predictions, but there is no InfoNCE.
- Epochs 2–3: CE + margin + user/target multi-positive InfoNCE (τ = 0.1, weights 0.1/0.1, queue 10k).
- Code: commits `9df0043`, `1a17d6c`; frozen copies in `output_dir/*_R3_collab_margin/code_*`.

**Ranking (full test).**

| run | HR@10 | NDCG@10 | reference |
|---|---:|---:|---|
| L1 Musical | 0.1223 | 0.0968 | Attempt 10: 0.1217 / 0.0939; plain R3 v1: 0.1226 / 0.0950 |
| L2 CDs | 0.1156 | 0.0817 | Attempt 10: 0.1166 / 0.0813; v4: 0.1152 / 0.0806 |

**Training observations.**
- The rising InfoNCE curve in W&B is a logging artefact. Anchors without a positive log 0, and the share of anchors with a positive rises over training (CDs user: 0.26 → 0.73). Per-anchor loss is flat on Musical (user ≈ 5.0, target ≈ 3.8, same as Attempt 10) and falls on CDs (user 4.95 → 4.32).
- CDs overfits in epoch 3: eval loss 1.257 → 1.072 → 1.293, train CE ≈ 0.08, margin loss falls to 0.11, gradient norm roughly doubles. The saved checkpoint is from epoch 3.

**Conclusion so far.** Ranking matches baseline on both datasets; no improvement. Whether the latent is used causally is untested (intervention suite pending).

## 2026-10-14: Diagnosis of the collaborative direction

Evidence gathered before designing L3–L5:
- **Attempt 10/11, two-phase:** `r = f(x_i)` gets no information from other users at inference, and the decoder already reads `x_i`. Better-organised `r` therefore carries nothing new (decoder-insensitivity H1).
- **Prototype grounding E4 (CDs):** mixing donor *latents* into `r` gave no gain over 3 seeds.
- **Outcome-memory pilot (`musical_margin2_memory2`):** donor *latent + observed next item* at inference. Outcome 0.1212 = shuffled outcome 0.1212 = random 0.1213 ≈ off 0.1202.
  - The gate learned to close: 0.44 → 0.003. The delivered correction was 0.005% of ‖r‖ (≈739).
  - Likely causes: (1) trained on top of a model that already fit the training targets, so evidence could not lower the loss; (2) a bounded additive correction to a large anchor vector.
- **Intervention-code note:** `shuffle_user` and `shuffle_target` in `experiment.py:645` pick the same pseudo-random latent. Neither compares a same-user swap with a different-user swap.
- **Related work:** IntuRec (arXiv 2606.27684) injects the model's own beam candidates into the first reasoning step and aligns it to the target item embedding. It gives no causal evidence that the decoder uses the injected state.

Working hypothesis refined: **donor reasoning helps only when paired with the donor's observed behaviour, delivered at inference, and trained before the model memorises the targets.**

## 2026-10-14: L3–L5: behaviour-grounded two-step latent reasoning (ideas 4 + 5)

**Design.** Full description in [grounded_latent_reasoning.md](../../grounded_latent_reasoning.md). Code commit `4671046`; frozen copy `output_dir/grounded_code/code_4671046`.
- Step 1: `r0` = R3 latent from own history.
- Step 2: retrieve 32 donors from other users whose next item happened before the query's latest history event. Each donor value = `V([r0_j; e(y_j)])`, with `e` = Qwen3 title embedding. Then `r1 = r0 + g·‖r0‖·dir(Σ a_j V_j)`, with one global learnable `g` (init 0.5), no gate, no cap.
- `r1` is supervised with full-catalogue item CE (weight 0.2).
- Retriever distilled (KL, weight 0.1) toward per-donor usefulness `u_j = log p(y_i | r1^(j)) − log p(y_i | r0)`, the gain in the query's *observed* next item.
- Trained jointly from Qwen2.5-1.5B with grounding on from step 0. Training donors: FIFO of 20k recent rows (no correctness gate). Inference donors: full train-split bank extracted from the final model.

**L3 smoke (job 323030, 512 rows, 1 epoch).**
- Losses finite; donors found for every anchor; best single-donor gain 0.12–0.16 nats.
- Bank filled (512) and reloaded; beam generation works.
- `knn` vs `off` top-10 lists differ on 12/16 test rows, so grounding is active at inference. Metrics on 16 rows are not meaningful.

**L4 / L5 (jobs 323041 / 323042; running).** 3 epochs, then full-test ranking under five controls on the same checkpoint:

| control | meaning | tests |
|---|---|---|
| `knn` | the method | — |
| `off` | `r1 = r0` | whether grounding helps at all |
| `shuffle_outcome` | same donors, outcomes permuted | whether behaviour is what helps (the thesis claim) |
| `latent_only` | same donors, outcomes zeroed | reasoning alone vs reasoning + behaviour |
| `random` | random eligible donors | the behaviour-trained retriever |

**Success criteria.**
1. `knn` > plain R3 (Musical HR@10 ≈ 0.1226; CDs ≈ 0.1152) and > `off`, with paired bootstrap CIs.
2. `shuffle_outcome` and `latent_only` both below `knn`.
3. `random` below `knn`.
4. Logs: `evidence_scale` stays away from 0; `item_hit10_r1 > item_hit10_r0`; `utility_top_gain > 0`.

**Results.** _Pending; fill in when jobs finish._

| run | knn | off | shuffle_outcome | latent_only | random |
|---|---|---|---|---|---|
| L4 Musical HR@10 / NDCG@10 | 0.1203 / 0.0938 | _running_ | | | |
| L5 CDs HR@10 / NDCG@10 | **0.1223 / 0.0843** | 0.1200 / 0.0833 | 0.1200 / 0.0837 | 0.1210 / 0.0838 | 0.1215 / 0.0839 |

**L5 (CDs) result, 2026-10-14.** n = 6,157; paired bootstrap, 2,000 resamples, seed 42; Δ = knn − control.

| comparison | ΔHR@10 [95% CI] | ΔNDCG@10 [95% CI] | significant |
|---|---|---|---|
| knn − off | +0.0023 [+0.0002, +0.0042] | +0.0010 [+0.0001, +0.0019] | yes, both |
| knn − shuffle_outcome | +0.0023 [+0.0010, +0.0037] | +0.0006 [+0.0001, +0.0012] | yes, both |
| knn − latent_only | +0.0013 [+0.0000, +0.0026] | +0.0005 [−0.0001, +0.0011] | no |
| knn − random | +0.0008 [−0.0005, +0.0021] | +0.0004 [−0.0003, +0.0011] | no |
| knn − L2 checkpoint (two-phase, separate training) | +0.0067 [+0.0010, +0.0123] | +0.0026 [−0.0010, +0.0062] | HR only |
| off − L2 checkpoint | +0.0044 [−0.0013, +0.0101] | +0.0016 [−0.0018, +0.0051] | no |

Interpretation:
- **First significant evidence for the behaviour-grounding claim.** Grounding at inference helps (knn > off), and it is the donors' *observed outcomes* that help: permuting outcomes among the same donors removes the whole gain (knn > shuffle_outcome, and shuffle_outcome = off on HR@10).
- The effect is small (+0.0023 HR@10, ~2% relative). knn is above every earlier CDs plain-R3 reference (v4 0.1152, Attempt 10 0.1166, L2 0.1156); vs L2 it is significant on HR@10 only.
- Not established: that the *retriever* matters (knn vs random n.s.) or that outcomes beat latent-only donors (borderline). Part of the gap to earlier runs comes from training with grounding (off is 0.1200), not from inference-time donors, and that part is not significant vs L2.
- Training diagnostics: `item_hit10_r1` = `item_hit10_r0` (0.15) by epoch 3, so grounding stops helping the item scorer on memorised training rows (the predicted risk). Retrieval KL falls to 0.07 (retriever fits the usefulness teacher). Eval loss 1.254 → 1.062 → 1.254 (the usual CDs epoch-3 overfit; the saved model is epoch 3).
- **Implementation caveat (bf16, affects L4 and L5 alike; same code `4671046`).** Training is pure bf16. Near 0.5–1.0 the bf16 spacing (≈0.004–0.008) exceeds an Adam step at lr 3e-4, so updates to parameters initialised there round away.
  - `evidence_scale` stayed at exactly 0.5 for all logged steps (L5: 1,156; L4: 1,561).
  - In the saved L5 checkpoint every LayerNorm scale in the new modules (`query`, `key`, `value`, `item_head`) is still exactly 1.0. Biases and Linear weights (initialised near 0) did train.
  - So L4/L5 are "grounding at a **fixed** 50% of ‖r0‖ with non-learned LayerNorm scales", not a learned grounding strength. The same affects `path_scale` and LayerNorm scales in L6.
- **Consequence for the controls.** Comparisons between controls on the same checkpoint stay valid. But `off` feeds the decoder `r0` alone, which it never saw in training (it always saw `r0` + a 50%-norm component), so knn > off partly reflects that unfamiliar input. **`shuffle_outcome` (same donors, same magnitude, outcomes permuted) is the clean test** and is significant on CDs; treat knn > off as the weaker claim.
- **Fix for future runs:** keep the new modules in fp32 while the base stays bf16 (the module forward already casts inputs to the module's weight dtype).

How to collect: `grep -h "eval_.*json:" run_logs/R3_grounded_{Musical.323041,CDs.323042}.out`; training diagnostics: `grep -o "'evidence_scale': [0-9.]*" run_logs/R3_grounded_Musical.323041.out | tail`.

**Known risks.**
- Later epochs may still memorise targets. Watch whether `evidence_scale` or the `knn`–`off` gap shrinks; the next step would be cross-fitting.
- Usefulness is measured with the step-2 item scorer, not a decoder pass.
- Train FIFO donors and the test bank come from slightly different encoders.

---

## 2026-10-14: L6: disentangled-interest latent reasoning (new direction)

Independent of L4/L5; priority is to make disentangled interests work, with no requirement to tie in collaborative filtering or behaviour grounding yet.

**Motivation.** The same history supports several interests that lead to different items, and only one explains the item the user actually chose. R3's `r = f(x)` is a single deterministic vector trained by CE on the realised item, so it can only average those interests into a generic context summary. That fits the observed collapse and mean/shuffle insensitivity. No earlier attempt conditions the reasoning on the item: Attempt 7's VAE used `mu, z = encoder(x)` with no sampling and no item-aware posterior.

**Model** (`src/latent/intent_reasoning.py`; final code commit `cc94b77`, frozen copy `output_dir/intent_code/code_cc94b77`; the first version `f59b20d` is superseded, see smoke findings).
- `r0` = R3 pooling. K = 4 intent paths `r_k = r0 + g·‖r0‖·normalize(U_k·trunk(r0) + b_k)`: a shared trunk with one output head per intent, and one global learnable `g` (init 0.5).
- Prior `p(k | x) = softmax(MLP(r0))`, used at inference.
- Credit (posterior) `q(k | x, y) ∝ p_item(y | r_k)`, training only: the path that best explains the observed item. `p_item` is a cheap scorer over the 5,030 Qwen3 item embeddings (τ = 0.07).
- Training is winner-takes-all. The decoder sees only `r_{k*}`, `k* = argmax q` (sampled from `q` for the first 0.1 epoch). Loss = `CE + 0.2·mixture item NLL (−log Σ_k p(k|x)·p_item(y|r_k)) + 0.1·CE(prior, k*) + 0.01·KL(EMA intent usage ‖ uniform)`. Cost: one decoder pass, about the same as plain R3.
- L6b adds `1.0·hinge(0.2 + NLL(r_{k*}) − NLL(r_{other intent}))`, a second decoder pass forcing the decoder to prefer the credited intent.

**Evaluation modes** (`INTENT_MODE`, same checkpoint):

| mode | meaning | tests |
|---|---|---|
| `mixture` | one beam search per intent, merged by `log p(k\|x) + seq score` | the method |
| `map` | most likely intent under the prior, one pass | whether merging is needed |
| `oracle` | posterior intent using the true test item (analysis only) | upper bound: value of picking the correct reasoning path |
| `random` | random intent | whether the prior matters |
| `r0` | no intent | whether intent reasoning is used at all |

Mixture eval also stores each intent's top-10 (`intent_predict`) for swap and diversity analysis.

**Training diagnostics to watch.** `usage_k` (collapse onto one intent?), `posterior_entropy`, `prior_posterior_agree`, `path_pair_cos`, `path_scale`, `item_hit10_{r0,selected,prior_map}`.

**Success criteria.**
1. `mixture` > plain R3 v1 (HR@10 0.1226 / NDCG@10 0.0950) and > `r0`, with paired CIs.
2. Intents are distinct: low overlap between per-intent top-10s; all intents used.
3. `oracle` ≫ `map`/`random` (paths carry different, item-relevant reasoning). The gap between `oracle` and `mixture` shows how much better intent prediction could gain.

**Smoke findings (512 rows, 1 epoch; pipeline checks, not results).**

| check | v1 `f59b20d` (job 323542) | v2 `e629c9e` (job 323551) |
|---|---|---|
| path pairwise cosine | 0.99 | −0.003 |
| mean top-10 overlap between intents | 0.94 | 0.69 |
| `map` vs `random` top-10 lists differ | 6/16 | 10/16 |
| prior–posterior agreement | — | 1.0 |

- **Fix 1 (`e629c9e`).** v1 built paths with one MLP over `[LN(r0) (1536-d); c_k (256-d)]`. The history latent dominated, so all paths were the same and the across-intent margin sat exactly at its 0.2 floor (the decoder saw no difference). Replaced with a shared trunk and per-intent output heads, which start near-orthogonal.
- **Fix 2 (`cc94b77`).** With the posterior `∝ prior × p_item`, a confident prior can confirm its own choice regardless of the item (agreement 1.0, `oracle` = `map` on all rows). Credit now uses item likelihood only, matching the idea that the observed item determines which reasoning path is correct. The prior stays in the mixture loss and at inference. Flag `posterior_uses_prior` restores the old rule.

**Watch in L6a/L6b logs.** `prior_posterior_agree` should fall below 1 (credit driven by the item); `usage_k` should stay spread; `path_pair_cos` should stay low; `item_hit10_selected > item_hit10_r0`.

**Results.** _Pending; fill in when 323553 / 323554 finish._

| run | mixture | map | oracle | random | r0 | intent top-10 overlap |
|---|---|---|---|---|---|---|
| L6a (no margin) HR@10 / NDCG@10 | | | | | | |
| L6b (margin) HR@10 / NDCG@10 | | | | | | |
| L6c (literature fixes) HR@10 / NDCG@10 | | | | | | |

How to collect: `grep -h "eval_.*json:" run_logs/R3_intent_Musical.323553.out run_logs/R3_intent_margin_Musical.323554.out run_logs/R3_intent_fixes_Musical.323697.out`; overlap: `intent_predict` field in `eval_mixture.json`.

## 2026-10-14: L6c: interest-collapse fixes from the multi-interest literature

Identical paths ("interest collapse") are a known failure in multi-interest recommendation. Fixes reviewed and how each was adapted (commit `cd96bd6`; all training-only, flags default off):

| paper | diagnosis | fix in paper | adaptation here | flag |
|---|---|---|---|---|
| **MIMA**, Mao et al., arXiv 2609.12842 (2026) | single-positive training: the best-matching interest absorbs every target, others get no supervision | multiple co-occurring positives; Hungarian matching gives each positive a distinct interest | positives = target + the user's next ≤3 distinct items (training labels only); Hungarian on `−log p_item(y \| r_k)`; decoder trains on the path matched to the target | `ASSIGNMENT=hungarian N_LATER=3` |
| **Re4**, Zhang et al., WWW 2022 (arXiv 2208.08011) | without regularisation interests are neither distinct nor reflect representative history items | Re-contrast: InfoNCE pulling each interest to its attended items (weight > 1/n), pushing the other items, the other interests and sampled items (τ = 0.02) | each path routes over history items in item space; same InfoNCE with τ = 0.1, 64 random catalogue negatives | `RECONTRAST_WEIGHT=0.1` |
| **REMI**, Xie et al., RecSys 2023 (arXiv 2302.14532) | routing collapse: an interest draws on only one item; easy negatives from uniform sampling | routing regularisation `‖diag((A−Ā)ᵀ(A−Ā))‖²` (λ = 100); interest-aware hard-negative mining | routing regularisation on the paths' history routing | `ROUTING_REG_WEIGHT=1.0` |

Not adopted: REMI's hard-negative mining (it corrects sampled softmax; the item scorer here already uses a full softmax over 5,030 items). Also noted: DRIM adds a diversity separator between interest vectors, a simpler form of the same idea as Re-contrast.

Data support (train split): 84% (Musical) / 86% (CDs) of rows have ≥2 history items and ≥1 later same-user item; 53% / 60% have ≥3 later items.

**Smoke 3 (job 323686, 512 rows, 1 epoch, all fixes; pipeline check).** `oracle` vs `map` top-10 lists differ on 9/16 rows (0/16 before), `mixture` vs `map` 9/16 (1/16 before), prior–credit agreement 0.0 at the last step (1.0 before), 2.5 positives per row, path cosine 0.018, intent top-10 overlap 0.79. REMI's term is near zero (routing max 0.30, near uniform); it only acts if a path starts collapsing onto one item.

**L6c (job 323697).** All three fixes, no margin. Compare with L6a (no fixes) for the effect of the fixes, and L6b (margin) for the decoder-side alternative.

Sources: [MIMA](https://papers.cool/arxiv/2609.12842), [Re4](https://arxiv.org/abs/2208.08011), [REMI](https://arxiv.org/abs/2302.14532), [REMI code](https://github.com/tokkiu/remi), [DRIM](https://astro.paperswithcode.com/paper/diversity-regularized-interests-modeling-for).


## 2026-10-14 21:10: progress snapshot

**L4 (Musical grounded), partial.** Training finished (eval loss 0.578 → 0.413 → 0.404; no epoch-3 overfit, unlike CDs). Bank: 66,500 rows. `knn` = HR@10 0.1203 / NDCG@10 0.0938 (n = 8,313). That is at or slightly below the Musical plain-R3 references (v0 0.1209 / 0.0941, v1 0.1226 / 0.0950) and L1 (0.1223 / 0.0968), so unlike CDs there is no gain over baselines so far. The decisive test is still `knn` vs `shuffle_outcome` on the same checkpoint, pending. Training diagnostic: `item_hit10_r1` (0.20) < `item_hit10_r0` (0.23) over the last 50 steps, so grounding did not help the item scorer on memorised training rows by epoch 3.

**L6a (intents, no fixes), epoch 1.88/3, mean of last 50 steps.**
- Intent usage 0.29 / 0.42 / 0.24 / 0.15: all used, intent 1 growing.
- `path_pair_cos` 0.08: paths stay distinct.
- Prior–credit agreement 0.48 (above chance 0.25, not collapsed).
- Item hit@10: credited path 0.16, `r0` 0.14, **prior's chosen path 0.10**. The credited path is only slightly better than `r0`, and the prior's pick is worse than `r0`: paths are distinct but not yet specialised in a way the history can predict.
- Eval loss after epoch 1: 0.575.

**L6b (intents + margin), epoch 0.6/3.** Usage 0.22 / 0.19 / 0.33 / 0.29; `path_pair_cos` −0.11 (more spread than L6a); prior–credit agreement 0.44; item hit@10 credited 0.12, `r0` 0.10, prior 0.08. Too early to compare.

**L6c (literature fixes):** still pending in the queue.

Note: L6a/L6b were launched before the bf16 finding, so their `path_scale` and LayerNorm scales are frozen (same as L4/L5). L6c (code `cd96bd6`) has the same issue; the fp32 fix is not yet applied (decision pending).


## 2026-10-15 08:51: L4 final; L6a results

**L4 (Musical grounded): no effect.** All five controls are within ±0.001 HR@10 of each other; none differs significantly from knn (table above). **The CDs result (L5: knn > shuffle_outcome) does not replicate on Musical.** The behaviour-grounding evidence is therefore one dataset, small (+0.0023 HR@10), and not replicated. knn (0.1203) is also at or slightly below the Musical plain-R3 references (v0 0.1209, v1 0.1226) and L1 (0.1223).

**L6a (disentangled intents, no fixes): the paths work; choosing among them does not.**
- **Distinct, used reasoning paths.** A random intent drops HR@10 to 0.0876 (−0.0244 vs mixture, significant). This is the first time in the investigation that swapping the latent changes ranking substantially (earlier swaps changed HR@10 by ≤0.003).
- **Large oracle headroom.** Choosing the path with the true item gives HR@10 0.1778 / NDCG@10 0.1259 (+45% / +33% over plain R3 v1). Caveat: best-of-4 selection with the label inflates any set of 4 variants. The needed control is best-of-4 over random perturbations of `r0` to separate disentanglement from selection; the very low `random` score suggests the paths really differ.
- **The prior is the bottleneck.** History alone cannot pick the right intent: mixture 0.1120 and map 0.1101 are significantly below L1 (−0.0103 HR@10) and L4 off. Merging intents beats the prior's single pick (+0.0019) but does not close the gap.
- Training: eval loss 0.575 → 0.461 → 0.470. In the last 50 steps the credited path's item hit@10 was 0.25 vs `r0` 0.20 vs the prior's pick 0.22. Usage figures are rough (bf16 usage-EMA drift: shares sum to ≈1.17).

**L6b / L6c:** training done (final eval loss 0.561 and **0.444**, vs L6a 0.470); `mixture` evaluation running (~3 h per mode).

**Next questions for this direction.**
1. Oracle control: best-of-4 over random perturbations of `r0` at matched magnitude.
2. Improve intent prediction from history: stronger prior weight (currently 0.1), richer prior input, or score aggregation (expected score over intents) instead of max.
3. Whether L6c's literature fixes improve mixture/map (pending).


## 2026-10-15: L6d: DivReason-style path aggregation (phase 2)

**Source.** An et al., "Beyond the Single Path: Divergent Reasoning for LLM-based Recommendation" (DivReason), SIGIR 2026, doi 10.1145/3805712.3809732. Same setting as ours: LatentR³ latent reasoning, Qwen2.5-1.5B, same data protocol. Their Instruments split matches ours exactly (66,500 / 8,312 / 8,313 rows, 5,030 items).
- Paths are training-free perturbations of one reasoning (MC dropout; directional perturbation along normalised k-means centroids of hidden states).
- Reasoning Path Aggregation: router `s_n = MLP(r_n)`; redundancy suppression `s'_n = s_n − β Σ_{j≠n} cos(r_n, r_j)`; threshold pruning; `softmax(s')`; `r_agg = Σ α_n r_n`.
- Training: SFT through the decoder on `r_agg`, then alternating accuracy/diversity GRPO with the backbone frozen. Ablation: path-sum, path-cat < no suppression < full.
- Their Instruments headers appear mislabelled (N@K > H@K). Reading the last four columns as N@5, N@10, H@5, H@10: LatentR³ H@10 0.1229 / N@10 0.0973, DivReason **0.1301 / 0.1071**. To verify against their code.

**What we take, and why.** L6a's bottleneck is choosing the path: oracle 0.1778 vs `map` 0.1101. Our prior only imitates the credited path; DivReason's router is trained through the recommendation loss. Blending is a risk for our distinct interests, so the softmax is sharp (τ = 0.5) and pruned.

**Design** (code `3204ab8`, frozen copy `output_dir/intent_code/code_3204ab8`).
- Start from the L6a checkpoint (paths, prior, decoder frozen).
- Router: LayerNorm → Linear(1536, 256) → GELU → Linear(256, 1) per path, in fp32 (bf16 fix).
- Redundancy suppression: cosine between path **offsets** `r_k − r0` (raw cosine ≈ 1 because all paths share `r0`); learnable β (log-parameterised, init 0.5).
- Weights `softmax(s'/0.5)`; at eval, weights < 0.05 are pruned, always keeping the top path.
- Trained with the frozen decoder's recommendation CE for 3 epochs, lr 1e-3, batch 128.
- Fit data:
  - **L6d-valid**: validation split, 90% fit / 10% held out for eval loss. Rows are unseen and later in time, like test.
  - **L6d-train**: 8,312 training rows (memorised), same size. Isolates whether fitting the selector on memorised rows fails.
- Eval modes (single pass): `aggregate` (router), `prior_blend` (prior weights, no router), `aggregate_uniform` (equal weights; DivReason's path-sum ablation). L6a's `map`, `mixture`, `oracle`, `r0` above are the references on the same paths and decoder.

**Smoke (job 324861, 231 fit rows, 1 epoch, 16 test rows).** Pipeline OK: 396,802 trainable parameters, gradient only into the router, checkpoint reload OK. All three modes give different top-10 lists on 16/16 rows. Router weights still soft (max ≈ 0.53, entropy 0.95 of max 1.39); watch whether training sharpens them, since the decoder never saw blended paths in phase 1.

**Success criteria.** `aggregate` > L6a `map` (0.1101) and `mixture` (0.1120), ideally ≥ plain R3 (≈ 0.122); `aggregate` > `prior_blend` (the router adds information over the prior); L6d-valid > L6d-train.

**Results (2026-10-15 13:45).** HR/NDCG at all cutoffs in the Musical table above. `prior_blend` and `aggregate_uniform` do not use the router, so the two jobs give identical values for them.

| mode | HR@10 | NDCG@10 | vs L6a `map` (HR@10 / NDCG@10, paired bootstrap) |
|---|---:|---:|---|
| L6a `mixture` (reference) | 0.1120 | 0.0895 | +0.0019 / +0.0012, significant |
| L6a `map` (reference) | 0.1101 | 0.0884 | — |
| L6d-train `aggregate` | 0.1096 | 0.0881 | −0.0005 / −0.0002, n.s. |
| `prior_blend` | 0.1090 | 0.0871 | −0.0011 / −0.0013, n.s. |
| `aggregate_uniform` | 0.1072 | 0.0787 | −0.0029 n.s. / **−0.0097 significant** |
| L6d-valid `aggregate` | 0.1051 | 0.0842 | **−0.0049 / −0.0042, significant** |

**Interpretation.**
- **A selector trained through the decoder does not beat the prior.** Neither router improves on L6a `map`; the validation-fit router is significantly worse. The gap to the oracle (0.1778) is untouched.
- **The router barely learns.** Validation-fit held-out loss 0.4728 → 0.4722 → 0.4716 over 3 epochs; β ≈ 0.498 (moved from 0.5, so the fp32 fix works, but the gradient is tiny). The validation-fit router ended with near one-hot weights on its last logged batch; the train-fit router stayed soft (max weight 0.47, entropy 1.10). Hypothesis "fit on unseen rows > fit on memorised rows" is not supported (train-fit 0.1096 > valid-fit 0.1051).
- **Blending hurts NDCG.** Equal weights keep HR@10 but lose NDCG@10 (−0.0097) and HR@1 (0.0563 vs 0.0718): blended paths blur the top of the list, consistent with the decoder never seeing blends in phase 1.
- **Same picture across L6 variants.** Oracle 0.1737–0.1796 for L6a/b/c, while map/mixture stay 0.1025–0.1143, all below plain R3 (≈ 0.122). The interests are separable given the item, but **the information to pick the right one does not seem to be available from history alone**, whether the selector is a prior, a mixture, or a decoder-trained router.
- Caveat still open: the oracle is best-of-K with the label and needs the perturbation control to separate disentanglement from selection.

**What this suggests next.**
1. Run the oracle control (best-of-4 over directional perturbations of `r0`) before investing more in selection. If perturbations also reach ≈ 0.17, the "headroom" is mostly selection with the label, not disentanglement.
2. If the headroom is real: give the selector information beyond `r0` (e.g. per-path decoder confidence, or the history items each path attends to), or train intents so the prior's choice is the training target from the start (sample by prior instead of credit).


## 2026-10-15 15:47: L6c final; L6b nearly done

**L6c (intent + MIMA / Re4 / REMI fixes), all modes done.** Paired bootstrap vs L6c `mixture` (HR@10 0.1143 / NDCG@10 0.0885):

| comparison | ΔHR@10 [95% CI] | ΔNDCG@10 [95% CI] |
|---|---|---|
| mixture − r0 (same model, no intent) | +0.0002 [−0.0043, +0.0051] n.s. | **+0.0037 [+0.0011, +0.0065]** |
| mixture − map | **+0.0071 [+0.0043, +0.0099]** | **+0.0028 [+0.0017, +0.0040]** |
| mixture − L6a mixture | +0.0023 [−0.0028, +0.0073] n.s. | −0.0010 [−0.0039, +0.0020] n.s. |
| mixture − L1 | **−0.0081 [−0.0131, −0.0030]** | **−0.0082 [−0.0111, −0.0053]** |

- The literature fixes do **not** significantly improve on L6a (mixture +0.0023 HR@10, n.s.).
- In L6c, using intents at all adds nothing at HR@10 over the model's own `r0` (0.1143 vs 0.1140), only a small NDCG gain. Its `r0` (0.1140) is far better than L6a's `r0` (0.1063).
- L6c's intents are less decisive than L6a's: random intent 0.0977 (L6a 0.0876), oracle 0.1737 (L6a 0.1778). Consistent with the literature fixes making paths more uniformly usable but less specialised.
- All deployable L6 modes remain significantly below plain-R3-level references (L1 0.1223).

**L6b (margin):** random intent collapses to HR@10 **0.0461**, the strongest intent dependence of any run (the across-intent margin trains the decoder to be very sensitive to which path it gets). Oracle 0.1796 (highest) but map 0.1025 / mixture 0.1050 (lowest): sharper paths make wrong choices costlier. `r0` mode still running.

**Overall L6 picture.** Oracle 0.174–0.180 in every variant; every deployable selection (map, mixture, prior blend, decoder-trained router) 0.102–0.114, below plain R3 ≈ 0.122. The more the decoder depends on the chosen path, the higher the oracle and the lower the deployable score. Selection from history is the binding constraint. The best-of-K perturbation control is still needed to quantify how much of the oracle reflects genuine disentanglement.

## 2026-10-15: Why routing fails (diagnosis on L6a per-intent lists)

Script: `intent_diagnosis.py` (job tmp; uses `eval_mixture.json` `intent_predict`, SASRec cluster20 labels). Test n = 8,313.

| measurement | value |
|---|---|
| mean top-10 overlap between two intents | 0.083 (lists nearly disjoint, ≈40 distinct candidates) |
| rows where 0 / 1 / 2 / 3 / 4 intents contain the target | 81.3% / 11.4% / 1.9% / 1.3% / 4.1% |
| union of intents' top-10s contains target ("any hits") | **0.1867** (≈ oracle 0.1778) |
| winning intent predicted from the *target's* SASRec cluster | 0.569 (chance 0.25) |
| model's own top-1 prediction in the target's cluster (history only) | 0.225 (chance 0.05) |
| intent vs recommended-item cluster, NMI | 0.064 (intents are not simply SASRec item regions) |

Interpretation:
1. The oracle is close to HR@10 over a ~40-candidate pool chosen with the label (any-hit 0.187), not evidence of reachable headroom.
2. Correct routing = locating the target among ~40 candidates; the winning intent depends on the target (0.57 from its cluster), and history alone locates the target's region only 22.5% of the time. The router sees the same information as the decoder, so prior / mixture / decoder-trained router plateau at 0.10–0.11.
3. Router training signal through the frozen decoder is weak (L6d held-out loss 0.4728 → 0.4716).
4. Committing to one list discards the other ~30 candidates; mixture beats every single-intent selection.

Next (proposed): (a) plain-R3 HR@40 with 40 beams vs intent union 0.187, to test whether intents add candidates beyond a wider beam; (b) item-level reranking of the pooled candidates with features outside the LLM (per-intent decoder scores, SASRec score, popularity, recency); (c) sum instead of max in the mixture merge.

## 2026-10-15: Offline merge rules over saved per-intent lists (no new generation)

Script: `scripts/consensus_rerank.py` (scores each pooled candidate from the saved per-intent top-10s only).

| merge rule | L6a HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | L6c HR@10 | L6c NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| saved `mixture` (max of prior + seq score) | 0.0724 | 0.0869 | 0.0953 | 0.1120 | 0.0724 | 0.0807 | 0.0842 | 0.0895 | 0.1143 | 0.0885 |
| Borda (sum of 10 − rank) | 0.0576 | 0.0778 | 0.0924 | 0.1195 | 0.0576 | 0.0694 | 0.0754 | 0.0843 | 0.1201 | 0.0860 |
| count of intents + best rank | 0.0551 | 0.0780 | 0.0902 | 0.1196 | 0.0551 | 0.0684 | 0.0734 | 0.0828 | 0.1184 | 0.0855 |
| reciprocal-rank fusion | 0.0562 | 0.0767 | 0.0897 | 0.1195 | 0.0562 | 0.0681 | 0.0734 | 0.0829 | 0.1184 | 0.0848 |
| pool recall (any intent lists target) | | | | 0.1867 | | | | | 0.1849 | |

Consensus raises HR@10 by +0.006–0.008 (L6a 0.1120 → 0.1195; L6c 0.1143 → 0.1201) but loses the top of the list (HR@1 0.072 → 0.058). Max-merge keeps per-intent confidence (good at rank 1); consensus keeps agreement (good for recall). Next: the proper mixture probability (sum over intents of prior × p(y | x, r_k)), which needs per-intent scores saved by the eval; optionally score every pooled candidate under every intent; include the `r0` path in the pool.

## 2026-10-14: L7: latent-only structural bottleneck (RGLT-style)

**Question.** The decoder-insensitivity study shows R3's decoder ignores `r`'s content because `r = f(x)` is redundant with the history it also reads. If the history is hidden from the target, so that all user information must pass through `r`, does the decoder use `r`'s content, and what does ranking cost?

**Inspiration.** RGLT (arXiv 2608.14107): the final latent state is the only thing used for the prediction, so the latent cannot be bypassed. RGLT also finds effect-based supervision beats state alignment (BRIGHT nDCG@10 34.20 vs 30.64).

**Design** (`src/latent/latent_bottleneck.py`, flag `--latent_only`, default off; commit `1aedd7e`, frozen copy `output_dir/latent_only_code/code_1aedd7e`):
- Every query from the thought position onward sees only: the fixed 56-token instruction prefix (identical for all users), the thought token `r`, and earlier target tokens. The "### User Input: …titles… ### Response:" span is hidden.
- The same 4D mask is used in training, teacher-forced scoring and generation; under KV-cache decoding and constrained beam search the first-pass mask is cached and extended. Position ids are recomputed over visible tokens, so the target cannot infer history length.
- `r` is still pooled from the history exactly as in R3. One latent token (the codebase produces one vector per row; RGLT uses 16). No new trainable modules.
- Recipe = plain R3 v1 (Qwen2.5-1.5B-Instruct, 3 epochs, batch 128, lr 3e-4, cutoff 512).
- Verification: unit tests (target logits identical across histories with `r` fixed; with the flag off they change; changing `r` changes them; cached and uncached beam search agree). In each job, `latent_bottleneck_check.py` fails the job if a history swap with `r` fixed moves target logits by > 1e-2.

**Controls (same checkpoint).** Standard constrained 10-beam ranking; interventions raw / mask / mean / shuffle_user / shuffle_target (`analysis/results/intervention/musical_instruments_latent_only/`); paired bootstrap raw vs each.

**Smoke (job 324125, 512 training rows, 1 epoch, 64 test rows).** Pipeline works end to end. Mask check: history swap with `r` fixed, max |Δlogit| = 4.8e-4 (mean 7.4e-5), so the history does not leak; `r` swap with history fixed, mean max |Δlogit| = 10.9, so the target depends on `r`. Ranking 0 at every cutoff on 64 rows after 512 training rows (not meaningful).

**Full run.** Job 324126 started 2026-10-14 23:19 and was preempted after 2.5 min without requeue; resubmitted 2026-10-15 as job 325712 (`--requeue`; same frozen code). Expected ~14–18 h.

**Success criteria.** (a) Ranking vs plain R3 v1 (HR@10 0.1226 / NDCG@10 0.0950) and plain SFT (0.1009 / 0.0790); a drop vs R3 is plausible because the decoder loses direct access to the titles. (b) `mean` and `shuffle_*` significantly below `raw`.

**Results (2026-10-16).** Standard ranking (`eval_raw.json`, n = 8,313): HR@1 0.0636, HR@3 0.0788, HR@5 0.0909, HR@10 **0.1085**, HR@20 0.1085; NDCG@1 0.0636, NDCG@3 0.0721, NDCG@5 0.0770, NDCG@10 **0.0827**, NDCG@20 0.0827.

Interventions (E1 pipeline, same checkpoint):

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0636 | 0.0788 | 0.0917 | 0.1089 | 0.0636 | 0.0720 | 0.0773 | 0.0828 |
| mask | 0.0047 | 0.0142 | 0.0180 | 0.0268 | 0.0047 | 0.0104 | 0.0120 | 0.0148 |
| mean | 0.0008 | 0.0089 | 0.0156 | 0.0295 | 0.0008 | 0.0049 | 0.0076 | 0.0122 |
| shuffle_user | 0.0049 | 0.0102 | 0.0148 | 0.0214 | 0.0049 | 0.0079 | 0.0098 | 0.0120 |
| shuffle_target | 0.0038 | 0.0097 | 0.0144 | 0.0211 | 0.0038 | 0.0072 | 0.0091 | 0.0113 |

Paired bootstrap raw − intervention (HR@10 / NDCG@10): mean +0.0794 / +0.0706; shuffle_user +0.0875 / +0.0708; shuffle_target +0.0878 / +0.0715; all significant.

Interpretation: **the bottleneck works as intended**. With the history hidden from target tokens, the decoder depends almost entirely on the sample's own `r`; replacing it with the mean or another sample's `r` drops HR@10 from 0.109 to 0.02–0.03. The cost is accuracy: raw HR@10 0.1085 vs plain R3 v1 0.1226 (−0.014; not bootstrapped, different checkpoint). This is the first configuration where content-dependence of `r` is large *and* ranking is still within ~11% of the baseline.

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| standard eval | | | | | | | | |
| raw | | | | | | | | |
| mask | | | | | | | | |
| mean | | | | | | | | |
| shuffle_user | | | | | | | | |
| shuffle_target | | | | | | | | |

(@20 = @10 with 10 beams.)

## 2026-10-15: L8: queue admission and diversity for collaborative InfoNCE (Attempt 10)

**Motivation.** On Musical, Attempt 10's raw-`r` similarity has random-pair cosine ≈ 0.7–0.8 and same-user pairs piled near 1. The token-accuracy gate (every target token correct) admits only ~11% of samples, so the 10k queue fills only around epoch 1.6 and then holds latents several hundred optimizer steps old, all from "easy" samples. Nothing stops one user from occupying many slots with near-identical latents.

**Changes** (commits `3aa5f2b` + bf16 fix `413b4c0`, frozen copy `output_dir/collab_queue_code/code_413b4c0`; all flags default off):
- Relative NLL gate replaces the token-accuracy gate: admit a sample when its target NLL ≤ the 25% quantile of the last 1,024 training samples (current micro-batch included). Admission ~25% at every stage (~32 entries per optimizer step; the 10k queue spans ~310 steps ≈ 0.6 epoch). Gate applies to candidates (positives and negatives), not anchors.
- L8b only, diversity-aware insertion: same-user entry with cosine ≥ 0.9 is refreshed; user with 2 entries → replace its oldest; target with 4 entries → replace its oldest; else FIFO.
- Queue 10k, no age limit. Everything else = Attempt 10 Musical (plain R3, τ 0.1, weights 0.1/0.1, 16 positives, 256 negatives, 3 epochs).
- New logs: `nll_gate_threshold`, `queue_user_diversity`, `queue_replaced`.

**Evaluation in the job.** Standard ranking; train-latent geometry (`analysis/results/latent_distribution/musical_instruments_collab_queue_{gate,diverse}/`); interventions raw / mask / mean / shuffle_user / shuffle_target (`analysis/results/intervention/musical_instruments_collab_queue_{gate,diverse}/`).

**Compare with** Attempt 10 Musical: HR@10 0.1208 / NDCG@10 0.0937; geometry eff. rank 41.0, random-pair cos 0.693, same-user 0.896, same-target 0.774, user/target η² 0.355/0.206; interventions: only `mask` significant (ΔHR@10 +0.0130).

**Success criteria.** (1) lower random-pair cosine and a same-user distribution less piled at 1, user/target structure kept; (2) ranking ≥ Attempt 10 / plain R3 v1 (0.1226 / 0.0950); (3) `mean`/`shuffle_*` significantly below `raw`. Earlier evidence (Attempt 10 CDs, decoder-insensitivity study) predicts (1) can improve without (3).

**Results (2026-10-16).** Standard ranking (n = 8,313; @20 = @10):

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L8a (NLL gate) | 0.0721 | 0.0895 | 0.1000 | 0.1217 | 0.0721 | 0.0820 | 0.0863 | 0.0933 |
| L8b (gate + diversity) | 0.0749 | 0.0902 | 0.1026 | 0.1210 | 0.0749 | 0.0838 | 0.0889 | 0.0949 |

Interventions (E1 pipeline):

| run | mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L8a | raw | 0.0723 | 0.0893 | 0.0995 | 0.1214 | 0.0723 | 0.0820 | 0.0862 | 0.0933 |
| L8a | mask | 0.0715 | 0.0875 | 0.0988 | 0.1193 | 0.0715 | 0.0807 | 0.0853 | 0.0920 |
| L8a | mean | 0.0713 | 0.0885 | 0.1001 | 0.1202 | 0.0713 | 0.0811 | 0.0858 | 0.0923 |
| L8a | shuffle_user | 0.0711 | 0.0881 | 0.1002 | 0.1204 | 0.0711 | 0.0809 | 0.0859 | 0.0924 |
| L8a | shuffle_target | 0.0711 | 0.0881 | 0.1002 | 0.1204 | 0.0711 | 0.0809 | 0.0859 | 0.0924 |
| L8b | raw | 0.0748 | 0.0909 | 0.1026 | 0.1214 | 0.0748 | 0.0840 | 0.0888 | 0.0949 |
| L8b | mask | 0.0121 | 0.0338 | 0.0481 | 0.0629 | 0.0121 | 0.0240 | 0.0298 | 0.0346 |
| L8b | mean | 0.0751 | 0.0907 | 0.1025 | 0.1211 | 0.0751 | 0.0841 | 0.0889 | 0.0949 |
| L8b | shuffle_user | 0.0749 | 0.0907 | 0.1020 | 0.1204 | 0.0749 | 0.0840 | 0.0886 | 0.0945 |
| L8b | shuffle_target | 0.0749 | 0.0908 | 0.1020 | 0.1204 | 0.0749 | 0.0840 | 0.0886 | 0.0945 |

Interpretation: both queue variants preserve baseline ranking (≈ Attempt 10, plain R3 v1). Neither makes the decoder use `r`'s content: mean and shuffled latents change HR@10 by ≤ 0.0012 (bootstrap CIs not computed). L8b shows strong activation dependence (mask → 0.0629) with no content dependence, the classic collapse signature; L8a barely needs `r` at all (mask −0.0021). `shuffle_user` and `shuffle_target` are again identical (known `experiment.py:645` quirk: both draw the same pseudo-random latent).

**Run history.** First submission (325685/325686, code `3aa5f2b`) crashed at the first training step: `from_pretrained(torch_dtype=bf16)` cast the NLL-history buffer to bf16 and `torch.quantile` rejects bf16. Fix `413b4c0` keeps the buffer float32 (regression test added); resubmitted with `--requeue`.

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| L8a gate | | | | | | | | |
| L8b diverse | 0.0749 | 0.0902 | 0.1026 | 0.1210 | 0.0749 | 0.0838 | 0.0889 | 0.0949 |
| *Attempt 10 (same eval)* | *0.0730* | *0.0905* | *0.1022* | *0.1208* | *0.0730* | *0.0829* | *0.0878* | *0.0937* |

(@20 = @10 with 10 beams.)

**L8b partial results (2026-10-15 22:45; training 3 epochs on L40S in 3h04m, eval loss 0.482 → 0.396 → 0.400).** Ranking above: equal to Attempt 10 and plain R3 v1. Interventions still running.

Queue behaviour (means over training logs): admission 0.23–0.26 (as designed); NLL threshold 0.76 → 0.18 over training. The caps bind hard: most insertions replace an existing entry, so the queue grows slowly (1.2k → 5.3k by epoch 3; never full). Queue user diversity 0.94 → 0.79 (L8a: 0.82 → 0.55). Same-user positive rate 0.12 → 0.43 (L8a ~0.55); same-target 0.44 → 0.60 (L8a ~0.47).

Train-latent geometry vs Attempt 10:

| | Attempt 10 | L8b |
|---|---:|---:|
| effective rank / participation | 41.0 / 11.9 | 48.6 / 10.0 |
| PCs for 50% / 90% variance | 4 / 134 | 4 / 179 |
| cosine to global mean | 0.832 | 0.660 |
| random-pair cosine | 0.693 | **0.440** |
| same-user cosine mean / median / q95 | 0.896 / 0.919 / 0.984 | 0.761 / 0.800 / 1.000 |
| same-target / same-CF-cluster cosine | 0.773 / 0.734 | 0.521 / 0.483 |
| adjusted η² user / target / CF | 0.354 / 0.206 / 0.136 | 0.327 / 0.150 / 0.116 |
| unique vectors | 0.837 | **0.504** |

Reading: the cloud is much less concentrated (random-pair cosine 0.69 → 0.44) and the gap same-user − random widens (0.20 → 0.32), but the target gaps stay small (same-target − random 0.08 → 0.08; CF 0.04 → 0.04) and the η² values fall slightly. Caveat: half the training latents are now exact duplicates at stored precision (unique 0.84 → 0.50), and same-user q95 = 1.0; check whether these are repeated templates before reading the geometry as healthier.

**Geometry incl. L8a (added 2026-10-19).**

| | Attempt 10 | L8a (gate) | L8b (gate + diversity) |
|---|---:|---:|---:|
| effective rank | 41.0 | 30.5 | 48.6 |
| cosine to global mean | 0.832 | 0.592 | 0.660 |
| random-pair cosine | 0.693 | **0.352** | 0.440 |
| same-user mean / median / q95 | 0.896 / 0.919 / 0.984 | 0.800 / 0.840 / 0.952 | 0.761 / 0.800 / 1.000 |
| same-target / same-CF cosine | 0.773 / 0.734 | 0.507 / 0.439 | 0.521 / 0.483 |
| gap to random: user / target / CF | 0.20 / 0.08 / 0.04 | **0.45 / 0.16 / 0.09** | 0.32 / 0.08 / 0.04 |
| adjusted η² user / target / CF | 0.354 / 0.206 / 0.136 | 0.382 / 0.189 / 0.135 | 0.327 / 0.150 / 0.116 |
| unique vectors | 0.837 | 0.856 | 0.504 |

Correction to the L8b reading above: most of the spread comes from the **NLL gate**, not the diversity rules. L8a has the lowest random-pair cosine and the largest same-user/target/CF gaps, with no duplicate inflation; adding the diversity rules (L8b) moved random pairs back up (0.35 → 0.44), halved the target and CF gaps and introduced the 50% duplicate vectors.

## 2026-10-16: L9: R3 stage 2 (GRPO) with the collaborative loss

**Question.** Does R3's stage-2 GRPO improve the L8b model, and does keeping the collaborative InfoNCE during GRPO add anything over stock stage 2?

**Stock stage 2** (`src/grpo_attention_tuning/`): only the latent producer (`attention`, 7.08M params) trains; the LLM is frozen. Each prompt is repeated 8 times; 7 copies get Gaussian noise (σ = 1.5) on `r`, the first stays noise-free. Reward = −perplexity of the true target title under each `r` (teacher-forced, no generation). Advantage = reward − the noise-free copy's reward; clipped PPO-style update (ε 0.2/0.28) with KL β 0.01 to the starting model.

**Collaborative term** (`collab_grpo.py`, flag `--collab`, default off; commit `197a192`, frozen copy `output_dir/grpo_stage2_code/code_197a192`): loss = GRPO + λ·(L_user + L_target), with the L8b queue (NLL gate best 25% of last 1,024; 10k slots; per-user cap 2, per-target cap 4, dedup cos 0.9) and L8b's projection heads (trainable).
- Anchor = the noise-free copy of each group only (noisy copies are exploration and would be fake same-sample positives). Gate NLL from that copy's target log-probs (no extra pass).
- Queue prefilled once from the stage-1 checkpoint over the training set before GRPO.
- λ auto-calibrated: λ = 0 for the first 20 optimizer steps while recording gradient norms of both losses on `attention` per micro-batch; then λ = 0.3 × median‖g_GRPO‖ / median‖g_InfoNCE‖ (printed in the log). Needed because the GRPO loss is tiny (advantages normalised by their global norm; earlier runs logged loss ~1e-3) while InfoNCE is ~3–4.

**Runs.** lr 1e-5, β 0.01, 8 generations, batch 256, micro-batch 8 (one group per micro-batch), 1 epoch, one GPU. Each preceded by a smoke job (256 prompts, batch 64, ranking on 64 test rows); full jobs start only if their smoke succeeds (`afterok`), with `--requeue`. Then ranking (`noise_eval.py`), train-latent geometry, interventions raw / mask / mean / shuffle_user / shuffle_target.
- L9B: L8b checkpoint + collaborative term.
- L9C: plain R3 v1 stage-1 checkpoint (`scratchDirectory/R3/output_dir/Musical_Instruments_v1/latent`) + stock stage 2.

**Context and risks.**
- No valid Musical stage-2 result existed: `Musical_Instruments_v1/nomul_attn_lr_5e-5/final_result.json` is byte-identical to the stage-1 file and that run saved no final model. So "plain R3 v1" (0.1226 / 0.0950) is a stage-1 number; L9C supplies the stage-2 baseline.
- CDs v4 stage 2 (lr 5e-5) collapsed: HR@10 ≈ 7× lower and zeroing or noising `r` then improved ranking (`latent_v4_epoch_collapse_report.md` §7). CDs v7 stage 2 was a near no-op. lr here is 1e-5.
- The decoder is frozen and barely reads `r`'s content, so InfoNCE reshaping `r` may not change predictions; the GRPO reward is the term that directly pushes `r` toward what the decoder needs.

**Compare.** L9B vs L9C: does L8b's stage 1 + collaborative stage 2 beat plain R3 + stock stage 2. L9B vs L8b and L9C vs plain R3 v1: what stage 2 adds. Not run: L8b + stock stage 2 (would isolate the collaborative term during GRPO).

**Status (2026-10-16 21:40).** Both smokes failed within a minute with the same error, so the full jobs never started (Slurm: `DependencyNeverSatisfied`; 326394/326395 still listed and need cancelling):

```
grpo_trainer.py:174 _generate_and_score_completions -> grpo_trainer.py:24 swap_adjacent_blocks
RuntimeError: shape '[-1, 2]' is invalid for input of size 1
```

`swap_adjacent_blocks` pairs adjacent completions but received a block of size 1, likely an odd or size-1 generation group/batch under the smoke settings.

**Fix (2026-10-16 22:00, commit `9e7f11d`).** Root cause: both variants run `micro_batch_size 8 = num_generations 8` (required for collab GRPO by `train_noise_grpo.py:129`), so each micro-batch holds one group, and `swap_adjacent_blocks` reshaped the group baselines with `view(-1, 2)`. The full runs would have crashed at step 1 too. Unpaired blocks are now left in place. The only use is `advantages = rewards − xx.mean()`, and a permutation does not change the mean, so results with ≥ 2 groups are unchanged. Tests: `tests/test_grpo_swap_blocks.py` (one group unchanged, two groups swapped as before, odd count, mean preserved; 7 passed).

**Resubmitted** with frozen code `output_dir/grpo_stage2_code/code_9e7f11d`; the blocked 326394/326395 were cancelled. Smokes use separate tags (`*_smoke`) so they don't write into the full runs' directories:
- L9B: smoke 328081 → full 328083 (afterok); base = L8b checkpoint; tag `musical_l9b_collab`.
- L9C: smoke 328082 → full 328084 (afterok); base = plain R3 v1 stage-1; tag `musical_l9c_stock`.

**Second L9B failure and change (2026-10-16 22:30, commit `a9c1efa`).** L9B smoke 328081 got past GRPO but aborted in λ calibration: "lambda calibration saw no micro-batch with an active InfoNCE term". The smoke fills the queue from its own 256 training rows (68 entries after gate and caps), so no anchor had a same-user/target positive and InfoNCE was zero. Change (user's suggestion): as in stage 1, anchors without positives contribute zero. λ stays 0 (pure GRPO) until `calibration_steps` have passed **and** ≥ 8 micro-batches had an active InfoNCE term, then calibrates as before; a one-time message is logged if waiting. For the full run (10k-entry queue from 66.5k rows) this should be identical to the original design. Tests: `tests/test_collab_grpo_calibration.py` (3 passed).
- New L9B smoke 328105 logged "only 0 micro-batches with an active InfoNCE term after 2 steps … lambda stays 0" and completed (64 ranked rows). Its InfoNCE gradient path was therefore not exercised; the full run 328106 is the first real test.
- Full runs running as of 2026-10-16 22:45: **L9B 328106** (code `a9c1efa`), **L9C 328084** (code `9e7f11d`; the stock variant does not use `collab_grpo.py`). Watch L9B's log for "collab lambda calibrated over N micro-batches".

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| L9B (L8b + GRPO + InfoNCE) | 0.0751 | 0.0907 | 0.1031 | 0.1215 | 0.0751 | 0.0840 | 0.0891 | 0.0950 |
| *L8b (stage-1 start)* | *0.0749* | *0.0902* | *0.1026* | *0.1210* | *0.0749* | *0.0838* | *0.0889* | *0.0949* |
| L9C (R3 v1 + GRPO) | | | | | | | | |

(@20 = @10 with 10 beams.)

**L9B results (2026-10-17).** Training 2,078 steps on an L40 in ~8.5 h; queue prefill 66,500 rows → 5,084 entries (gate threshold 0.180). λ calibrated over 487 micro-batches: median ‖g_GRPO‖ 1.68e-2, median ‖g_InfoNCE‖ 6.54, **λ = 7.7e-4** (30% share).

Ranking (table above) is unchanged from its L8b starting point (ΔHR@10 +0.0005). Interventions (E1 pipeline; paired bootstrap raw − mode, 2,000 resamples, seed 42):

| mode | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | ΔHR@10 [95% CI] | ΔNDCG@10 [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| raw | 0.0754 | 0.0907 | 0.1027 | 0.1216 | 0.0754 | 0.0842 | 0.0891 | 0.0951 | | |
| mask | 0.0117 | 0.0340 | 0.0478 | 0.0626 | 0.0117 | 0.0239 | 0.0295 | 0.0343 | +0.0591 [+0.0535, +0.0653] | +0.0608 [+0.0563, +0.0659] |
| mean | 0.0753 | 0.0900 | 0.1016 | 0.1214 | 0.0753 | 0.0838 | 0.0886 | 0.0949 | +0.0002 [−0.0014, +0.0019] | +0.0002 [−0.0006, +0.0010] |
| shuffle_user | 0.0747 | 0.0902 | 0.1018 | 0.1202 | 0.0747 | 0.0836 | 0.0884 | 0.0943 | +0.0014 [−0.0004, +0.0032] | +0.0009 [−0.0001, +0.0018] |
| shuffle_target | 0.0747 | 0.0902 | 0.1018 | 0.1202 | 0.0747 | 0.0836 | 0.0884 | 0.0943 | +0.0014 [−0.0004, +0.0032] | +0.0009 [−0.0001, +0.0018] |

Same bootstrap for L8b: mask +0.0585 [+0.0526, +0.0644]; mean +0.0002 [−0.0013, +0.0018]; shuffle +0.0010 [−0.0006, +0.0025] (HR@10). So L9B keeps L8b's signature: strong activation dependence, no content dependence.

Training signals: logged KL exactly 0 at every step; grad norm 0.007 → 0.012; user/target InfoNCE flat (1.42 → 1.40 / 1.94 → 1.87); same-user positive rate 0.44 → 0.49; queue 5.1k → 5.9k. Attention weights moved little vs L8b (relative change: query/key 0.3%, value 1.4%, value bias 13%). Geometry vs L8b: effective rank 48.6 → 48.8, random-pair cosine 0.440 → 0.434, same-user 0.761 → 0.759, same-target 0.521 → 0.516, η² user/target unchanged (0.327/0.150); unique vectors 0.50 → 0.61.

Reading: stage 2 at lr 1e-5 is a near no-op on L8b, with or without the collaborative term (λ is small by design, InfoNCE stays flat). The KL = 0 log means the policy's target log-probs never measurably left the reference; consistent with a frozen decoder that ignores `r`'s content, but not verified (could also be a logging issue). Without L9C we cannot yet say whether stock stage 2 does anything for plain R3 on Musical.

**L9C status.** Preempted 2026-10-17 04:12 at step 1757/2078 after 6h24m; Slurm did not requeue it despite `--requeue`. `output_dir/Musical_Instruments_R3_grpo_musical_l9c_stock/model/checkpoint-1600` exists; resuming needs `--resume_from_checkpoint` passed through the sbatch (not supported by the script yet).


## 2026-10-16 21:40: status update

- **L6b final.** `r0` HR@10 0.0936 / NDCG@10 0.0757 (lowest of all L6 variants). The margin makes the model depend on its intent paths: random 0.0461, r0 0.0936, mixture 0.1050, oracle 0.1796.
- **L7 done:** strong content dependence (mean / shuffle drop HR@10 by ≈0.08–0.09, significant) at raw HR@10 0.1085 (−0.014 vs plain R3 v1).
- **L8a / L8b done:** baseline-level ranking (0.1217 / 0.1210), no content dependence.
- **L9:** both smokes failed (`swap_adjacent_blocks` size-1 group); full jobs blocked.

**L9 resubmission (2026-10-18).** L9B was a no-op at lr 1e-5 (vs L8b: HR@10 +0.0005 [−0.0010, +0.0020], NDCG@10 +0.0003 [−0.0004, +0.0009]; 38 of 8,019 paired rows changed top-10 hit). New runs, all **ranking only** (`EVAL=ranking`; geometry and interventions to be run later on request with `SKIP_TRAIN=1`), code `5c56bf0` (adds `LR` and `EVAL` to the sbatch; training code unchanged from `a9c1efa`), `--requeue`:
- L9B at lr **1e-4** (job 330291) and **5e-4** (job 330292); λ re-calibrates per run. Risk: CDs v4 stage 2 collapsed at 5e-5.
- L9C restarted from scratch at lr 1e-5 (job 330293), new tag `musical_l9c_stock_lr1e-5`; the preempted run's `checkpoint-1600` is kept in its own folder.

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| L9B lr 1e-4 | 0.0308 | 0.0399 | 0.0452 | 0.0577 | 0.0308 | 0.0362 | 0.0383 | 0.0424 |
| L9B lr 5e-4 | 0.0343 | 0.0375 | 0.0405 | 0.0482 | 0.0343 | 0.0361 | 0.0373 | 0.0398 |
| L9C lr 1e-5 | 0.0741 | 0.0905 | 0.1026 | 0.1223 | 0.0741 | 0.0835 | 0.0885 | 0.0947 |
| *L8b (start of L9B)* | *0.0749* | *0.0902* | *0.1026* | *0.1210* | *0.0749* | *0.0838* | *0.0889* | *0.0949* |
| *L9B lr 1e-5* | *0.0751* | *0.0907* | *0.1031* | *0.1215* | *0.0751* | *0.0840* | *0.0891* | *0.0950* |
| *R3 v1 (start of L9C)* | *0.0747* | *0.0901* | *0.1020* | *0.1226* | *0.0747* | *0.0836* | *0.0884* | *0.0950* |

**Results (2026-10-18 14:05; 1e-4 still ranking).** Paired bootstrap vs each run's starting model (2,000 resamples, seed 42; @20 = @10):
- L9B lr 5e-4 − L8b: HR@10 **−0.0728 [−0.0792, −0.0671]**, NDCG@10 −0.0551 [−0.0597, −0.0507]. Collapse. Grad norm 0.22 (first 5% of steps) → 5.7 (5–20%) → 26 (20–50%) → 7.8 (last 10%); user/target InfoNCE rose 1.41/1.93 → 3.55/3.27 (λ calibrated to 7.4e-4, so the InfoNCE term is not the driver).
- L9C lr 1e-5 − R3 v1: HR@10 −0.0002 [−0.0014, +0.0008], NDCG@10 −0.0003 [−0.0009, +0.0003]. Stock stage 2 is a no-op on plain R3 at 1e-5 (grad norm ≈ 0.004 throughout).

**The KL penalty is structurally inactive (found 2026-10-18).** Logged KL is exactly 0 in every stage-2 run, including the collapsing one. Cause: `grpo_trainer.py::_generate_and_score_completions` computes `ref_per_token_logps` with `self.ref_model` on `original_embeds`, which already hold the `r` produced by the *current policy's* attention module; only the attention module trains and the reference decoder is the same frozen decoder, so reference and policy log-probs are identical by construction. β = 0.01 therefore never anchors the attention module to its starting point. With `num_iterations = 1`, `clip_ratio` is also 0 by construction (old log-probs = current). So stage 2 has no trust region at all; consistent with the CDs v4 collapse diagnosis ("no absolute anchor to the SFT baseline"). A real anchor would need the reference log-probs computed from the *reference* attention module's `r` (code change, not made).

Reading: on Musical, R3 stage 2 does nothing at lr 1e-5 (plain R3 and L8b alike) and destroys ranking at 5e-4. No setting tested helps.

**L9B lr 1e-4 (done 2026-10-18 14:37).** Also collapsed: HR@10 − L8b **−0.0633 [−0.0696, −0.0575]**, NDCG@10 −0.0525 [−0.0573, −0.0481]. Grad norm 0.02 (first 5%) → 1.9 (5–20%) → 5.3 (20–50%) → 7.2 (last 10%); user InfoNCE flat (1.38 → 1.41), target InfoNCE rose 1.94 → 2.78; λ 7.0e-4; KL 0 throughout. Sweep summary on L8b: 1e-5 no change, 1e-4 and 5e-4 collapse. Without a working KL anchor there is no learning rate in between that we know to help.

## 2026-10-19: L10: behaviour-grounded reasoning on Toys and Games

**Question.** Does the only collaborative method with a significant gain (L5 on CDs: knn − shuffle_outcome HR@10 +0.0023 [+0.0010, +0.0037]) carry over to a third dataset? L4 (Musical) showed no effect; the hypothesis from L4/L5 is that it helps where a user's own history explains the next item less well (CDs validation loss 1.25 vs Musical 0.40) and where donors still add information at the end of training (item hit@10 `r1` − `r0` at epoch 3: CDs +0.008, Musical −0.001).

**Toys baseline (existing results, n = 6,738).** Plain R3 v0 checkpoint 1263: HR@10 0.1061 / NDCG@10 0.0678 (mean / shuffle `r` neutral). Attempt 10: HR@10 0.1045 / NDCG@10 0.0639 (mean / shuffle neutral, mask −0.038): the same pattern as Musical/CDs.

**Setup.** Identical to L4/L5 (frozen code `output_dir/grounded_code/code_4671046`, `CATEGORY=Toys_and_Games`), including the known bf16 issue (grounding strength fixed at 0.5) for comparability. Toys had no item-title table, so job 332483 builds `data/Toys_and_Games/info/Toys_and_Games_qwen3emb_1536.pt` (6,299 items; `src/latent/item_title_embeddings.py`, commit `93f819c`): Qwen3-Embedding-8B, title only, last token, 1536 dims, after re-embedding 256 CDs titles and requiring cosine ≥ 0.99 with the stored CDs table. One Toys title contains a tab (item 1672, "63074\tMagformers …"); the id is taken from the last field and the title kept as in the CSVs. The grounded job 332484 starts only if the embedding job succeeds (`afterok`), with `--requeue`.

**Controls.** knn (method), off, shuffle_outcome, latent_only, random; paired bootstrap knn − each.

**Results (2026-10-20).** Log: `run_logs/R3_grounded_Toys.332484.out`. Item-table recipe check passed (CDs re-embed cosine 0.99990, min 0.99972). Paired bootstrap knn − control (2,000 resamples, seed 42):

| comparison | ΔHR@10 [95% CI] | ΔNDCG@10 [95% CI] |
|---|---|---|
| knn − off | **+0.0019 [+0.0001, +0.0037]** | **+0.0013 [+0.0005, +0.0021]** |
| knn − shuffle_outcome | +0.0006 [−0.0007, +0.0019] | +0.0004 [−0.0001, +0.0010] |
| knn − latent_only | +0.0010 [−0.0001, +0.0024] | +0.0006 [−0.0000, +0.0011] |
| knn − random | +0.0003 [−0.0010, +0.0018] | +0.0001 [−0.0005, +0.0007] |

Training diagnostics: item hit@10 `r0` / `r1` epoch 1 0.086 / 0.092, epoch 3 0.222 / 0.227 (`r1` − `r0` +0.005); best single-donor gain 0.065 → 0.145; eval loss 0.768 / 0.594 / 0.626.

Reading: knn beats no-donors (`off`) significantly, but **not** shuffled outcomes. The behaviour-specific claim (CDs' clean test) does not hold on Toys, and knn > off is the weak claim (bf16: the decoder never saw bare `r0` in training). Toys sits between Musical and CDs on both candidate factors (validation loss 0.59–0.63 vs Musical 0.40 / CDs 1.25; end-of-training donor gain +0.005 vs −0.001 / +0.008) and its result is in between too: an effect vs `off` like CDs, no outcome-specific effect like Musical. Consistent with the self-predictability hypothesis, but three datasets and single seeds.

| control | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| knn | 0.0390 | 0.0647 | 0.0801 | 0.1066 | 0.0390 | 0.0539 | 0.0603 | 0.0688 |
| off | 0.0383 | 0.0631 | 0.0801 | 0.1046 | 0.0383 | 0.0526 | 0.0596 | 0.0675 |
| shuffle_outcome | 0.0387 | 0.0644 | 0.0798 | 0.1060 | 0.0387 | 0.0536 | 0.0599 | 0.0684 |
| latent_only | 0.0386 | 0.0647 | 0.0804 | 0.1055 | 0.0386 | 0.0537 | 0.0602 | 0.0682 |
| random | 0.0390 | 0.0646 | 0.0806 | 0.1063 | 0.0390 | 0.0538 | 0.0604 | 0.0687 |

(@20 = @10 with 10 beams.)

## 2026-10-19: Attempt 10 rerun on Toys

User request: rerun collaborative reasoning on Toys exactly as originally run, ranking only. Job 332796 runs `scripts/latent_collaborative_infonce_toys_full.sbatch` unchanged (training 3 epochs + standard ranking) with `POSITIVE_MIN_TOKEN_ACCURACY=1.0`, `OUTPUT_SUFFIX=_rerun_20261019`, `--requeue`.
- Gate: the script's default became 0.5 in commit `cec5f93` (2026-09-28); the original unsuffixed Toys run is taken to be gate 1.0 by analogy with the Musical naming (unsuffixed = 1.0, `_acc075` / `_acc05` = variants). Not recorded in its config, so this is an inference.
- Code: frozen copy of this branch at `93f819c` (`output_dir/collab_toys_code/code_93f819c`, with `data/`, `output_dir/`, `run_logs/` symlinked to the repo; job run with `--chdir` there), because the main checkout's `src/` carries another session's uncommitted edits to `layers.py` and `latent_attention_eval.py`. The collaborative code at `93f819c` has the L8 options, all off by default, so it behaves as Attempt 10.
- Compare with: original Toys Attempt 10 HR@10 0.1045 / NDCG@10 0.0639 (seed-to-seed spread), plain R3 v0 0.1061 / 0.0678.

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| Attempt 10 Toys (original) | 0.0324 | 0.0585 | 0.0779 | 0.1045 | 0.0324 | 0.0474 | 0.0553 | 0.0639 |
| Attempt 10 Toys rerun | | | | | | | | |
| Attempt 10 Toys, gate 0.5 (job 332797) | 0.0380 | 0.0641 | 0.0801 | 0.1073 | 0.0380 | 0.0530 | 0.0596 | 0.0683 |
| *Attempt 10 Toys, gate 0.75 (existing)* | *0.0373* | *0.0631* | *0.0779* | *0.1008* | *0.0373* | *0.0525* | *0.0585* | *0.0660* |

**Toys gate results (2026-10-20).** Gate 1.0 rerun 332796 was cancelled at 23:20 (scancel, not by this session); gate 0.5 (332797) finished: HR@10 0.1073 / NDCG@10 0.0683 (plain R3 v0 0.1061 / 0.0678). Paired bootstrap vs earlier runs (test-set resampling only; does not include training variance, which is ~0.002 HR@10 between seeds on Musical): − original A10: HR@10 +0.0028 [−0.0019, +0.0076], NDCG@10 +0.0044 [+0.0015, +0.0073]; − gate 0.75: HR@10 +0.0065 [+0.0015, +0.0111], NDCG@10 +0.0023 [−0.0006, +0.0049]. Gate 0.5 is at least as good as the earlier Toys Attempt 10 runs and level with plain R3; separate trainings, so not a clean gate effect.

## 2026-10-23: Attempt 10 reproduction on CDs

User request: reproduce Attempt 10 (collaborative InfoNCE) on CDs with the same settings. Job 335891 runs `scripts/latent_collaborative_infonce_full.sbatch` unchanged (gate 1.0 is its default; 10k FIFO, τ 0.1, weights 0.1/0.1, 16 positives, 256 negatives, 3 epochs, then standard ranking), `OUTPUT_SUFFIX=_repro_20261023`, `--requeue`, from the frozen copy `output_dir/collab_toys_code/code_93f819c` (L8 options off by default, so Attempt-10 behaviour; avoids the main checkout's uncommitted edits). Seed as in the original (script passes none; library defaults), so GPU nondeterminism is the only expected source of difference.

Reference numbers for the original run (job 283515): the Attempt-10 doc reports HR@10 0.1166 / NDCG@10 0.0813 (@1 0.0533); recomputing its current `final_result.json` gives 0.1150 / 0.0817 (@1 0.0551), so the file may have been overwritten by a later re-evaluation. Gate sweep on CDs (same code family): 0.75 → 0.1181 / 0.0818, 0.5 → 0.1152 / 0.0822, 0 → 0.1101 / 0.0768. Plain R3 v4: 0.1152 / 0.0806.

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| Attempt 10 CDs (doc) | 0.0533 | 0.0780 | 0.0932 | 0.1166 | 0.0533 | 0.0675 | 0.0738 | 0.0813 |
| Attempt 10 CDs (current file) | 0.0551 | 0.0788 | 0.0924 | 0.1150 | 0.0551 | 0.0688 | 0.0744 | 0.0817 |
| Attempt 10 CDs reproduction | | | | | | | | |

## 2026-10-25: scratch purge; D1: queue drift (staleness) measurement

**Scratch purge.** `/common/scratch/users/m/mk.ngohuu.2024/R3/output_dir` was emptied before 2026-10-23 09:42 (the folder was recreated by the CDs reproduction job). Lost: everything stored only there, incl. the original Attempt 10 CDs/Musical runs and their gate sweeps, and plain R3 v0/v1 checkpoints (`Musical_Instruments_v1/latent`, L9C's starting point). Copies in `R3/output_dir/` and `analysis/` are intact. Store new outputs under `R3/output_dir/`.

**A10 CDs reproduction (job 335891) finished** 2026-10-23 14:34 (gate 1.0 confirmed in its log); its numbers are not yet logged, and the original run's prediction file no longer exists for a paired comparison.

**D1 question.** The Attempt 10 queue stores latents computed by an older model (entries live hundreds of optimizer steps), like a replay buffer without a target network. Is the drift between a stored `r_t(x)` and its current value `r_{t+D}(x)` large enough to give InfoNCE inconsistent positives/negatives? If yes, a momentum (MoCo-style) encoder or a queue-free large-batch scheme (GradCache) is justified; if no, staleness is not the bottleneck.

**Design** (commit `fbf53ca`, frozen copy `output_dir/collab_drift_code/code_fbf53ca`; job 338330, `--requeue`): Toys, Attempt 10 with token-accuracy gate 0.5 (best Toys collaborative setting), 2 epochs (~842 optimizer steps), model-only checkpoint every 100 steps, all kept (new `--save_steps` / `--save_total_limit` / `--save_only_model` options; defaults unchanged), no ranking eval. Then `scripts/latent_drift_analysis.py` re-encodes 2,000 training samples (458 users, ~7,000 same-user pairs) with every checkpoint and, for every pair (t, t+D), reports: raw and centred cos(r_t, r_{t+D}); the same through the step-(t+D) user/target projection heads (what InfoNCE compares); per-anchor Spearman correlation of similarities to the other samples with stale vs fresh candidates; top-16 neighbour overlap; and the fresh same-user − random gap at t+D as the scale drift should be compared with.

**Reading.** Raw/projected cos > ~0.95 and rank correlation > ~0.9 at D ≈ 300 → staleness negligible. Drift comparable to the same-user gap → queue feeds inconsistent targets.

**Results (2026-10-25).** Log: `run_logs/R3_collab_drift_Toys.338330.out`; data `analysis/results/drift/toys_acc05.json`. 842 optimizer steps (421/epoch), checkpoints at 100…800 and 842; lr decays linearly 3e-4 → 0.

Queue turnover in this run (gate 0.5): admission 0.79 (first half epoch) → ~1.0 afterwards, so ~100–128 entries per step and the 10k queue (full by epoch 0.27) spans only **~80–100 optimizer steps**. Positive availability: same-user 0.45 → 0.65, same-target 0.61 → 0.81.

Drift at the relevant age (D = 100 steps), by training stage:

| stale step t → fresh t+100 | raw cos | user-head cos | target-head cos | rank corr | top-16 overlap |
|---|---:|---:|---:|---:|---:|
| 100 → 200 | 0.600 | 0.681 | 0.744 | 0.765 | 0.590 |
| 200 → 300 | 0.765 | 0.837 | 0.783 | 0.870 | 0.726 |
| 300 → 400 | 0.829 | 0.881 | 0.814 | 0.910 | 0.772 |
| 400 → 500 | 0.901 | 0.928 | 0.891 | 0.945 | 0.817 |
| 500 → 600 | 0.937 | 0.962 | 0.869 | 0.971 | 0.867 |
| 600 → 700 | 0.979 | 0.987 | 0.974 | 0.990 | 0.917 |
| 700 → 800 | 0.996 | 0.997 | 0.991 | 0.998 | 0.961 |

Longer ages (what slower-filling queues would see), from step 100–300: D = 300 → user-head 0.58–0.86, target-head 0.39–0.49, rank corr 0.68–0.88; D = 600–700 from step 100 → user-head 0.56, **target-head ≈ 0.0**. Fresh same-user − random gap through the user head: 0.85–0.88 at every checkpoint.

Reading:
1. **Staleness is real early and negligible late.** At the queue's actual age (~100 steps) a stored latent is substantially off in the first ~400 steps (≈ first epoch: rank corr 0.77–0.91, top-16 overlap 0.59–0.77) and essentially exact in the second epoch (≥ 0.97 / 0.87–0.96), tracking the linear lr decay.
2. **The target relation is the most fragile.** Through the target head, stale latents drift much faster than through the user head (D = 300 from step 300: 0.49 vs 0.86; old entries reach ≈ 0). Same-target positives from an old queue entry are close to arbitrary vectors.
3. **Queue age matters more than the gate itself.** Gate 0.5 keeps the queue young (~100 steps). The original gate 1.0 (~11% admitted on Musical) makes entries live several hundred steps; at that age in epoch 1 the target-head drift is ≈ 0.4–0.0, i.e. the queue feeds largely inconsistent target positives. Possibly part of why gate 0 hurt on CDs and gate 0.5 was best on Toys, but not tested.
4. Implication: a momentum encoder (MoCo) or queue-free large batches would mainly fix the first epoch and slow-filling queues; with a fast-turning queue late in training staleness is not the bottleneck. Combined with earlier evidence, the main limit remains the decoder bypass (content of `r` unused), which drift does not address.

Note: `tests/test_collab_grpo.py::test_lambda_is_zero_during_calibration_then_set_from_gradient_ratio` now fails (fake trainer lacks `min_active_micro_batches`, added to `collab_grpo.py` in `a9c1efa`); test-only, unrelated to D1.
