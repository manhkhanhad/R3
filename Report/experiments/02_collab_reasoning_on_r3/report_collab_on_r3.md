# E02 — Collaborative reasoning applied directly to plain R3 does not change predictions
Back to [main plan](../../main.md). Dates: 2026-09 (Attempt 10 CDs) to 2026-11-07 (WIDER K = 1 corrected). Status: done (motivation; negative).

## Question
The original idea behind C2 is that people buy an item for shared reasons. Each sample's latent should therefore be grounded in other users' latents: rows of the same user and rows with the same target should have compatible `r`. Applied to plain R3, does this make `r` better organised, and does it change what the decoder predicts?

## Setup
- **Attempt 10 (A10)** = original collaborative reasoning (`analysis/document/collaborative_reasoning.md`, avoid doc §10):
  - two projection heads `g_u`, `g_t` (LayerNorm → Linear 1536→256) on the deployed `r`;
  - multi-positive InfoNCE: positives = same user (user head) or same target (target head); negatives share neither; τ 0.1, ≤ 16 positives, 256 negatives;
  - loss = CE + 0.1·L_user + 0.1·L_target;
  - 10k FIFO queue of detached latents, gated on token accuracy 1.0 (gate 0.5 / 0.75 variants);
  - heads and queue are training-only; inference is plain R3.
- **Attempt 11:** same, with soft SASRec pair weights (sequence / item similarity); Musical, 304495.
- **L8a / L8b** (Musical; 325727 / 325728, code 413b4c0): A10 with a relative-NLL queue gate (best 25% of the last 1,024); L8b adds diversity caps (2 entries per user, 4 per target, same-user near-duplicate refresh).
- **WIDER K = 1** (Toys, 355749, code 001a5a2a): cross-user diversity penalty 0.1·mean(z·q)² on unit `r`, q = EMA batch-mean direction. 355111 is a superseded run with a 32× weight bug (λ_eff 3.2).
- **D1 queue drift** (Toys, 338330, code fbf53ca): A10 gate 0.5, 2 epochs, checkpoint every 100 steps; 2,000 training rows re-encoded with every checkpoint.
- Datasets and test n: CDs 6,157; Musical 8,313; Toys 6,738. Latent tests by `experiment.py`.
- Table 2 Δ and p-values were **computed for this report** (read-only) from the saved per-row files, with a paired t-test and paired bootstrap (2,000, seed 42). The CDs A10 bootstrap reproduces the logged CIs exactly (mask +0.0284 [+0.0227, +0.0339] in raw − mode form).

## Results

### 1. Ranking (all cutoffs; HR@1 = NDCG@1; @20 = @10)

| dataset | model (run) | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CDs | A10 (283515; doc numbers) | 0.0533 | 0.0780 | 0.0932 | 0.1166 | 0.0675 | 0.0738 | 0.0813 |
| CDs | A10 (same run, current file) | 0.0551 | 0.0788 | 0.0924 | 0.1150 | 0.0688 | 0.0744 | 0.0817 |
| CDs | *plain R3 v4 ckpt 1152 (reference)* | not recorded | not recorded | not recorded | *0.1152* | not recorded | not recorded | *0.0806* |
| CDs | A10 reproduction (335891, gate 1.0) | not recorded | | | | | | |
| Musical | A10 (job not recorded; L8 same-eval) | 0.0730 | 0.0905 | 0.1022 | 0.1208 | 0.0829 | 0.0878 | 0.0937 |
| Musical | Attempt 11 (304495) | not recorded | not recorded | not recorded | 0.1208 | not recorded | not recorded | 0.0953 |
| Musical | L8a NLL gate (325727) | 0.0721 | 0.0895 | 0.1000 | 0.1217 | 0.0820 | 0.0863 | 0.0933 |
| Musical | L8b gate + diversity (325728) | 0.0749 | 0.0902 | 0.1026 | 0.1210 | 0.0838 | 0.0889 | 0.0949 |
| Musical | *plain R3 v1 (reference)* | *0.0747* | *0.0901* | *0.1020* | *0.1226* | *0.0836* | *0.0884* | *0.0950* |
| Toys | A10 original (gate 1.0 inferred) | 0.0324 | 0.0585 | 0.0779 | 0.1045 | 0.0474 | 0.0553 | 0.0639 |
| Toys | A10 gate 0.75 | 0.0373 | 0.0631 | 0.0779 | 0.1008 | 0.0525 | 0.0585 | 0.0660 |
| Toys | A10 gate 0.5 (332797) | 0.0380 | 0.0641 | 0.0801 | 0.1073 | 0.0530 | 0.0596 | 0.0683 |
| Toys | WIDER K = 1 corrected (355749) | 0.0350 | 0.0628 | 0.0790 | 0.1070 | 0.0512 | 0.0578 | 0.0669 |
| Toys | *plain R3 (A) 345857 (reference)* | *0.0367* | *0.0625* | *0.0795* | *0.1052* | *0.0514* | *0.0584* | *0.0667* |
| Toys | *R3_ori stage 1 (reference)* | *0.0377* | *0.0607* | *0.0778* | *0.1023* | *0.0510* | *0.0580* | *0.0659* |

Recorded comparisons:
- Attempt 11 vs v1: HR@10 −0.0018 [−0.0063, +0.0023], NDCG@10 +0.0002 [−0.0021, +0.0025]. Two A10 Musical seeds differ by 0.0024 HR@10.
- Toys gate 0.5 vs original A10: HR@10 +0.0028 [−0.0019, +0.0076], NDCG@10 +0.0044 [+0.0015, +0.0073]. Separate trainings, so this is not a clean gate effect.
- WIDER 355749 vs plain A: HR@10 +0.0018 [−0.0025, +0.0061], NDCG@10 +0.0002 [−0.0021, +0.0025], n.s.
- Buggy 355111 (λ_eff 3.2): HR@10 0.0870, −0.0183 sig vs A.

All collaborative variants preserve but do not beat plain R3.

### 2. Latent test (Δ = mode − raw, p_t / p_b; bold = significant)

| run | raw HR@10 / NDCG@10 | mask HR@10 | mean HR@5 | mean HR@10 | mean NDCG@10 | shuffle HR@5 | shuffle HR@10 | shuffle NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| CDs A10 (latent test 296325) | 0.1178 / 0.0820 | **−0.0284** (1e-22/<5e-4) | +0.0002 (.85/.91) | −0.0005 (.62/.70) | +0.0002 (.60/.62) | −0.0008 (.40/.46) | −0.0011 (.30/.35) | −0.0002 (.70/.70) |
| Musical A10 | 0.1217 / 0.0939 | **−0.0130** (5e-10/<5e-4) | **−0.0016** (.028/.027) | −0.0013 (.14/.17) | −0.0005 (.22/.23) | −0.0013 (.10/.12) | −0.0010 (.28/.28) | −0.0007 (.11/.10) |
| Toys A10 original | 0.1036 / 0.0639 | **−0.0381** (9e-38/<5e-4) | −0.0004 (.47/.52) | +0.0004 (.58/.62) | −0.0001 (.70/.69) | −0.0003 (.65/.69) | +0.0009 (.32/.37) | +0.0001 (.87/.87) |
| Musical L8a (325727) | 0.1214 / 0.0933 | −0.0020 (.065/.072) | +0.0006 (.46/.49) | −0.0012 (.17/.19) | **−0.0010** (.020/.019) | +0.0007 (.35/.43) | −0.0010 (.29/.31) | −0.0009 (.059/.052) |
| Musical L8b (325728) | 0.1214 / 0.0949 | **−0.0585** (3e-83/<5e-4) | −0.0001 (.87/.94) | −0.0002 (.76/.80) | +0.0001 (.85/.84) | −0.0006 (.41/.49) | −0.0010 (.22/.24) | −0.0004 (.38/.39) |
| Toys WIDER (355749) | 0.1070 / 0.0669 | −0.0009 (.33/.39) | +0.0001 (.86/.96) | +0.0007 (.37/.40) | +0.0000 (.92/.90) | +0.0001 (.85/.91) | +0.0009 (.34/.38) | +0.0003 (.45/.47) |
| Toys A10 gate 0.5 (332797) | — | not recorded (ranking only) | | | | | | |
| Musical Attempt 11 (304495) | — | not recorded (latent test not run) | | | | | | |

No run meets the content criterion (mean AND shuffle significant). Two isolated single-metric mean effects (Musical A10 HR@5, L8a NDCG@10, ≈0.001) come with n.s. shuffle at every cutoff. Mask hurts strongly in A10 and L8b (activation dependence). L8a and WIDER barely need the slot at all. At the decoder level (Musical, 2,000 rows), A10's swap KL is 0.00053 vs 0.073 for dropping one history item (ratio 0.0073).

### 3. Geometry: collaborative losses do reshape `r`

| | CDs v4 plain R3 | CDs A10 | Musical A10 | Musical L8a | Musical L8b | Toys R3_ori | Toys WIDER 355749 |
|---|---:|---:|---:|---:|---:|---:|---:|
| effective rank | 21.3 | **111.0** | 41.0 | 30.5 | 48.6 | 1.00 | **232.6** |
| random-pair cosine | 0.885 | 0.359 | 0.693 | **0.352** | 0.440 | 1.000 | 0.002 |
| same-user / same-target cosine | 0.911 / 0.892 | 0.818 / 0.492 | 0.896 / 0.773 | 0.800 / 0.507 | 0.761 / 0.521 | — | — |
| adjusted η² user / target | 0.173 / 0.057 | **0.476 / 0.156** | 0.354 / 0.206 | 0.382 / 0.189 | 0.327 / 0.150 | — | — |

For WIDER the cosine shown is mean pairwise cosine. Its pooling moves to 100% history titles, and ‖r_A − r_B‖/‖r_A‖ = 1.46, yet swap KL is 0.0014 vs 0.067 for dropping the last item.

### 4. D1: queue drift (Toys, gate 0.5; stale entry at step t vs fresh at t + 100)

| t → t+100 | raw cos | user-head cos | target-head cos | rank corr | top-16 overlap |
|---|---:|---:|---:|---:|---:|
| 100 → 200 | 0.600 | 0.681 | 0.744 | 0.765 | 0.590 |
| 300 → 400 | 0.829 | 0.881 | 0.814 | 0.910 | 0.772 |
| 500 → 600 | 0.937 | 0.962 | 0.869 | 0.971 | 0.867 |
| 700 → 800 | 0.996 | 0.997 | 0.991 | 0.998 | 0.961 |

- The queue spans ~80–100 optimizer steps at gate 0.5.
- At longer ages (D = 600–700 from step 100), the target-head cosine is ≈0.0.
- The fresh same-user − random gap is 0.85–0.88 at every checkpoint.

## Reading
- **Geometry changes, predictions do not.** Collaborative InfoNCE does what it is designed to do on the deployed `r`:
  - CDs effective rank 21.3 → 111.0;
  - random-pair cosine 0.885 → 0.359;
  - user structure ×2.8, target structure ×2.7.
  Queue variants (L8a) and WIDER spread `r` further (rank 233). None of this reaches the prediction: ranking stays at plain R3, and mean or another user's `r` works as well as the own one on CDs, Musical and Toys.
- **Why.** The decoder reads the history directly and treats `r` as redundant (E01, H1). Reshaping a vector the decoder ignores cannot change the output. The strongest case is WIDER: `r` became fully user-specific, and the decoder stopped needing even its presence.
- **Not the queue.** Queue staleness (D1) is real in epoch 1 (rank correlation 0.77–0.91; target head most fragile) and negligible in epoch 2. A momentum encoder would fix the wrong problem; the limit is decoder use, not pair quality.
- **Therefore C1 comes first.** Collaborative grounding can only matter once the decoder actually uses `r`'s content.
- **Caveats:**
  - mostly single seeds;
  - A10 CDs has two checkpoint hashes (provenance gate) and conflicting reference numbers;
  - the Toys A10 original's gate is inferred;
  - the best Toys run (gate 0.5) and Attempt 11 have no latent test.

## Paper use
**Motivation (second panel of the motivation figure/table):**
- CDs A10: effective rank 21.3 → 111.0 and user η² 0.173 → 0.476, vs latent test mean −0.0005 / shuffle −0.0011 HR@10 (n.s.) and mask −0.0284 (sig). Use it as a "geometry vs use" bar pair.
- Add the Musical A10 and Toys A10 rows of Table 2 as replication.
- WIDER 355749 (rank 233, all latent tests n.s., HR@10 0.1070 ≈ 0.1052) as the extreme case.

**Appendix:** L8a/L8b, Attempt 11 ranking, D1 drift table.

**Dev only:** buggy WIDER 355111; Toys gate sweep.

## Sources
- `analysis/document/research_notes/Latent reasoning for recommendation/research_log.md`:
  - status board (lines 7–37);
  - L1/L2 and diagnosis (lines 106–139);
  - L8 (lines 474–556);
  - A10 Toys rerun (lines 691–705);
  - A10 CDs reproduction (lines 707–717);
  - scratch purge and D1 (lines 719–755).
- `analysis/document/avoid_reasoning_collapse_and_shortcut_learning.md`: §10 (Attempt 10: design, 283515 ranking, 292397 geometry, 296325 latent test) and §11 (Attempt 11, 304495).
- `analysis/document/collaborative_reasoning.md` (design and provenance gate).
- `analysis/document/decoder_insensitivity_to_latent.md` §1 and §4 (collaborative InfoNCE column).
- `analysis/document/research_notes/Latent utilization/research_log.md`: WIDER K = 1 (lines 1320–1345, 1563–1622).
- Result files:
  - `analysis/results/intervention/{latent_collaborative_infonce, musical_instruments_collaborative_infonce, toys_and_games_collaborative_infonce, musical_instruments_collab_queue_gate, musical_instruments_collab_queue_diverse, toys_and_games_wider_0.1_fixed}/`;
  - `analysis/results/drift/toys_acc05.json`;
  - `R3/output_dir/wider/Toys_and_Games_wider_0.1_fixed/`.
