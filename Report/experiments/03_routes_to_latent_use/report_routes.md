# E03 — Simpler routes to latent use: bottlenecks, re-pooling, extra information
Back to [main plan](../../main.md). Dates: Attempt 1 (CDs, before 2026-10) and 2026-10-14 to 2026-11-12. Status: done (CVAE V2 never produced results).

## Question
Before and alongside MW-R-H we tried simpler ways to make R3's decoder use the **content** of `r`. Do any of them get
content use (mean AND shuffle significantly hurt) at ranking ≥ R3?

The prediction from redundancy (`r = f(x)` holds nothing the visible history lacks) is that every method lands in one
of two corners:
- it hides the history, so `r` gets used and ranking drops; or
- it keeps the history visible, so ranking holds and `r` goes unused.

## Setup
- **Recipe:** the R3 recipe unless noted.
  - Base Qwen2.5-1.5B (Musical: -Instruct), lr 3e-4, batch 128, ≤ 3 epochs, early stopping, constrained 10-beam
    ranking.
- **Latent tests:** `experiment.py generate` (raw / mask / mean / shuffle_user), or a method-specific equivalent (delete,
  drop, none, shuffle of the side signal).
- **Significance:** paired bootstrap, 2,000 resamples, seed 42, on all test rows.
- **Baselines.** Each run is compared with the baseline its log section used:
  - **Toys:** R3_ori stage 1 0.1023 / 0.0659; plain R3 retrain A (345857) 0.1052 / 0.0667; T2 constant token (338745)
    0.1064 / 0.0678; plain R3 v0 0.1061 / 0.0678; no-latent SFT (344012) 0.1045 / 0.0658.
  - **Musical:** plain R3 v1 0.1226 / 0.0950.
  - **CDs:** R3_ori 0.0989 / 0.0717 (355050); v4 0.1152 / 0.0806 for Attempt 1.
- **Test n:** Toys 6,738, CDs 6,157, Musical 8,313. P1 used the first 1,500 Musical test rows.
- **Only bootstrap CIs were recorded for these runs; no paired t-test was run.** "−" means no comparison was recorded.
  "sig" / `*` means the 95% CI excludes 0.

## Results

### A. Bottlenecks that hide (part of) the history from the target

| attempt | run | data | HR@10 / NDCG@10 | Δ vs baseline HR@10 / NDCG@10 | latent test (raw − mode, HR@10) | verdict |
|---|---|---|---|---|---|---|
| L7 latent-only bottleneck (target sees instruction prefix + `r` only) | 325712 | Musical | 0.1085 / 0.0827 (E1 raw 0.1089 / 0.0828) | −0.014 vs v1 0.1226 (**not bootstrapped**, different checkpoint) | mean +0.0794*, shuffle_user +0.0875* | used, ranking −11% |
| Continuous thought K=1 + bottleneck | 338742 | Toys | 0.0801 / 0.0439 | −0.0263* / −0.0239* vs T2 | shuffle +0.0751*; mean → 0.0043 | used, ranking −25% |
| Continuous thoughts K=8 + bottleneck (K=4 338743 collapsed, 0.0033) | 338744 | Toys | 0.0819 / 0.0445 | −0.0245* / −0.0234* vs T2 | shuffle +0.0772* | used, ranking −23% |
| Gist bottleneck M=4 | 343806 | Toys | 0.0899 / 0.0510 | −0.0165* / −0.0168* vs T2 | delete +0.0861* (→ 0.0039) | used, ranking −16% |
| Gist bottleneck M=16 | 343968 | Toys | 0.0911 / 0.0517 | −0.0153* / −0.0161* vs T2; = M=4 (n.s.) | delete +0.0889* (→ 0.0022) | used, ranking −15% |
| Soft bottleneck, history attention bias b = −4 | 353055 | Toys | 0.1002 / 0.0626 | −0.0021 n.s. / −0.0033* vs R3_ori; −0.0062* vs T2 | mask +0.0180*, mean +0.0006, shuffle +0.0001 (n.s.) | unused (= constant baseline 353056, 0.1023) |
| Recency split m=1 (last item visible, older items only via `r`) | 353011 | Toys | 0.1043 / 0.0701 | +0.0021 n.s. / +0.0043* vs R3_ori; −0.0021 n.s. vs T2 | mean +0.0000, shuffle +0.0001 (n.s.) | unused (> constant baseline 353012 0.0981, +0.0062*) |
| B1 steered reading (`r` selects 2 history items, rest deleted) | 343006 | Toys | 0.0824 / 0.0481 | −0.0240* / −0.0197* vs T2 | shuffle −0.0021 (n.s.; shuffle slightly better) | unused, ranking fails; selector bypasses `r` |
| Staged training, 1 latent-only + 2 full epochs | 353228 | Toys | 0.1008 / 0.0600 | −0.0015 n.s. / **−0.0059*** vs R3_ori | mask +0.0341*, mean +0.0401*, shuffle +0.0567* (run log 353734); v2 D 2.20 nats (353831) | used; HR@10 holds, NDCG@10 drops |
| Staged training, 2 latent-only + 1 full epoch | 353229 | Toys | 0.0870 / 0.0477 | −0.0153* / −0.0182* vs R3_ori | mask +0.0329*, mean −0.0003, shuffle +0.0001 (n.s.; run log 353741); D 0.007 | unused, ranking fails |
| Attempt 1 attention curriculum (reasoning_only / history_only batches → both) | 248939 / 249601 | CDs | 0.0952 / 0.0685 | not recorded vs v4 0.1152 | mask −0.0097*, mean −0.0013, shuffle −0.0010 (n.s.) | unused (mode-conditional policy) |
| Attempt 1-A, alpha capped at 0.75 (never trains "both") | 257759 | CDs | 0.1038 / 0.0712 | not recorded vs v4 0.1152 | mean −0.0070*, shuffle −0.0068*; mask n.s. | used, ranking below v4 (no CI) |
| Attempt 1-B, 2 epochs alpha 0.75 + 1 epoch three-way | 283592 / 292536 | CDs | 0.1007 / 0.0722 | not recorded vs v4 0.1152 | mask +0.0109*, mean −0.0002, shuffle +0.0010 (n.s.) | unused: one joint epoch erases dependence |
| History dropout (option C), item p = 0.3 / 0.5 / 0.7 | 358573 / 358574 / 358575 | Toys | 0.1042 / 0.0629; 0.0988 / 0.0592; 0.1011 / 0.0598 | not recorded vs plain v0 0.1061 (log: 2–7% below) | mask +0.0275* / +0.0430* / +0.0368*; mean, shuffle n.s. at every p | unused; ranking at eval_p = p 0.0861 / 0.0644 / 0.0562 |

**L7 row-level analysis** (Musical, analysis only, `analysis/results/bottleneck_vs_r3/`):
- L7 (0.1089) is compared with plain R3 v0 ckpt-1038 (0.1219). The whole gap is on copy-like rows.
- On targets bought the same day as the last history item (3,976 rows), R3 scores 0.2002 and L7 0.173 (contribution
  −0.013). On the other 4,337 rows both score 0.0500.
- On same-brand-as-last rows (12.7%), L7 loses 0.074.
- So the bottleneck loses exact-identity copying, not preference prediction. The same holds for gist M=16 on Toys: 87%
  of its gap vs T2 is on rows whose brand word appears in the history.

### B. Re-pooling or reshaping `r`, history visible

| attempt | run | data | HR@10 / NDCG@10 | Δ vs baseline HR@10 / NDCG@10 | latent test (raw − mode, HR@10) | verdict |
|---|---|---|---|---|---|---|
| History-only pooling (keys = history title tokens) | 353334 (interventions 353953) | Toys | 0.1036 / 0.0646 | +0.0013 / −0.0012 vs R3_ori (n.s.) | mask +0.0095*, mean +0.0013, shuffle +0.0012 (n.s.); v2 D 0.0095 | unused (`r` user-specific, pair cos 0.38) |
| SASRec-informed pooling, title / sasrec candidates (fine-tuned from R3) | 345719 / 345720 | Toys | 0.1030 / 0.0650; 0.1042 / 0.0649 | +0.0007 / +0.0019 vs R3_ori (n.s.) | title: drop −0.0021* (dropping helps), shuffle n.s.; sasrec: drop +0.0019*, shuffle +0.0016* | ≈ unused; gain also with candidates dropped |
| Forced attention share A2 (ρ = 0.3 fixed) | 354497 | CDs | 0.0968 / 0.0697 | −0.0021 / −0.0020 vs R3_ori (n.s.) | mask +0.0335*, mean −0.0002, shuffle +0.0002 (n.s.) | unused; `r` becomes a sink (user part 33% → 11%) |
| Attention share B2 (budget 0.3) | 354390 | CDs | 0.0976 / 0.0695 | −0.0013 / −0.0023 (n.s.) | mask +0.0344*, mean +0.0000, shuffle −0.0002 (n.s.) | unused |
| Attention floor C2 (regulariser ρ̄ 0.3, λ 10; 32× grad-accum bug, not rerun) | 354567 | CDs | 0.1004 / 0.0715 | +0.0015 / −0.0002 (n.s.) | mask +0.0250*, mean +0.0003, shuffle +0.0003 (n.s.) | unused; `r` near-constant (0.5%) |
| DivReason N=4 (SFT stage) | 343405 | Toys | 0.1079 / 0.0682 | +0.0013 / +0.0003 vs plain matched 343065 (n.s.) | mean −0.0010, shuffle −0.0004, single path −0.0009 (n.s.) | unused |
| DivReason + sqrt-JS path diversity | 345579 | Toys | 0.1012 / 0.0633 | −0.0052* / −0.0045 vs T2; −0.0067* vs DivReason | mean −0.0007, shuffle −0.0012 (n.s.); path0 alone +0.0049* | path-sensitive, content unused |
| Two-head fusion (fast bypass + slow gist head) | 344555 | Toys | fused 0.1026 / 0.0646 | −0.0019 vs no-latent (n.s.); fused − fast +0.0031 (n.s.) | fused_delete +0.0245* (broken slow head corrupts the sum); no swap test | no benefit from the latent |
| Guidance scaffold A (plain R3 retrain) | 345857 | Toys | 0.1052 / 0.0667 | +0.0030 vs R3_ori (n.s.) | not run | reference |
| Guidance scaffold B (SASRec candidates) / C (random) / D (annealed to 0) | 345883 / 346494 / 348523 | Toys | 0.1033 / 0.0662; 0.1043 / 0.0655; 0.1023 / 0.0651 | B − A −0.0019, C − A −0.0009, D − A −0.0030 (all n.s.) | B: shuffle +0.0010, drop −0.0013 (n.s.); C: drop +0.0001 (n.s.) | pooling un-collapses, decoder still ignores `r` |
| WIDER K=1, corrected (λ_eff 0.1) | 355749 | Toys | 0.1070 / 0.0669 | +0.0018 / +0.0002 vs A (n.s.) | mask +0.0009, mean −0.0007, shuffle −0.0009 (n.s.) | unused (effective rank 233) |
| CVAE latent (option A), without / with dropout 0.5 | 359244 / 359245 | Toys | not recorded | — | — | **failed** (`KeyError: 'target_item'` in `cvae_latent.py`, run logs); no result |

### C. Giving `r` information the history lacks

| attempt | run | data | HR@10 / NDCG@10 | Δ vs baseline HR@10 / NDCG@10 | latent test (HR@10) | verdict |
|---|---|---|---|---|---|---|
| Category latent, additive, trained with the oracle category; eval oracle | 349156 | Toys | 0.3776 / 0.2260 | ceiling (not deployable) | another user's category 0.0218 (350065) | used strongly |
| Same, eval with LLM top-1 category (pred1) | 350064 | Toys | 0.0914 / 0.0554 | −0.0108* / −0.0105 vs R3_ori; −0.0138* vs A | right rows 0.3847 vs R3 0.2960; wrong rows 0.0227 vs 0.0568 | over-trust, ranking fails |
| Same, mixture over top-5 categories (mix5) | 350066 | Toys | 0.0910 / 0.0573 | −0.0113* / −0.0085 vs R3_ori | = pred1 (n.s.) | ranking fails |
| Category inject (IntuRec-style list of 5) | 349157 / 350070 | Toys | pred 0.1049 / 0.0658 | −0.0003 vs A (n.s.) | oracle 0.1061, shuffle 0.1036 (n.s.) | unused |
| Item oracle (target item's vector): r3title / qwen3 / sasrec / charac | 353428 / 353429 / 353430 / 353431 | Toys | oracle 0.9617 / 0.9512 / 0.9433 / 0.9482 | ceiling | shuffle 0.0036 / 0.0027 / 0.0043 / 0.0030; SASRec top-1 item (pred1) 0.0684 / 0.0671 / 0.0662 / 0.0755 | lookup copying; unusable with a predicted item |
| SASRec history vector in `r` (sasrec_raw) | 354714 | Toys | 0.1049 / 0.0661 | −0.0003 vs A (n.s.) | shuffle +0.0101*, drop +0.0021 (n.s.) | weakly used, no gain |
| Transition-neighbour latent v1 (saturated reader) | 356391 | Toys | 0.1052 / 0.0662 | +0.0000 vs A (n.s.) | none +0.0010, shuffle +0.0004, shuffle_r +0.0000 (n.s.) | unused |
| Transition latent v2 (fixed reader) | 357936 | Toys | 0.1070 / 0.0676 | +0.0018 vs A (n.s.) | none +0.0016*, shuffle_r +0.0019*, shuffle +0.0012 (n.s.) | weakly used, no gain |
| Transition latent + CARE negatives | 357937 | Toys | 0.1058 / 0.0670 | +0.0006 vs A (n.s.) | none / shuffle / shuffle_r n.s. | unused |
| Transition latent v3 (reader lr ×10, r3 → reader → joint) | 358208 | Toys | 0.1034 / 0.0642 | −0.0018 vs A (n.s.); NDCG@10 −0.0034* vs v2 | none +0.0012, shuffle −0.0003 (n.s.) | unused (reader top-1 14%) |
| Grounded donors L4 | 323041 | Musical | 0.1203 / 0.0938 | not bootstrapped vs v1 0.1226 | all 5 donor controls within ±0.001 (n.s.) | no effect |
| Grounded donors L5 (grounding fixed at 0.5 by bf16) | 323042 | CDs | 0.1223 / 0.0843 | +0.0067* vs L2 (HR only) | knn − shuffle_outcome +0.0023*; knn − random +0.0008 (n.s.); 11-04 re-check: random − shuffle_outcome +0.0015* | small; behaviour-specific claim weakened |
| Grounded donors L10 | 332484 | Toys | 0.1066 / 0.0688 | not bootstrapped vs v0 0.1061 | knn − off +0.0019*; − shuffle_outcome +0.0006 (n.s.) | weak claim only |
| Diffusion-refined thought (r' = diffuse(r) → r3title target) | 354573 | Toys | 0.0978 / 0.0551 | not recorded (R3_ori 0.1023) | mean +0.0013, shuffle +0.0018 (n.s.); oracle 0.6903; x̂0 nearest-item HR@10 0.0261 | unused |
| Same, regression control | 354956 | Toys | 0.0969 / 0.0562 | not recorded | mean +0.0019, shuffle +0.0000 (n.s.; NDCG mean +0.0011*); x̂0 nearest HR@10 0.0939 | unused |
| TRM recursive refine (8 cycles × 6) | 363429 | Toys | 0.0975 / 0.0581 | not recorded | shuffle +0.0025*, mean +0.0018 (n.s.; NDCG +0.0014*); oracle +0.0001 (n.s.) | weakly used, ranking below R3 (no CI) |
| CFG probe P1 on L7 (no training), w = 1.5 / 3 / 5 | 341711 | Musical, 1,500 rows | raw 0.1360 / 0.1084; w 1.5: 0.1400 / 0.1098; w 3: 0.1260 / 0.1015; w 5: 0.1127 / 0.0793 | w 1.5 +0.0040 (n.s.); w 3 NDCG −0.0068*; w 5 −0.0233* / −0.0290* | under w = 3: mean +0.0987*, shuffle +0.1093* | promotion gate failed |

The staged-training intervention rows and the diffusion, regress and TRM numbers come from `run_logs/`; they are not in
the research log.

### D. Trade-off table from the log (2026-11-09, reproduced as written)

| family | run (dataset) | HR@10 | Δ vs baseline | shuffle / mean ΔHR@10 | verdict |
|---|---|---:|---:|---|---|
| structural bottleneck | L7 latent-only (Musical) | 0.1085 | −0.014 (−11%) sig | −0.088 / −0.079 sig | used, ranking fails |
| structural bottleneck | K=1 thought + bottleneck (Toys) | 0.0801 | −0.026 (−25%) sig | −0.075 sig | used, ranking fails |
| structural bottleneck | K=8 thoughts + bottleneck (Toys) | 0.0819 | −0.025 (−23%) sig | −0.077 sig | used, ranking fails |
| gist bottleneck | M=4 / M=16 (Toys) | 0.0899 / 0.0911 | −0.016 (−15%) sig | delete → 0.004 | used, ranking fails |
| soft bottleneck | history bias b=−4 (Toys) | 0.1002 | −0.005 | n.s. | unused |
| partial bottleneck | recency split m=1 (Toys) | 0.1043 | −0.001 n.s. | exactly 0 | unused |
| staged attention | latent-only → full (1+2) (Toys) | 0.1008 | −0.002 n.s. (NDCG −0.006 sig) | per-sample D 2.20 nats (HR interventions not completed) | sensitive, ranking borderline |
| staged attention | full → latent-only (2+1) (Toys) | 0.0870 | −0.015 sig | D 0.007 | unused, ranking fails |
| latent reshaping | Attempt 10 / L1 / L8a / L8b InfoNCE (Musical) | 0.1208–0.1223 | ≈ 0 | ≤ 0.0015 n.s. | unused |
| latent reshaping | WIDER K=1 corrected (Toys) | 0.1070 | +0.002 n.s. | n.s. | unused (diverse but unread) |
| latent reshaping | history-only pooling (Toys) | 0.1036 | −0.002 n.s. | n.s. | unused |
| latent reshaping | DivReason path diversity (Toys) | −0.005 | n.s. | path-sensitive, content n.s. | unused |
| extra signal in r | SASRec history vector (Toys) | 0.1049 | ≈ 0 | shuffle −0.010 sig, drop n.s. | weakly used, no gain |
| extra signal in r | transition-neighbour latent v2 (Toys) | 0.1070 | +0.002 n.s. | none −0.0016 / shuffle_r −0.0019 sig | weakly used, no gain |
| extra signal in r | two-head fast + slow gist (Toys) | 0.1026 | −0.003 n.s. | slow head alone 0.0705 | unused |
| decoder objective | margin only (Attempt 2, Musical) | 0.1176 | −0.005 (−4%) | sensitivity ×9 (HR interventions not run) | used, small ranking cost |
| **decoder objective + reshaping** | **margin + WIDER 356322 (Toys)** | **0.1030** | **−0.002 n.s.** | **−0.073 / −0.031 sig** | **both criteria met** |
| + collaborative | MWC-G donor grounding (Toys) | 0.1064 | +0.001 n.s. | −0.068 / −0.068 sig (via r0; donors unused) | both met, no collaborative effect |
| + collaborative | MWC-I user+target InfoNCE (Toys) | 0.0966 | −0.009 sig | −0.075 / −0.077 sig | used, ranking fails |

Corrections to the table, from the sources:
- The L7 "sig" was never bootstrapped (old log: "not bootstrapped, different checkpoint").
- The soft-bottleneck −0.005 is vs T2 (−0.0062*). Vs R3_ori it is −0.0021 n.s.
- The DivReason row has its columns shifted. Run 345579 scores HR@10 0.1012, which is −0.0052* vs T2.
- The staged 1+2 HR interventions did complete (353734): mean −0.0401*, shuffle −0.0567*.

### E. All cutoffs for the most-cited runs (HR@1 = NDCG@1; @20 = @10)

| run | HR@1 | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L7 325712 (Musical) | 0.0636 | 0.0788 | 0.0909 | 0.1085 | 0.0721 | 0.0770 | 0.0827 |
| K=1 thought + bottleneck 338742 (Toys) | 0.0180 | 0.0370 | 0.0497 | 0.0801 | 0.0288 | 0.0340 | 0.0439 |
| gist M=16 343968 (Toys) | 0.0236 | 0.0438 | 0.0595 | 0.0911 | 0.0351 | 0.0415 | 0.0517 |
| staged 1+2 353228 (Toys) | 0.0288 | 0.0543 | 0.0738 | 0.1008 | 0.0434 | 0.0513 | 0.0600 |
| recency split m=1 353011 (Toys) | 0.0419 | 0.0678 | 0.0830 | 0.1043 | 0.0570 | 0.0632 | 0.0701 |
| history-only pooling 353334 (Toys) | 0.0341 | 0.0592 | 0.0757 | 0.1036 | 0.0489 | 0.0556 | 0.0646 |
| WIDER K=1 fixed 355749 (Toys) | 0.0350 | 0.0628 | 0.0790 | 0.1070 | 0.0512 | 0.0578 | 0.0669 |
| sasrec_raw 354714 (Toys) | 0.0367 | 0.0608 | 0.0779 | 0.1049 | 0.0504 | 0.0574 | 0.0661 |
| category pred1 350064 (Toys) | 0.0276 | 0.0512 | 0.0650 | 0.0914 | 0.0412 | 0.0469 | 0.0554 |
| R3_ori stage 1 (Toys) | 0.0377 | 0.0607 | 0.0778 | 0.1023 | 0.0510 | 0.0580 | 0.0659 |
| plain R3 retrain A 345857 (Toys) | 0.0367 | 0.0625 | 0.0795 | 0.1052 | 0.0514 | 0.0584 | 0.0667 |

All cutoffs for the other runs are in the log sections listed under Sources.

## Reading
1. **Two failure corners.**
   - **Hide the history and `r` is used, but ranking drops.** This covers L7, the K-thought and gist bottlenecks, and
     B1, at −11% to −25% HR@10.
     - More latent capacity does not help: K=8 = K=1, M=16 = M=4.
     - The loss sits on copy rows (L7 row analysis, gist brand split): a pooled vector cannot carry exact title
       identity.
   - **Keep the history visible and ranking holds, but `r` goes unused.** This covers the soft bottleneck, recency
     split, history dropout, history-only pooling, attention share, WIDER, DivReason, guidance, transition and
     diffusion.
     - Mask often hurts (the slot must hold some vector), but mean and shuffle are n.s.
   - **Forcing `r` to be read or to be diverse does not create use.** Under forced attention, `r` turns into a
     constant sink (A2/C2). Under WIDER, `r` becomes diverse but unread.
2. **Extra information is used only when it is unambiguous, and then the decoder over-trusts it.**
   - The oracle category (0.3776) and oracle item (≥ 0.94) are used strongly.
   - Their deployable versions (category pred1 0.0914, item pred1 ≤ 0.0755) fall below R3.
   - Realistic side signals give ≤ 0.002 of content effect and no gain (SASRec vector, transitions, donors).
3. **Soft or partial variants slide back to "unused."**
   - Any setup where the decoder still sees the history during part of training or inference tends to lose
     dependence: the soft bias, recency split, Attempt 1-B, staged 2+1 and dropout.
   - Staged 1+2 is the exception: it keeps dependence (mean −0.040*, shuffle −0.057*) with HR@10 n.s. vs R3_ori.
     But NDCG@10 is significantly lower (−0.0059), it is a single seed, and it was only compared with R3_ori.
4. **Why this motivates the hinge.**
   - Routing and side information both act on what `r` *contains* or what the decoder *can see*. Neither gives the
     decoder a reason to prefer the correct `r` while the history is visible.
   - A history-swap hinge acts on the *effect* of `r`: it penalises a decoder that scores the target as well with
     another user's `r` as with its own. It works with the history in view and with identical masks in training and
     test.
   - MW (356322) is the first run in the table to meet both criteria. MW-R-H (E05/E06) then weights the hinge away
     from history-explained rows, which is where the bottlenecks lose.
5. **Caveats.**
   - Every run is a single seed.
   - Bootstrap CIs cover test sampling only. Seed variance is about 0.002 HR@10 (Musical) or more (Toys MW-R-H).
   - No paired t-tests were recorded.
   - Baselines differ across rows (R3_ori, A, T2, v0, v1).
   - The diffusion, regress and TRM rows have no recorded comparison with R3.
   - C2 ran with the 32× grad-accum bug.
   - L4, L5 and L10 have a grounding scale frozen by bf16.

## Paper use
- **Related work / motivation** (main text: one paragraph and a small table): the two-corner picture. Use the reproduced
  trade-off table (D, with the corrections), shortened to one row per family: L7, gist M=16, recency split,
  history-only pooling, attention share A2, WIDER K=1, sasrec_raw, category pred1, MW.
- **Appendix table:** all rows of A–C with run IDs, the all-cutoff table E, and the L7 copy-row analysis as evidence
  that the bottleneck's cost is copying.
- **Appendix (negative results):** DivReason reproduction (343405 = plain R3), two-head fusion, guidance scaffold,
  diffusion / TRM, P1.
- **Do not cite as results:**
  - CVAE V2 (no result);
  - C2 (grad-accum bug);
  - the L7 −0.014 as "sig";
  - L5 as evidence of behaviour grounding without the 11-04 caveat.

## Source notes (gaps and contradictions found when compiling)
- **Trade-off table:** see the corrections under D. The table says "HR interventions not completed" for staged 1+2,
  but they did complete (353734). Staged 2+1 (353741) was also never logged: mean and shuffle n.s.
- **Results that exist only in run logs:**
  - Diffusion 354573 (0.0978 / 0.0551) has an empty table in the log.
  - Regress 354956 (0.0969 / 0.0562) and TRM 363429 (0.0975 / 0.0581, shuffle +0.0025*) have no log entry.
  - None of these three has a recorded bootstrap vs R3.
- **CVAE V2 (359244 / 359245):** listed as "submitted" on the status board, but both failed with
  `KeyError: 'target_item'`. There are no results.
- **Attempt 1:** CLAUDE.md says the curriculum leaves `r` ignored at test time. That holds for the original schedule
  and for follow-up B, but follow-up A (alpha 0.75, 257759) shows content use (mean −0.0070*) at HR@10 0.1038 vs v4
  0.1152. No CI vs v4 was recorded for any Attempt 1 run.
- **L5:** the "first significant behaviour-grounding evidence" was weakened by the 11-04 re-check (random −
  shuffle_outcome +0.0015*, knn ≈ random). L4 and L10 also have no bootstrap vs plain R3.
- **No paired t-tests** were recorded for any E03 run, although CONVENTIONS asks for them. History dropout has no CI vs
  plain R3; the log only says it is "2–7% below".
- **Margin-only control on Toys:** never run, so "WIDER removed the margin's cost" rests on a cross-dataset comparison
  (the log flags this).
- **Naming:** B1 steered reading (343006) deletes history items, so it is under bottlenecks. SASRec-informed
  (candidate) pooling (345719 / 345720) is a separate pair of runs, under re-pooling.

## Sources
- **Research log** `R3/analysis/document/research_notes/Latent utilization/research_log.md`:
  - status board (V1, V2); T1–T3 (l.173–209); P1 (l.232–266); B1 (l.270–369);
  - DivReason (l.310–401, 668–715); gist (l.411–528); two-head (l.530–563);
  - candidate pooling (l.639–736); guidance A–D (l.651–806); category latent (l.808–844);
  - recency split / soft bottleneck (l.874–1019); staged training and history pooling (l.911–1034, 1182–1227);
  - item oracle (l.978–1097, 1229–1271); sasrec_raw (l.1160–1308); attention share (l.1143–1561);
  - WIDER (l.1563–1621); L7 row analysis (l.1623–1632);
  - transition latents (l.1642–1745, 1800–1817); history dropout (l.1761–1798); trade-off table (l.1937–1965).
- **Older log** `.../Latent reasoning for recommendation/research_log.md`: L4/L5 (l.141–330), L7 (l.426–470), L10
  (l.658–690).
- `R3/analysis/document/avoid_reasoning_collapse_and_shortcut_learning.md`, Attempt 1 §1.2 (l.103–275).
- **Run logs:**
  - `R3/run_logs/R3_staged_l1f2_eval.353734.out`, `R3_staged_l2f1_eval.353741.out`;
  - `R3_diffusion_diffusion.354573.out`, `R3_diffusion_regress.354956.out`;
  - `R3_recursive_refine.363429.out`;
  - `R3_var_cvae.359244.out`, `R3_var_cvae_drop0.5.359245.out`.
- **Result dirs:**
  - `R3/output_dir/{gist_results,continuous_thoughts_results,steer_results,divreason_results,two_head_results,guidance_scaffold,candidate_pooling,category_latent,item_oracle,sasrec_latent,transition_latent,wider,attention_share,diffusion_refine,recursive_refine,variational}/`;
  - `R3/analysis/results/{intervention,bottleneck_vs_r3}/`.
