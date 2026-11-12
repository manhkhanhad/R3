# Prototype-Guided Reasoning

**Status: designed, not yet started.** A retrieval-augmented mechanism for R3's latent thought vector `r`, independent of the μ/z decomposition line (Attempts 6–8, `avoid_reasoning_collapse_and_shortcut_learning.md`). Written up for later implementation; not yet queued.

## Motivation

Every mechanism tried against latent-reasoning collapse so far (schedule, margin, contrastive, gradient mask, KL/anchor, memory-bank alignment — see the main investigation doc) works by *constraining* `r`: penalizing the decoder's access to raw history, or shaping `r`'s geometry. None of these give `r` information it didn't already have — `r` is a pooled readout of the same history tokens the decoder attends to directly, so it is structurally a compression of `x`, never an addition to it. That structural fact is the reason every constraint-based fix trades off against raw task performance to some degree (quantified in the main doc's x-path decomposition table).

Prototype-guided reasoning instead gives `r` access to information that is not derivable from the current user's history at all: **verified, cross-user reasoning patterns**, mined and quality-checked offline by a sibling project. This is a genuine information advantage, not a constraint, so it does not carry the same structural tax — and it is a different, independent line from Attempt 9's baseline-preserving latent residual correction (which learns only from the current history; this gives inference-time, cross-instance, pattern-level signal).

## Data source

`research/LLMReasoningPrototype` — a sibling project (RALR: Retrieval-Augmented Latent Reasoning for Recommendation). Its offline pipeline (see `docs/evidence_bank_generation.md`) mines and verifies reasoning patterns in three stages:

1. **Generate**: per-instance, LLM-generated (`Qwen/Qwen2.5-7B-Instruct`), preference-framed, grounded facts explaining a known positive interaction.
2. **Cluster**: facts embedded, UMAP + HDBSCAN, one canonical rationale regenerated per cluster.
3. **Verify**: the canonical rationale is tested on held-out instances from its own cluster — can it rank the true positive above random negatives? `quality_score` = pass rate; clusters that fail a minimum pass rate are discarded, not kept at reduced quality.

Both stages have been run for this dataset. Verified output: `research/LLMReasoningPrototype/data/evidence_bank/CDs_and_Vinyl_5_2015-10-2018-11_train_evidence_bank.jsonl`.

**Checked contents**: 514 verified pattern-level records. Each has:
```json
{
  "id": "ev_000004",
  "facet": "kg_path",                          // or "general"
  "anchor": {"shared_genre": "Pop"},            // structural fields, may be {}
  "rationale": "listeners who favor pop tend to also enjoy catchy melodies and nostalgic, timeless collections",
  "support": 1908,                              // cluster member count
  "quality_score": 0.8,                         // verification pass rate
  "embedding": [...]                            // 768-dim
}
```
`quality_score` ranges 0–1, mean 0.467 — a substantial fraction of clusters are only weakly verified, so quality-weighting at retrieval time (not just relevance) matters, not an optional refinement.

## Mechanism

`r` (R3's existing `SelfAttentionLayer` pooled output, unchanged) plays the role of a **query** into the verified prototype bank. Retrieval surfaces individual prototype evidence vectors, which are **concatenated onto the decoder's input** alongside `r` and `x` — not blended into a replacement for `r`.

1. **Multi-aspect query projection**, not a single `proj_down`. Project `r_i` through `n` separate, orthogonally-initialized heads:
   ```
   q_i^(a) = W_a r_i   for a = 1, ..., n
   ```
   each a learned `1536 → 768` map (new parameters; `SelfAttentionLayer` itself is unchanged) into the bank's embedding space. This follows the disentangled multi-projector in FACE (Wang et al., "FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens," arXiv:2510.15729, NeurIPS 2025), which projects a single CF embedding into `n` concept-specific vectors — there to decouple entangled user/item preferences before vector-quantizing each into an LLM token, here to decouple entangled reasoning aspects (e.g. genre affinity, structural/`kg_path` relation, sentiment framing) before retrieval. A single query vector otherwise forces one geometric direction to stand in for every aspect a prototype might match on — the single-head design retrieves whatever's closest overall, not necessarily the *set* of distinct things `r_i` is actually about. Orthogonal init (rather than sharing weights across heads) keeps the heads from collapsing onto the same subspace early in training, per FACE §3.2.2.

2. **Per-head retrieval — MMR within each aspect, not pooled across aspects.** Each head's query `q_i^(a)` runs its own MMR pass over the bank, independently selecting `k_a` prototypes maximizing
   ```
   λ · sim(q_i^(a), e_j) − (1 − λ) · max_{s ∈ selected} sim(e_j, e_s)
   ```
   Retrieving per-head rather than pooling all `n` queries into one shared MMR call is what actually makes the heads specialize: a single shared pool would let whichever head's query happens to align best globally dominate the top-`k`, and the others would just contribute near-duplicates of it — defeating the point of disentangling in the first place. At only 514 candidates this is still brute-force, no ANN index needed, even run `n` times.

3. **Per-prototype projection — no weighted-sum blending, and not CE-supervised alone.** Every retrieved prototype `e_j` (from any head, quality_score attached) is projected individually through one shared, weight-tied map:
   ```
   p_j = proj_up( [ e_j ; quality_score_j ] )
   ```
   into a `1536`-dim vector matching `r`'s dimensionality. `quality_score_j` is concatenated in as a scalar feature rather than used to pre-blend prototypes into a combined vector — the network learns how much to trust it via CE gradient, instead of a hand-tuned softmax deciding upfront.

   `proj_up` needs its own auxiliary supervision beyond downstream CE — see **Cross-space projection collapse** below. The evidence bank's 768-dim space (an independently-trained sentence embedder, unrelated to R3's 1536-dim hidden space) and R3's hidden space share no geometry a priori; asking `proj_up` to learn a faithful mapping between them from downstream-task gradient alone, spread thin over 514 prototypes retrieved with wildly uneven frequency, is a much weaker and more indirect training signal than it looks. Train `proj_up` (paired with a small inverse `proj_down': 1536 → 769` used only during training) with a reconstruction loss against the *full* prototype bank, not just retrieved instances:
   ```
   L_recons = Σ_{j=1}^{514} || proj_down'(proj_up([e_j; quality_score_j])) − [e_j; quality_score_j] ||²
   ```
   This gives every prototype — including the long tail that's rarely or never retrieved during a given epoch — dense, direct supervision to preserve its identity in the projected space, independent of retrieval frequency. FACE's own ablation is direct evidence this matters, not just architectural caution: removing their analogous reconstruction loss (`w/o recons`, Table 3) measurably hurts Recall@20/NDCG@20 on both their datasets, and they note the resulting descriptors "suffer from the lack of diversity" without it — the same failure mode being guarded against here.

4. **Concatenation fusion, not additive/gated.** The decoder's input for instance `i` becomes
   ```
   decoder_input_i = Concat( x_i, r_i, p_1, ..., p_k )
   ```
   where `p_1..p_k` (`k = Σ_a k_a` across all `n` heads) are the projected prototypes as extra positions, and `r_i` passes through **unchanged** — no residual is added into it. This replaces an earlier additive design (`r'_i = r_i + proj_up(Σ_j w_j e_j)`) that ran MMR specifically to select a *diverse* set of prototypes, then immediately undid that diversity by softmax-weighting the set back down into a single blended vector — which regresses toward the set's centroid, exactly the generic, low-information point MMR was selecting *against*. Concatenating instead lets the decoder's own attention (already present via `SelfAttentionLayer`'s architecture) learn per-instance how to weight and combine the individual prototypes, rather than baking a fixed retrieval-time combination rule into the input. If the sequence-length growth from `k` extra positions turns out to be a real cost (decoder compute, or degraded attention over a longer sequence), weighted-sum fusion remains a documented fallback — at the cost of reintroducing the averaging risk above.

5. The decoder consumes the extended sequence directly. `SelfAttentionLayer` (producing `r`) and the E1 pipeline are otherwise unchanged; only the decoder's effective input length grows by `k` positions.

**Why multi-head over single-head here specifically**: the bank already carries an aspect-like structural field (`facet`: `kg_path` vs. `general`, plus the `anchor` dict), so the data has latent aspect structure to disentangle against — this isn't multi-head for its own sake. Multi-head retrieval determines *which* `k_a` prototypes get surfaced per aspect (better recall across distinct things `r_i` might be about); concatenation determines *how* the decoder is exposed to what got surfaced — the two are complementary, not redundant. FACE's ablation (their Fig. 4, descriptor number `n`) is the closest available evidence that disentangling into multiple concept-specific projections beats a single pooled projection, though their setting (quantizing into a large LLM vocabulary) differs enough in scale from ours (MMR over 514 prototypes) that their specific `n` doesn't transfer — see Open implementation decisions.

## Relationship to the μ/z decomposition line

Orthogonal by design — `proj_down`/`proj_up` sit on top of whatever produces `r` (vanilla `SelfAttentionLayer`, or the Attempt 7/8 `mu_head + z_head` encoder). Composable with whichever μ/z configuration wins the Run A/B decision gate (main doc), but not dependent on it; can equally be tested against the vanilla decomposition first as a cleaner single-variable test.

## Known risks

- **Cross-space projection collapse.** `proj_up` has to map a totally foreign 768-dim space (the evidence bank's independently-trained sentence embedder) into R3's 1536-dim hidden space, with no shared geometry between them to start from. Left to downstream CE gradient alone, this is a weak and — worse — *uneven* signal: whichever prototypes get retrieved often (popular/generic ones, likely correlated with higher `support` and non-empty `anchor`) get many gradient updates and a reasonably-shaped projection, while the long tail that's rarely retrieved gets almost none, so most of the 514-entry bank ends up arbitrary or degenerate in projected space rather than meaningfully placed — a subtler, harder-to-notice failure than the query-side head collapse above, since it wouldn't show up as heads converging, just as most prototypes being functionally noise once projected. The reconstruction loss in Mechanism step 3 is the mitigation, built in from the start here (unlike head collapse) because FACE's own ablation (`w/o recons`, Table 3) already shows this specific failure mode costs real performance in their setting — this isn't speculative the way head collapse currently is.
- **Decoder shortcut via prototype tokens.** Concatenation exposes prototype vectors to the decoder the same way `x`/`r` are exposed — if a subset of prototypes turns out to be easier to attend to (or just more predictive by coincidence) than genuine history-conditioned reasoning, the decoder could shortcut through them instead of doing the reasoning `r` is meant to encode. This is the same class of collapse the constraint-based mechanisms in the main doc were built to fight, now reintroduced from a different input rather than solved by construction just because the information source itself is genuine. The `mask`/`mean`/`shuffle` checks in the predicted E1 signature below are exactly what's meant to catch this — concatenation isn't automatically safe just because the added information is real.
- **Head collapse — diagnose via logging, don't pre-build a fix.** FACE itself has no training-time regularization against projection-head collapse, only orthogonal init (§3.2.2) — it isn't ablated or shown to be a problem in their setting, so there's no evidence yet it's a real risk here versus a theoretical one. Log pairwise cosine similarity between `q_i^(a)` across heads (or overlap between per-head retrieved prototype sets) during training as a cheap diagnostic. Only add an explicit orthogonality penalty or diversity-in-retrieval loss if the logs actually show heads converging — don't build that mitigation speculatively before there's a signal it's needed. Note this risk is conditional on whether `L_commit` (Training curriculum, Phase 2) is adopted: without it, `W_1..W_n` never receive gradient at all, so they can't collapse *or* specialize — they'd just stay at orthogonal init.
- **Retrieval quality ceiling, worse per-head than pooled.** 514 prototypes is already small for one MMR pool; splitting attention across `n` independent per-head MMR passes over the *same* 514 candidates means each head's effective candidate pool for genuinely-relevant, non-empty-`anchor`, well-verified prototypes is smaller still. A pre-filter on non-empty anchors plus a `quality_score` floor at retrieval time (needed regardless of head count) matters more here than in the single-head design, not less.
- **No train/test leakage**: retrieval is always against the *train*-split bank, at both train and inference time — the same pattern already used safely for `shuffle_user`/`shuffle_target` reusing `train_vec` in the E1 pipeline.

## Training curriculum, if staged like FACE (§3.4)

FACE's own joint objective (`L = L_R + μL_map + λL_align`) was unstable enough end-to-end that they never trained it from a cold start — a 3-step curriculum (pretrain backbone → train the quantized mapping alone → joint fine-tune) was needed because "simultaneously learning the recommendation backbone parameters from scratch and the discrete LLM mapping function creates conflict." R3's design has a similar shape (an existing task-trained backbone + a new mapping module bridging into a foreign embedding space) but differs in one way worth being explicit about before copying the recipe as-is: FACE's backbone never has to *change* to consume the mapping's output — the frozen pretrained LLM already knows how to read real vocabulary tokens, so nothing on the backbone side needs retraining once the mapping produces valid ones. R3's decoder, by contrast, is being asked to consume `k` entirely new input positions it has never seen — so "freeze the backbone" can't mean the same thing throughout; it only holds before the decoder is exposed to prototype tokens at all.

**Phase 1 — Backbone as-is, no prototype machinery.** Reuse (or continue training) R3's current E1 checkpoint exactly as it exists today: `x`-path, `SelfAttentionLayer` → `r`, decoder consuming `(x, r)` — no `W_1..W_n`, no `proj_up`/`proj_down'`, no concatenated prototype positions. Equivalent to FACE reusing an already-converged GMF/LightGCN/etc. checkpoint as `R`; if a suitable E1 checkpoint already exists from other experiments in this line, this phase is likely "start from that," not a new run.

**Phase 2 — Mapping pretrained alone, backbone frozen, decoder never sees prototypes yet.** Train `proj_up`/`proj_down'` via `L_recons` against the *full* 514-prototype bank (Mechanism step 3) — this needs no forward pass through R3 at all, since it's a direct autoencoder over fixed bank embeddings, so it's cheap and can run standalone before touching the backbone.

This phase is also where the query-side differentiability gap has to be resolved, not deferred: **MMR's top-k selection is a hard argmax, so it blocks gradient — as designed, `W_1..W_n` never receive any training signal at all, at any phase**, since CE loss only ever sees the *identity* of the selected prototypes via `proj_up`, never a differentiable path back to which ones got selected. FACE hits the identical problem in its quantization step (nearest-codeword lookup is also a hard argmax) and solves it with a Straight-Through Estimator plus a commitment loss (`L_Q`, eq. 9) pulling the query toward what it retrieved and vice versa. The R3 analog, added here: after the frozen backbone produces `r_i` for a sample of training instances, run retrieval and add
```
L_commit = Σ_a ( ||sg[e_{j*}] − q_i^(a)||² + β ||sg[q_i^(a)] − e_{j*}||² )
```
for the top-1 retrieved `e_{j*}` per head (`sg[·]` = stop-gradient, `β` per FACE's eq. 9). Without this, `W_1..W_n` stay at their random orthogonal init forever — a fixed random projection is a legitimate retrieval baseline, so that's not necessarily wrong, but it should be a deliberate choice made in this phase, not an unnoticed gap discovered after Phase 3 shows retrieval behavior never changes.

**Phase 3 — Joint fine-tune, backbone unfrozen, prototypes concatenated for the first time.** Unfreeze at least the decoder (it has to learn to use `k` input positions it has never seen) and optimize
```
L = L_E1 + μ · L_recons + ν · L_commit
```
jointly, keeping `L_recons` (and `L_commit`, if adopted) active throughout rather than dropping them after Phase 2 — FACE's own joint objective (eq. 17) keeps `L_map` alongside `L_R` and `L_align` for exactly this reason: without it, joint gradient can still drift the mapping back toward collapse once it starts flowing, undoing Phase 2's pretraining. Whether `SelfAttentionLayer` itself (the part producing `r`) should stay frozen or unfreeze alongside the decoder in this phase is the same question Run A/B is already resolving for the μ/z line (main doc) — unfreezing it here reopens the shared-weight-leakage risk flagged above, so it's worth waiting on that outcome rather than re-litigating it independently here.

## Predicted E1 signature, if it works

`mask` should hurt (the concatenated prototype tokens carry content unavailable from `x` alone). `mean`/`shuffle` should hurt at least as much as in Attempt 8 (a stranger's `r` retrieves a different, wrong set of prototypes — compounding the mismatch rather than just losing personalization). Raw performance should not need to drop because this is an information addition, not an access restriction. This rationale is distinct from Attempt 9, which instead protects an already-trained base model while learning a zero-initialized correction.

## Open implementation decisions (not yet resolved)

- `k_a` (prototypes retrieved per head) and `λ` (MMR relevance/diversity trade-off) — no prior to set these from; would need a small sweep. `k_a` likely needs to be smaller than the single-head design's `k` given the same 514-candidate pool is now split `n` ways.
- `n` (number of aspect heads) — FACE's own sweep (their Fig. 4, descriptor number) found gains from 1 up to ~8 heads on their setting, flattening by 16 with duplication past that. That setting is a much larger bank/vocabulary than ours, so their numbers don't transfer directly; with only 514 prototypes here, `n` in the 2–4 range is a more plausible starting point, but this needs its own sweep rather than a borrowed default. Start at `n=1` (equivalent to the original single-head design) as the baseline to beat before scaling up.
- Minimum `quality_score` / non-empty-`anchor` filter threshold before a prototype is eligible for retrieval at all.
- Loss weights `μ` (reconstruction) and `ν` (commitment, if adopted) — no prior to set these from; see **Training curriculum** below for when each loss is active and why. Whether the 3-phase staging is actually necessary for R3 (vs. training everything jointly from the start being stable enough) is itself unresolved — worth checking training curves for instability before assuming staging is needed.
- Whether `proj_up` should be a full MLP or a more constrained map (e.g. low-rank affine + LayerNorm) — an unconstrained MLP has more capacity to find a degenerate CE-minimizing shortcut than a smaller one does; the reconstruction loss above is the primary mitigation, but a deliberately narrower `proj_up` may be a cheap second line of defense worth trying if collapse still shows up empirically.
- Concatenation vs. weighted-sum fusion (§ Mechanism, step 4) — concatenation is the current default per the centroid-collapse concern; weighted-sum remains documented as a fallback if `k` extra decoder positions prove too costly.
- Whether the `k` prototype positions need a type/segment embedding distinguishing them from `x`/`r` positions, and whether they need positional encoding at all given MMR's selection order has no inherent decoder-relevant meaning (a set, not a sequence) — likely no positional encoding, but unverified.
- Total `k = Σ_a k_a` grows with `n`; decoder input length and attention cost grow with it too — worth tracking as a joint constraint on `(n, k_a)` rather than tuning them independently.
- Whether heads need FACE's transformer-based inter-head mixing (their §3.2.2, `Transformer_e` over the `n` projected vectors before quantization) or whether independent per-head MMR + concat is sufficient. Descoped for a first version — R3's heads feed a retrieval step, not a joint quantization codebook, so the coupling FACE's transformer provides may matter less here; add it only if per-head retrievals turn out to be redundant with each other in practice.
- Whether to retrieve once (this doc's design) or iteratively refine over multiple hops (RALR's fuller sketch: `h_0 → [query_gen → retrieve → gated fuse] × T`, adaptive halting). Descoped for a first version — the single-hop design above tests the core claim (verified cross-user prototypes as an information advantage) without the added complexity of a multi-hop training loop and a halting head; multi-hop is the natural next step if single-hop shows a real, non-costly effect.
