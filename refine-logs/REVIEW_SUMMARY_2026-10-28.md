# Review Summary — latent-utilization-2026-10-28

All reviews via Codex CLI (gpt-6-astra unless noted); full raw traces under `.aris/traces/`.

## Round A — Idea triage (Phase 4 of idea-creator), xhigh, session `01a182a8-f75d-7783-9c3a-1eac344806c8`
29 deduped clusters ranked. Pilot slots: C28, C6, C10b (protected high-upside). Next training bets: C1, C3. Eliminated on objective flaws: C23 (I(r;y|x)=0 for deterministic r=f(x)), C9 (teacher mixture at inference needs unavailable target-conditioned explanations). Key standing instruction: objections become named risks, never new modules.

## Round B — Novelty verdicts, xhigh, session `01a182bd-6f9e-7c31-874b-b4575f528a4a`
C10b: 6/10 PROCEED (closest Refiner 2610.36890, concurrent; delta = reference-centered bilinear correction on a frozen LLM title-generation baseline + beat-baseline-AND-beat-interventions requirement). C1: 5/10 PROCEED (closest PerSRec 2601.03479; delta = the audited dependence–accuracy frontier). Claim boundaries recorded in FINAL_PROPOSAL.

## Round C — Critical review, **ultra**, session `01a182c1-fe9d-7800-be4a-d18b304a69a3`
Bottom line: proceed with the capped two-run experiment; do NOT grant lead status on audit passage alone. Six material gaps (audit≠accuracy-goal; mean≡baseline is one test; substitution can pass; dependence≠new-information; underspecified frozen producer; statistics) — all folded into the proposal as gates G1–G5 + promotion gate K4. Delivered the agreed run order E0–E4 and the 8-row results-to-claims matrix. Venue: RecSys/SIGIR-credible with multi-dataset replication + attributing ablations; NeurIPS needs the broader cross-setting story.

## Verdict carried into planning
READY (conditional): thesis stable, gates explicit, no unresolved REVISE items. The one reviewer-named specification gap (frozen CF-infused producer) is resolved in FINAL_PROPOSAL §Method step 2.

## Round D — v2 plan check (lead change to C1+C11), xhigh, session `01a182e8-4cb7-7a01-bcb0-02bb429511e6`
Trace: `.aris/traces/research-review/2026-10-28_run02/review_v2.out`. Verdict REVISE with three blocking items, all applied verbatim to FINAL_PROPOSAL/EXPERIMENT_PLAN v2: (1) recent-only baselines need equal training budget or demonstrated convergence; (2) "within noise of 0.1226" replaced by a prespecified noninferiority test (paired, margin δ_NI = 0.005 HR@10); (3) C11's "structural cancellation" rationale was wrong — CFG gives h + wδ (history retained, δ amplified); rationale corrected to amplified-incentive with stop-gradient on the null branch, and P1 split into three gates (contrast magnitude / decoding effect = promotion gate / intervention behavior). Frontier claims restricted to tested splits; validation-selected m and w frozen before confirmation. C1 audit design otherwise judged sound.
