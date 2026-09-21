# RECON_NOTES.md — Phase -1 deliverable

Working note for the diffusion-LatentR3 implementation task
(`document/CLAUDE_CODE_PROPOSAL_diffusion_latentr3.md` §1). One line per §2
assumption: confirmed / actually-different-because-X.

Branch: `dev/LatentR3_Diffusion` @ `d0ddc93`. Backbone config read from
`output_dir/CDs_and_Vinyl/latent/config.json`.

---

## §2 assumption table — verification

| # | Assumption | Verdict | Evidence |
|---|---|---|---|
| 1 | Backbone Qwen2.5-1.5B, hidden dim `d` | **Confirmed, with correction** — it is Qwen2.5-1.5B-**Instruct**, not base. `hidden_size = 1536`, 28 layers, `vocab_size = 151666` (151665 + `<\|Thought\|>`). | `scripts/latent_train.sh:19`; `output_dir/CDs_and_Vinyl/latent/config.json` |
| 2 | `LatentRATT` = one attention layer on the LLM's final hidden states, output in the same `d`-dim embedding space | **Confirmed.** `SelfAttentionLayer` (own Q/K/V + `VanillaRoPE`, `base=1000` — note: *not* Qwen's 1e6) reads `hidden_states[-1]` and writes a `d`-dim vector straight into the `<\|Thought\|>` token's input embedding. | `src/latent/layers.py:32-120`, `:145-158` |
| 3 | Default latent token count N=1 | **Confirmed but hardcoded in two places, one of them architectural** — see FLAG 1. | `src/latent/reasoning_dataset.py:122`; `src/latent/layers.py:99-119` |
| 4 | Two-pass inference, pass 1 **autoregressively** builds `r = [r_1..r_N]` via a LatentRATT loop | **MATERIALLY WRONG — see FLAG 2.** There is no autoregressive latent loop anywhere in the repo. `generate_embs` does one LLM forward + one attention readout, filling the thought position in a single shot. | `src/latent/layers.py:145-158` |
| 5 | Stage 1 (SFT) trains full model (LLM + LatentRATT), plain next-token CE on the item title only, no direct supervision of `r` | **Confirmed.** Standard HF `Trainer`, `loss_type="ce"`; labels are `-100` over the whole prompt and only the target title + EOS is supervised. | `src/latent/latent_attention_train.py:71`, `:109-140`; `src/latent/reasoning_dataset.py:137-141` |
| 6 | Stage 2 (RL) freezes base LLM, trains only LatentRATT | **Confirmed.** `model.model` and `model.lm_head` both set `requires_grad=False`; only `model.attention` stays trainable. | `src/grpo_attention_tuning/train_noise_grpo.py:73-76` |
| 7 | RL exploration = Gaussian perturbation of the latent, `r^1 = r` unperturbed | **Confirmed, with a caveat** — perturbation scale is a hardcoded `1.5`, and only **batch row 0** is zeroed, not row 0 of every group. See FLAG 5. | `src/grpo_attention_tuning/grpo_trainer.py:129-133` |
| 8 | Reward = negative exponentiated per-token PPL of the ground-truth title, no generation needed | **Confirmed.** `-(-logps.sum(-1)/labels_length).exp()`, computed from a teacher-forced forward pass. | `src/grpo_attention_tuning/grpo_trainer.py:156` |
| 9 | Advantage = batch-relative baseline from each group's unperturbed sample | **Confirmed.** Group-0 rewards are broadcast, cross-GPU-swapped via `swap_adjacent_blocks`, then `advantages = rewards - xx.mean()`, normalized by `torch.norm` (not by group std — the standard GRPO z-score line is present but commented out at `:177`). | `src/grpo_attention_tuning/grpo_trainer.py:172-176` |
| 10 | GRPO needs exactly 2 GPUs, "related to the advantage function" | **Confirmed, and it is load-bearing.** `swap_adjacent_blocks(xx, num_generations)` reshapes to `(-1, 2)` and flips — this hardcodes a 2-process layout. | `src/grpo_attention_tuning/grpo_trainer.py:17-26`, `:174` |

---

## Flags — material mismatches (proposal §1.4 says surface, don't paper over)

### FLAG 1 — N=1 is architectural, not a config knob
The proposal's Phase gate assumes `Lb` can be swept over {1,2,4,6,8,10}. In the
current code N=1 is baked into the *attention layer's shape logic*, not just data:

- `reasoning_dataset.py:122` appends exactly one literal `<|Thought|>` token.
- `SelfAttentionLayer.forward` selects **one query per batch row**
  (`Q_selected = Q[torch.arange(batch_size), indices]`, `layers.py:108`) and
  raises if `attention_mask.size(0) != thought_id_idx.size(0)`
  (`layers.py:49-50`). With 2+ thought tokens per sequence, `where_thought_ids`
  has `2*batch_size` rows and this mismatches.
- The `K` parameter threaded through `train()` → `LatentRDataset(K=...)` **looks**
  like the latent-count knob but is dead code: `self.K` is assigned
  (`reasoning_dataset.py:51`) and never read. `latent_train.sh` never sets it.

Consequence: supporting `Lb > 1` is new work in the dataset *and* the attention
layer, not a flag flip. This matches the proposal's own named trigger
("N is hardcoded somewhere non-obvious").

### FLAG 2 — there is no autoregressive latent loop to replace
The proposal's stated purpose (§ "Scope, in one sentence") is to fix
*error-accumulated reasoning* by replacing "LatentRATT's single autoregressive
latent token" with joint bidirectional refinement. But `generate_embs`
(`layers.py:145-158`) is a **single-shot, non-autoregressive pooling readout**:
one LLM forward pass, then one attention call whose query is the hidden state at
position `thought_idx - 1` attending over all positions before it.

At N=1 "AR loop" and "single-shot readout" are indistinguishable, so this costs
nothing today — but the *premise* of the comparison does not hold. There is no
sequential latent generation in this repo, hence no error accumulation to fix,
and no existing AR baseline for the `Lb` sweep to be compared against. If `Lb>1`
were added naively to the current layer, the natural extension is still
one-shot-parallel (all block positions computed from the same pass-1 hidden
states), which would very likely produce the near-identical vectors the
proposal's own Phase-1b acceptance check warns about.

**This is the load-bearing flag: the proposal's falsifiable signature (§5 Phase
gate — "diffusion curve stays flat where LatentR3's degrades") requires an AR
`Lb`-scaling baseline that does not currently exist and would itself have to be
built.**

### FLAG 3 — a full diffusion attempt already exists and reached a negative result
Branch `DiffuReason` (`f5b1eb2`, 2 commits ahead) already contains
`src/latent/diffusion.py` (199 lines), +240 lines in `layers.py`, three scripts,
and `reports/diffureason_stage1_tuning.md` (976 lines) documenting **13
experiments**, with outputs in `output_dir/CDs_and_Vinyl_exp*/`.

Its headline full-scale result on CDs (6,157-row test, SE on HR@10 ≈ 0.004):

| Run | Config | NDCG@10 | HR@10 |
|---|---|---|---|
| Exp 1 — no diffusion (baseline) | `latent_R=1, diff_T=0` | 0.07376 | 0.10248 |
| Exp 10 — diffusion, tuned | `latent_R=3, diff_T=4, alpha=0` | 0.07223 | 0.10119 |
| Exp 12 — diffusion, `proj_mse` | `alpha=1.0` | 0.06898 | 0.09696 |
| Exp 11 — diffusion, paper-exact | `alpha=1.0, mse` | 0.00131 | 0.00325 |

Report's own verdict: *"DiffuReason does not help, but it also does not hurt,
once both arms are trained on the same, adequate data budget. It washes out."*

That design is **not** this proposal's design (it is DDPM-style anchoring with an
`R=3` think-loop, not flow matching over a bidirectionally-refined block), so it
is not a direct refutation. But it is the same hypothesis class on the same task
and backbone, and it establishes: (a) the no-diffusion baseline number to beat,
(b) that small-scale wins on this task did **not** survive scaling — Exp 8 looked
like +16% at 5k rows and vanished at full scale, and (c) reusable architecture
lessons (residual denoiser, loss-scale normalization, `eps_init_scale`).

Item (b) directly threatens the proposal's Phase-gate plan, which runs the
`Lb` sweep at CDs scale and treats the result as go/no-go.

### FLAG 4 — dataset is not what §0 assumes
§0 says "Amazon 2018 categories, 8:1:1 chronological split — unchanged". Actual:

- `scripts/latent_train.sh` and `attention_grpo.sh` both point at
  `data/CDs_and_Vinyl_3_2022-10-2023-10_randNeg` — a **3-core, 2022-10..2023-10,
  random-negative** set (66,123 / 4,776 / 6,513 rows; 14,897 items). That is
  Amazon **2023**, not 2018, and the split is not 8:1:1.
- The Amazon-2018 5-core set the prior report used still exists at
  `data/CDs_and_Vinyl` (49,251 / 6,156 / 6,157; 5,841 items).
- `scripts/grpo_eval.sh` points at a third path (`data/${category}/...*11.csv`)
  and a checkpoint dir (`nomul_attn_lr_5e-4`) that does not exist in
  `output_dir/` — this script is stale as shipped.
- Only `CDs_and_Vinyl` exists locally. Phase 4's Toys / Games / Instruments would
  need to be downloaded and processed first.

So "Phase 0: reproduce the baseline" is ambiguous as written — the two datasets
give different numbers, and the paper-comparable one (§5 Phase 0 acceptance:
R@5≈0.080, R@10≈0.098, N@5≈0.065, N@10≈0.070) is the **2018** set, which the
current scripts do *not* use. Prior Exp 1 on the 2018 set (HR@5 0.08299,
HR@10 0.10248, N@5 0.06745, N@10 0.07376) matches those targets well.

### FLAG 5 — minor, but relevant to Phase 2 reuse
- Perturbation scale `1.5` is hardcoded (`grpo_trainer.py:131`); `noise[0,:]=0`
  zeroes only global batch row 0 while the advantage code treats element 0 of
  *every* group as the unperturbed baseline (`:173`). These agree only when
  `micro_batch_size == num_generations` (both are 8 in `attention_grpo.sh`) —
  an undocumented coupling any new trainer must preserve.
- Eval is constrained beam search with `num_beams=10`
  (`latent_attention_eval.py:69`), so HR@10 and HR@20 are always identical.
  Metrics come from `src/utils/calc.py`.

---

## Bottom line

§2 rows 1, 2, 5, 6, 8, 9, 10 confirmed. Row 3 confirmed but hardcoded
architecturally (FLAG 1). Row 4 is materially wrong (FLAG 2). Row 7 confirmed
with caveats (FLAG 5). Plus two scope-level surprises the proposal did not
anticipate: substantial prior negative results (FLAG 3) and a different dataset
than assumed (FLAG 4).

Per proposal §1.4, FLAGs 1, 2 and 4 are stop-and-surface conditions rather than
things to silently adapt around.
