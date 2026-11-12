# Experiment Tracker — latent-utilization-2026-10-28

| ID | What | Claim | Budget | Status | Result |
|---|---|---|---|---|---|
| E0 | Pool-headroom + asset check | gate | ≤1 GPU-h | NOT STARTED (awaiting user go-ahead) | — |
| E1 | C28 nine-plus-one splice | K5 | ~0 GPU | NOT STARTED (awaiting user go-ahead) | — |
| E2 | C6 beam-vs-catalog audit | K5 | ≤2 GPU-h | NOT STARTED (awaiting user go-ahead) | — |
| E3a | CF correction fit | K1 | hours, 1 GPU | BLOCKED on E0 | — |
| E3b | matched no-CF fit | K2 | hours, 1 GPU | BLOCKED on E0 | — |
| E4 | Locked evaluation + comparators | K1–K4 | ≤2 GPU-h | BLOCKED on E3 | — |
| E5 | C1 recency-split frontier (m∈{2,5,10}) | backup | 3 × ~1 GPU-day | CONDITIONAL (C10b-fail branch) | — |
| R1 | Toys + CDs replication of winner | confirmation | TBD | BLOCKED on E4/E5 | — |

Process: discuss each run with the user before implementation (project rule). Log every launch and result in `analysis/document/research_notes/Latent utilization/research_log.md` and update this tracker.
