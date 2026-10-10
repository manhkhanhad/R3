# Training Retrievers from Generator Feedback, and Case/Latent Memories (with Recommendation Applications)

Verification legend: **[V]** = checked this session against the arXiv/ar5iv page or venue page linked. **[M]** = stated from prior knowledge of the paper, with the arXiv link given, but NOT re-fetched this session. The report writer should treat [M] numbers as needing a spot check before quoting exact figures. Formulas marked [M] follow the standard published forms.

---

## Q1. How do REALM, RAG, FiD-distillation, EMDR2, Atlas, RePlug, perplexity/attention distillation, LOOP and end-to-end marginal likelihood learn the retriever from the generator? Exact objectives and failure modes

### Takeaway
Two families exist. (1) **Marginal-likelihood** methods (REALM, RAG, EMDR2) treat the document as a latent variable and maximize log Σ_{z∈top-k} p(z|x) p(y|x,z). (2) **Distillation** methods (FiD attention distillation, Atlas PDist/LOOP, RePlug LSR) build a *target distribution over the retrieved candidates* from the generator's own signal (attention, likelihood of y given d, leave-one-out loss) and minimize KL(target ‖ retriever softmax). Both families only score documents the current retriever already put in its top-k. In Atlas no loss clearly wins (the gaps are about 2–5 points), and every one of them beats a fixed retriever. Documented failure modes: retriever collapse and the generator then ignoring retrieval (RAG), saturation at low recall on one-to-many targets (Hindsight), stale indexes, and generators that do not use what they retrieve (Krishna et al.).

### Cited Findings

**REALM** (Guu et al., ICML 2020, arXiv 2002.08909) [M] — https://arxiv.org/abs/2002.08909
- Objective: p(y|x) = Σ_{z∈Z} p(y|z,x) p(z|x), with p(z|x) = exp(f(x,z)) / Σ_{z'} exp(f(x,z')), f(x,z) = Embed_input(x)ᵀ Embed_doc(z). The sum is approximated over the top-k by MIPS. Gradient intuition from the paper: the retriever score of z goes up iff p(y|z,x) > p(y|x), i.e. a document is rewarded when it helps *more than average*.
- Stale index: the document embeddings and the MIPS index are refreshed **asynchronously** every few hundred steps. The cost of a stale index is accepted.
- Heuristics added to avoid degenerate solutions: salient-span masking, a "null document" in the top-k (for when no retrieval is needed), blocking trivial retrieval of the exact source sentence, and ICT (inverse cloze) warm start of the retriever. The paper presents the warm start as needed because a random retriever gives no useful gradient (cold start).

**RAG** (Lewis et al., NeurIPS 2020, arXiv 2005.11401) [V] — https://ar5iv.labs.arxiv.org/html/2005.11401
- RAG-Sequence: p(y|x) ≈ Σ_{z∈top-k} p_η(z|x) Π_i p_θ(y_i|x,z,y_<i). RAG-Token: p(y|x) ≈ Π_i Σ_{z∈top-k} p_η(z|x) p_θ(y_i|x,z,y_<i). Trained by minimizing −log p(y|x) end-to-end. There is no retrieval supervision.
- Stale index workaround: "keep the document encoder (and index) fixed, only fine-tuning the query encoder". Only the query side learns.
- Gain from learning vs freezing the retriever (dev, Table 6): NQ EM 37.8→43.5 (Token) and 41.2→44.0 (Sequence); WebQuestions 37.1→46.5 (Token); Jeopardy Bleu-1 11.8→15.3 (Sequence). BM25 beat the learned dense retriever on FEVER.
- **Retrieval collapse** (Appendix): on some tasks (e.g. story generation) "the retrieval component would 'collapse' and learn to retrieve the same documents regardless of the input". The generator then learned to ignore the documents, and RAG performed like plain BART.

**FiD attention distillation** — "Distilling Knowledge from Reader to Retriever" (Izacard & Grave, ICLR 2021, arXiv 2012.04584) [M] — https://arxiv.org/abs/2012.04584
- The target is a softmax over passages of the reader's aggregated cross-attention scores (averaged over heads, layers and tokens). The retriever is trained by KL(p_attn ‖ p_retr), with p_retr = softmax(s(q,d)) over the retrieved set. Training is iterative: train the reader, distill into the retriever, re-index, retrieve again, repeat. No gold passage labels are needed.
- Atlas re-implements this as ADist, scoring each token by α_n‖v_n‖₂ (attention weight × value norm) [V, see the Atlas entry].

**EMDR2** (Sachan et al., NeurIPS 2021, arXiv 2106.05346) [M for the paper; V for Atlas's restatement] — https://arxiv.org/abs/2106.05346
- Expectation-maximization view with the retrieved docs as latent variables. The retriever term in Atlas's form is log[Σ_k SG(p_LM(a|q,d_k)) · p_retr(d_k|q)], with a stop-gradient on the LM, so the retriever moves mass toward the documents under which the reader assigns high likelihood to the answer. The reader is trained jointly on log p(a|q, top-K) (FiD over all K).

**Atlas** (Izacard et al., JMLR 2023, arXiv 2208.03299) [V] — https://ar5iv.labs.arxiv.org/html/2208.03299
- Shared retriever distribution: p_retr(d|q) = exp(s(d,q)/θ) / Σ_{k=1..K} exp(s(d_k,q)/θ), normalized over the **top-K only**.
- ADist: KL(p_attn ‖ p_retr), with SG on p_attn.
- EMDR2: log[Σ_k p_LM(a|q,d_k) p_retr(d_k|q)], with SG on p_LM.
- PDist (perplexity distillation): target p_k = softmax_k(log p_LM(a|d_k,q)), with KL(target ‖ p_retr). This is identical in spirit to RePlug's LSR.
- LOOP (leave-one-out): target p_loop(d_k) ∝ exp(−log p_LM(a | D_K∖{d_k}, q)), so a document scores high if *removing* it hurts the answer most. It is more expensive because each target needs K forward passes over K−1 docs.
- Table 1 (64-shot NQ / 64-shot avg): fixed retriever 39.9/42.2; ADist 40.9/43.0; EMDR2 43.3/43.3; PDist 45.0/45.7; LOOP 41.8/43.7. At 1024 shots LOOP has the best NQ (47.1). The authors found "no systematic winner" and chose PDist as more stable than EMDR2/ADist and cheaper than LOOP.
- Index staleness study: a full re-index every 1,000 steps costs about 30% overhead (37M docs). Re-ranking a stale top-L (L=10K) with the fresh encoder costs about 10%, but misses true top-K docs that are absent from the stale top-L. **Query-side-only fine-tuning** (frozen doc encoder) has no overhead and matched or beat full updates in few-shot (45.7 vs 44.1 avg at 64-shot), but was worse with more data (51.0 vs 51.7 at 1024-shot).
- Headline: Atlas-11B gets 42.4% NQ with 64 examples, about 3 points above PaLM-540B with about 50× fewer parameters.

**RePlug / LSR** (Shi et al., arXiv 2301.12652; published at NAACL 2024 [M for venue]) [V for abstract] — https://arxiv.org/abs/2301.12652
- The LM is a **black box**. Each retrieved doc is prepended separately and the output probabilities are ensembled. LSR ("LM-supervised retrieval") [M for exact form]: P_R(d|x) = softmax_{d∈D'}(s(d,x)/γ); Q_LM(d|x,y) = softmax_{d∈D'}(log P_LM(y|d,x)/β); loss = KL(P_R ‖ Q_LM) averaged over training contexts. The doc encoder is updated and the index is recomputed every T steps.
- Gains [V]: +6.3% on GPT-3 175B language modelling and +5.1% on Codex 5-shot MMLU.

**"Shall we pretrain autoregressive LMs with retrieval?"** (Wang et al., EMNLP 2023, arXiv 2304.06762) [M] — https://arxiv.org/abs/2304.06762
- Note: this is a **RETRO reproduction/scaling study** (RETRO vs GPT on text generation, factuality, toxicity, downstream tasks). It is *not* the source of perplexity distillation, which comes from Atlas (PDist) and RePlug (LSR). The brief conflated the two. The paper uses RETRO's frozen retriever and does not learn the retriever from the LM.

**Other LM-supervised retrievers (text)** [M]
- ART, "Questions Are All You Need to Train a Dense Passage Retriever" (Sachan et al., TACL 2023, arXiv 2206.10658): the target is the PLM likelihood of *reconstructing the question* from each passage, trained with KL to the retriever. It needs no labeled pairs. — https://arxiv.org/abs/2206.10658
- EPR (Rubin et al., NAACL 2022, arXiv 2112.08633): labels candidate in-context examples positive/negative by the LM's score p(y|e,x) and trains a contrastive dual encoder. — https://arxiv.org/abs/2112.08633
- LLM-R (Wang, Yang, Wei, EACL 2024, arXiv 2307.07164): trains a reward model from LLM feedback on candidate demos, distills it into a dense retriever, and iterates. — https://arxiv.org/abs/2307.07164
- UDR (Li et al., ACL 2023, arXiv 2305.04320) [V via search]: unified listwise ranking from LM feedback plus iterative candidate mining. It outperformed baselines on 30+ tasks and transferred across LMs from 1.3B to 175B. — https://arxiv.org/abs/2305.04320
- Dr.ICL (Luo et al., arXiv 2305.14128) [V via search]: mines demos that lead to the most accurate predictions and trains a demonstration retriever on them. It shows off-the-shelf BM25/GTR retrieve "query-specific knowledge" rather than useful exemplars. — https://arxiv.org/pdf/2305.14128
- Stochastic RAG (Zamani & Bendersky, SIGIR 2024, arXiv 2405.02816) [M]: replaces top-k marginalization with differentiable sampling without replacement (Gumbel-top-k) to optimize end-to-end. — https://arxiv.org/abs/2405.02816

**Generator ignoring retrieval** [M]
- Krishna, Roy & Iyyer, "Hurdles to Progress in Long-form QA" (NAACL 2021, arXiv 2103.06332): on ELI5, conditioning the RAG-style generator on *random* retrieved documents barely hurt quality, i.e. the generator was not grounded in its retrieval. — https://arxiv.org/abs/2103.06332

### Inferences
- In every method above the training signal ranks **only candidates the current retriever already surfaced** (Atlas normalizes p_retr over the top-K). If the right evidence is never in the candidate set, these losses cannot teach the retriever to find it. This is a likely mechanism for the "23% on misses" ceiling in our setting. With a **small fixed corpus** (CF clusters), the remedy is cheap: score *all* clusters (or a large random superset) with the decoder rather than only the top-k.
- Our corpus is fixed and small, so Atlas/RAG's stale-index problem largely disappears. Full re-scoring every step is feasible.
- PDist/LSR with our decoder is directly applicable: the target is q(c) ∝ exp(log p_dec(target item | x, evidence c)/β) over clusters c, trained with KL(q ‖ p_retr). It differs from "train on the target cluster label" because the label says which cluster *contains* the target, whereas PDist says which evidence *makes the decoder succeed*. The 63% decoder-conversion figure means these two can differ.
- LOOP is the only objective that captures complementarity among several pieces of evidence. It is relevant only if multiple evidence slots are fed at once.
- The collapse risk (RAG Appendix; generator learns to ignore retrieval) applies if the decoder is trained jointly. Freezing the decoder while training the retriever (as RePlug/Atlas-EMDR2 SG does) avoids the feedback loop.

### Gaps
- I did not re-fetch the exact RePlug LSR equation, the β/γ values, or the update period T. The REALM, FiD-distill and EMDR2 equations are given from memory and match Atlas's restatement.
- No found source quantifies how often the gold evidence is absent from the top-k candidates during LM-supervised training (the "candidate recall ceiling"), except Hindsight's Success@100 saturation (Q4).

---

## Q2. Recommendation papers that train a retriever/selector (items, users, behaviors, demos, reasoning) from recommender/LLM feedback

### Takeaway
Industrial CTR work has trained behavior retrievers from the downstream predictor for years (UBR4CTR, ETA, SIM). In LLM-based recommendation, learned demonstration/history retrievers trained on LLM feedback are recent and sparse. One rec ICL study found a trained contrastive demo retriever was *less robust* than LLM-embedding similarity. Most "retrieval-augmented LLM rec" papers (ReLLa, RecMind) use fixed semantic retrieval or agentic tool calls rather than a retriever learned from the LLM's loss.

### Cited Findings
- **UBR4CTR / UBR** (Qin et al., SIGIR 2020; journal version TOIS 2023, DOI 10.1145/3579354; arXiv 2005.14171) [V via search]: a learnable self-attentive query generator selects features to form a search query over the user's full behavior archive. The retrieved behaviors feed an attention CTR model. Training **alternates** an epoch of the predictor and an epoch of the retriever, with queries regenerated in between, so the retriever is optimized from the CTR model's signal. [M: the retriever is trained with a REINFORCE-style policy gradient using the predictor's likelihood as reward. Verify.] — https://arxiv.org/abs/2005.14171
- **ETA** (Chen et al., arXiv 2108.04468) [V via search]: makes long-sequence behavior retrieval end-to-end inside the CTR model using SimHash/LSH over embeddings shared with the main model. The retrieval keys are **learned latents** updated by the CTR loss. — https://arxiv.org/pdf/2108.04468
- **SIM** (Pi et al., CIKM 2020, arXiv 2006.05639) [M]: two-stage "General Search Unit" (hard category-based search or soft embedding MIPS) plus "Exact Search Unit". The soft-search embeddings are trained with an auxiliary CTR task. — https://arxiv.org/abs/2006.05639
- **RIM, Retrieval & Interaction Machine** (Qin et al., KDD 2021, arXiv 2108.05252) [V via search]: retrieves **other rows/samples** (cross-sample neighbours, with their labels) from the training table via a search engine, then models interactions between the target row and the retrieved rows. Evaluated on 11 datasets (CTR, top-n, rating), it beat the state of the art. The retrieval is BM25-style and not learned. Conceptually closest to "other samples as evidence". — https://arxiv.org/abs/2108.05252
- **RaSeRec** (Zhao et al., arXiv 2412.18378) [V via search; details partly from a secondary review]: memory bank of **latent user representations paired with their next-item targets**, retrieved by Faiss kNN. Two stages: (i) collaborative pre-training that "learns to recommend and retrieve" (contrastive/InfoNCE with similar-preference sequences as positives); (ii) retrieval-augmented fine-tuning with dual-channel cross-attention (query = user rep, keys = retrieved reps, values = their targets, plus the reversed channel). Model-agnostic, with SASRec backbone. — https://arxiv.org/abs/2412.18378
- **ICL demonstrations for sequential rec** — "The Whole is Better than the Sum" (Wang et al., arXiv 2403.10135) [V via search]: compared five demonstration selectors for LLM sequential rec, including a **trained contrastive retriever**. LLM-embedding similarity was *more robust* than the trained retriever and needed no training. Retrieval beat random selection. — https://arxiv.org/pdf/2403.10135
- **AdaptRec** (2025) [V only via summary page; unverified details]: has the LLM itself evaluate and select similar-user histories as demonstrations. Argues that Euclidean CF similarity retrieves demos that don't help LLM reasoning, and reports gains of up to 23% in few-shot. — https://papers.lunadong.com/paper/10518
- **RAR, retrieval-augmented conversational rec with RL** (arXiv 2604.04457, 2026) [V via search snippet only]: the retriever proposes candidates, the LLM refines them, and the retriever is **post-trained by RL from LLM feedback with the LLM frozen**. — https://arxiv.org/html/2604.04457
- **ReLLa** (Lin et al., WWW 2024, arXiv 2308.11131) [M]: "semantic user behavior retrieval". It selects the top-K history items most semantically similar to the target via frozen LLM embeddings to fix LLMs' long-context weakness in rec. The retriever is **not** learned from feedback. — https://arxiv.org/abs/2308.11131
- **RecMind** (Wang et al., NAACL Findings 2024, arXiv 2308.14296) [M]: an LLM agent with "self-inspiring" planning and tool calls (database/search). Its retrieval is agentic and prompt-driven, with no trained retriever. — https://arxiv.org/abs/2308.14296
- Adjacent (not rec): GenICL (arXiv 2505.19966) frames demo selection as preference learning from LLM feedback, arguing metric-learning retrievers optimize a surrogate rather than ICL performance [V via search] — https://arxiv.org/html/2505.19966v1

### Inferences
- The rec literature closest to our setup is UBR4CTR/ETA (the retriever is trained from the downstream predictor's loss, over the user's own items) and RIM/RaSeRec (evidence = *other samples* and their labels/latents). None of these papers report a "retriever hit rate on base-model misses", so the 23% number has no direct literature benchmark.
- 2403.10135's negative result (a trained contrastive demo retriever is less robust than embedding similarity) mirrors our finding that label-trained retrievers underperform. It suggests that label/contrastive supervision is the wrong target. That points toward PDist/LSR-style supervision from the decoder's likelihood under each evidence.

### Gaps
- I found no recommendation paper that explicitly applies Atlas/RePlug-style KL-to-LM-posterior training to select CF clusters or latent evidence for an LLM recommender. The search returned none, and this looks like an open niche (not confirmed exhaustively).
- The exact UBR4CTR retriever loss (REINFORCE vs differentiable) was not confirmed from the full text.
- The venue for 2403.10135 and AdaptRec was not verified.

---

## Q3. kNN-LM, RETRO, memorizing transformers, latent/hidden-state memories, CBR, memory networks for rec, session-kNN. Which retrieve latents rather than text?

### Takeaway
The latent-retrieving memories are kNN-LM, Memorizing Transformers, TRIME, ETA, RaSeRec and the memory networks for rec. They retrieve hidden states keyed by hidden states, and none of them *learn which* entry to retrieve beyond the similarity in the model's own representation space. RETRO, REALM, RAG and CBR-KBQA retrieve text. In session-based rec, plain nearest-neighbour heuristics (SKNN/V-SKNN) are repeatedly found competitive with or better than neural models. Simple neighbour retrieval is a strong baseline, not a strawman.

### Cited Findings
- **kNN-LM** (Khandelwal et al., ICLR 2020, arXiv 1911.00172) [M]: datastore of (f(context) hidden state → next token). p(y|x) = λ p_kNN(y|x) + (1−λ) p_LM(y|x), with p_kNN(y|x) ∝ Σ_{(k_i,v_i)∈N} 1[y=v_i] exp(−d(k_i, f(x))). **Latent keys, no training of the retriever**: the keys are the frozen LM's representations. It reached state-of-the-art WikiText-103 perplexity without extra training, with large gains on rare/long-tail patterns. — https://arxiv.org/abs/1911.00172
- **"Why do Nearest Neighbor LMs Work?"** (Xu et al., ICML 2023, arXiv 2301.02828) [M]: attributes the gains partly to using a different input representation (the attention-layer output) and to the kNN distribution's softmax temperature and interpolation. — https://arxiv.org/abs/2301.02828
- **TRIME** (Zhong et al., EMNLP 2022, arXiv 2205.12674) [M]: *trains* the LM with in-batch memories (latent keys) via a contrastive objective, so the representation is shaped for retrieval at training time. This is the latent-memory analogue of learning the retriever. — https://arxiv.org/abs/2205.12674
- **Memorizing Transformers** (Wu et al., ICLR 2022, arXiv 2203.08913) [M]: an external non-differentiable cache of past (key, value) attention pairs. One layer does approximate kNN lookup over up to about 262K tokens and gates it with local attention. It retrieves **hidden states**, and gradients do not flow into the memory. — https://arxiv.org/abs/2203.08913
- **RETRO** (Borgeaud et al., ICML 2022, arXiv 2112.04426) [M]: retrieves **text chunks** with a **frozen BERT** retriever from a database of about 2T tokens via chunked cross-attention. It is comparable to GPT-3 on the Pile with 25× fewer parameters. The retriever is never trained from the LM.
- **CBR in NLP**: CBR-KBQA (Das et al., EMNLP 2021, arXiv 2104.08762) [M] retrieves similar *cases* (question + logical form) and revises them. It retrieves text/programs, and its retriever is trained with a supervised case-similarity objective. — https://arxiv.org/abs/2104.08762
- **Memory networks for rec** [M]: RUM (Chen et al., WSDM 2018) gives each user a key-value item/feature memory read by attention. MIMN (Pi et al., KDD 2019, arXiv 1905.09248) is a NTM-style user interest memory for long sequences. Both are **latent** memories addressed by attention over the model's own embeddings. — https://arxiv.org/abs/1905.09248
- **Session-kNN strength** (Ludewig, Mauro, Latifi, Jannach; arXiv 1803.09587; RecSys 2019 short paper; UMUAI 2021 journal) [V via search]: across 12 algorithms (6 neural), "simple heuristic methods based on nearest-neighbors schemes are preferable over conceptually and computationally more complex methods in most cases". On precision/recall, simple techniques "in most cases outperform recent neural approaches". — https://arxiv.org/abs/1803.09587 ; https://eldorado.tu-dortmund.de/items/caa7d2e3-7cf4-4cc1-91f3-fdf3014e6454/full
- **"Are we really making much progress?"** (Ferrari Dacrema, Cremonesi, Jannach, RecSys 2019, arXiv 1907.06902) [M]: of 18 neural top-n algorithms, 7 were reproducible with reasonable effort, and 6 of those were beaten by simple tuned neighbourhood/graph heuristics. — https://arxiv.org/abs/1907.06902
- RaSeRec (Q2) retrieves **latent user representations + target items**. ETA (Q2) retrieves behaviors via LSH over **learned embeddings**. RIM (Q2) retrieves raw rows.

### Inferences
- Our corpus (CF clusters / other samples' latents) is a latent memory in the kNN-LM/RaSeRec family. The literature's main lever there is the **key representation** (kNN-LM: which layer; TRIME: train the representation with retrieval in the loop), not a separate learned retriever. A cheap experiment: re-key the memory with the decoder's own hidden state at the thought slot (kNN-LM style) instead of label-trained embeddings.
- kNN-LM's interpolation (λ p_kNN + (1−λ) p_model) is a natural non-parametric baseline to put against "decoder converts evidence into a hit". RaSeRec's "retrieve neighbours' *targets* as values" is close to what CF-cluster evidence does.
- Given Ludewig/Dacrema, a session-kNN / item-kNN evidence selector should be reported as a baseline. It may beat the trained 23%-on-misses retriever.

### Gaps
- No found paper trains a latent-memory retriever with an Atlas/RePlug-style KL to the downstream model's per-entry likelihood. TRIME is the nearest, and it is contrastive at the LM level.
- Exact kNN-LM/RETRO perplexity numbers were not re-verified this session.

---

## Q4. Posterior vs prior retrieval (a retriever conditioned on the target during training). Does it find evidence the prior misses?

### Takeaway
Yes, in the clearest study (Hindsight, ICLR 2022). A target-conditioned guide retriever Q(z|x,y) trained with an ELBO lifts the *prior* retriever's Success@5 (58.8) above the marginal-likelihood retriever's Success@100 (55.8). The prior thereby learns to surface evidence that marginal-likelihood training never reached. The gains are large on one-to-many tasks (dialogue) and small on one-to-one tasks (MS MARCO NLGen, about 1–2 points). End-to-end top-1 quality barely moves.

### Cited Findings
- **Hindsight** (Paranjape, Khattab, Potts, Zaharia, Manning; ICLR 2022; arXiv 2110.07752) [V] — https://ar5iv.labs.arxiv.org/html/2110.07752 ; venue: https://iclr.cc/virtual/2022/poster/6736
  - Objective: maximize ELBO = E_{z∼Q(·|x,y)}[log P_θ(y|x,z)] − KL(Q(z|x,y) ‖ P_η(z|x)). P_η is the prior retriever (ColBERT on x), Q the guide/posterior retriever (ColBERT on [x; y]), P_θ the BART generator. The KL is reverse KL (mode-seeking).
  - Training: at each round, build a closed set of the top-r (r=100) under Q and P_η and sample k passages. Passages are sampled from an **α-mixture of P_η and Q**, with low α early and higher α later, so training gradually matches test-time (prior-only) retrieval.
  - Diagnosis of marginal-likelihood (RAG-style) failure: the generator's probability plays two conflicting roles (supervising label-relevance and grounding). On one-to-many tasks the generator is flat, so the supervision is weak. The retriever saturates at low recall, top-k at training time misses label-relevant passages, and the generator is less grounded.
  - Wizard of Wikipedia dev, MarginalizedLoss → ELBO: MRR 43.8→49.0; Success@5 49.9→58.8; Success@10 52.8→63.9 (the paper reports this as a 23% relative gain; the numbers give about 21%); Success@100 55.8 (marginalized, saturated) vs 69.3. The posterior Q itself reaches S@10 = 88.4 (it sees y). Grounding (top-1 Novel-F1) 17.46→20.78. End-to-end top-1 F1 is nearly unchanged (18.79→18.86), while max-over-top-5 F1 goes 26.52→27.56. KILT WoW F1 18.11→19.19.
  - The ELBO retriever retrieves more diverse passages. The marginalized one picks many similar "safe" passages, which is a mild form of collapse.
- **Related latent-variable / posterior retrieval** [M]: REALM/RAG/EMDR2 compute an *implicit* posterior p(z|x,y) ∝ p(z|x)p(y|x,z), but only over the prior's top-k. Hindsight's contribution is a posterior that **searches the whole corpus** with y in the query. Older knowledge-grounded dialogue work (e.g. PostKS, Lian et al., IJCAI 2019, arXiv 1902.04911) used a posterior knowledge selector conditioned on the response with a KL(posterior ‖ prior) loss over a small candidate set. — https://arxiv.org/abs/1902.04911

### Inferences
- This maps most directly onto our problem. The evidence corpus is small, so the posterior can be computed **exactly** rather than learned: Q(c|x,y) ∝ p_η(c|x) · p_dec(y|x,c), or simply ∝ p_dec(y|x,c), over *all* clusters. A target-conditioned teacher of this kind is precisely what PDist/LSR use, without the top-k restriction. Train the prior with KL(Q ‖ p_η) (forward KL, mass-covering) or Hindsight's reverse KL.
- Hindsight's "one-to-many" diagnosis applies. For a missed user, several clusters may each yield the hit (the decoder converts any correct evidence 63% of the time). The posterior over evidence is multi-modal, which favours mode-seeking/sampling objectives over target-label cross-entropy, which forces a single "correct" cluster.
- Hindsight's caveat: big retrieval-recall gains gave only small top-1 end-to-end gains. Recall@k of the prior should be expected to improve more than top-1 hit rate.
- "Decoder confidence is below chance as a selector" means the *decoder's confidence without seeing y* is a bad prior. The posterior signal p_dec(y | x, c) *with the target* is a different and much stronger signal. It is only available at training time, which is exactly the setting Hindsight designs for.

### Gaps
- I found no recommendation paper using a Hindsight-style posterior retriever over CF evidence.
- No found paper reports how much a posterior-distilled prior closes the gap to the posterior oracle (Q S@10 = 88.4 vs prior 63.9 in Hindsight is the only data point found).
