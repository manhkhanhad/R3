# Complementary auxiliary models and new information for hard cases in sequential recommendation

Scope note: about 17 search/fetch calls. Venue/arXiv IDs marked "verified" were confirmed on arXiv/ACM/NeurIPS pages in this session. Items marked "(unverified here)" come from background knowledge and need checking before citation.

## (a) Making an auxiliary model complementary to the base model's errors (boosting, residual, routing, DRO, ensembles, learning-to-defer)

### Takeaway
No published recommender work I found trains a second model only on the first model's misses and shows that this helps on those misses. Boosting for CF exists (AdaBoost-style reweighting, ACML 2013; BoostFM, IUI 2017), but it is evaluated on aggregate top-N metrics, not on the base model's failures. The most direct recent evidence is negative. RouteRec (2026) finds a large oracle headroom between recommenders but shows that a learned per-request selector does worse than the best single model. Selection fails, but item-level aggregation of candidate lists partly recovers the gap. Ensembles that use diversity (EMKD) and worst-case objectives (DRO/CVaR) give modest aggregate gains.

### Cited Findings
- **Classic boosting for CF.** Jiang et al., "Novel Boosting Frameworks to Improve the Performance of Collaborative Filtering" (ACML 2013, PMLR v29). It adapts AdaBoost and combines several homogeneous recommenders built on the same CF algorithm with different sample weights. — [PMLR](https://proceedings.mlr.press/v29/Jiang13.html); [mlanthology](https://mlanthology.org/acml/2013/jiang2013acml-novel)
- **BoostFM.** Yuan et al., "BoostFM: Boosted Factorization Machines for Top-N Feature-based Recommendation", appeared at **IUI 2017** (pp. 45–54), not SIGIR. It is an adaptive boosting framework over FM component recommenders, trained with pairwise and listwise L2R losses and with the data re-weighted each round. It reports beating SOTA top-N baselines on aggregate metrics. — [Glasgow eprints](https://eprints.gla.ac.uk/135914)
- **AdaFM** (Li et al., 2018, arXiv 1804.06027) uses boosting to choose adaptive FM ranks. It is not error-focused complementarity. — [arXiv 1804.06027](https://arxiv.org/pdf/1804.06027)
- **Residual refinement inside one model.** "Slow Thinking for Sequential Recommendation" (arXiv 2504.09627) trains a model to predict the residual between its current prediction and the target-item representation, then refines iteratively. This is one model refining itself, not a second model fit to the first one's misses. — [arXiv 2504.09627](https://arxiv.org/html/2504.09627v1)
- **Diversity ensembles.** EMKD, "Ensemble Modeling with Contrastive Knowledge Distillation for Sequential Recommendation" (SIGIR 2023, arXiv 2304.14668). It trains parallel sequence encoders and combines their output distributions, with intra- and cross-network contrastive learning plus logit-level KD. It reports significant gains over SOTA. Its premise is that diversity across networks drives accuracy. — [arXiv 2304.14668](https://arxiv.org/pdf/2304.14668). FLAME (arXiv 2604.04038) condenses ensemble diversity into a single network. — [alphaXiv](https://www.alphaxiv.org/abs/2604.04038)
- **Per-request routing between recommenders (RouteRec, arXiv 2607.09908, AgentSearch workshop @ SIGIR 2026).** On MovieLens-1M with 4 traditional recommenders (CF, sequential, content-based) plus 1 LLM reranker:
  - Hard request-level selection under a leakage-free 5-fold out-of-fold protocol gets HR@10 = 0.223, below BM25 at 0.254. Selective LLM escalation does not help.
  - The quality oracle reaches HR@10 = 0.584.
  - The union of candidate top-10 lists contains the held-out item for 50.8% of users, and 58.4% when all agents are pooled.
  - Gated all-agent item-level aggregation reaches HR@10 = 0.295, using LLM calls on 70.2% of requests.
  - The authors conclude that oracle headroom is not enough, because labeled switching cases are few and sparse (one leave-one-out target per user), and that item-level aggregation is the more promising direction.
  - Source: [arXiv 2607.09908](https://arxiv.org/abs/2607.09908)
- **LLM + CF hybrids, which fuse rather than route.**
  - A-LLMRec (arXiv 2404.11343) notes that LLM-based recommenders underperform simple CF in warm scenarios because they lack collaborative knowledge. It injects pretrained CF embeddings into the LLM. — [arXiv 2404.11343](https://arxiv.org/abs/2404.11343)
  - Laser (arXiv 2409.01605) uses an MoE "M-Former" whose router picks query experts per user from collaborative characteristics, over a frozen ID-based SR model. This is the closest thing to "gating by user type". — [arXiv 2409.01605](https://arxiv.org/pdf/2409.01605)
- **Multi-LLM routing.** MLTFR, "Multi-LLM Token Filtering and Routing for Sequential Recommendation" (arXiv 2604.18200, Apr 2026), routes across several LLMs' token embeddings to cover what a single LLM misses. Routing here happens at the embedding-source level, not as a CF/LLM decision. — [arXiv 2604.18200](https://arxiv.org/abs/2604.18200)
- **Learning-to-defer.** The classic framework is Madras et al. (2018). Recent LLM-cascade work trains a router that defers hard queries to stronger models (arXiv 2509.22984). I found no paper on learning-to-defer between an LLM recommender and a CF model. — [arXiv 2509.22984](https://arxiv.org/html/2509.22984v1)
- **Worst-case / hard-example objectives.**
  - Streaming-DRO (Wen et al., "Distributionally-robust Recommendations for Improving Worst-case User Experience", **WWW 2022**) optimizes worst-case subgroup performance. It reports beating vanilla DRO on both worst-case and overall metrics. — [NSF PAR](https://par.nsf.gov/servlets/purl/10348170)
  - "Improving Minimax Group Fairness in Sequential Recommendation" (Acharya et al., Amazon; arXiv 2501.18117) applies Group DRO, Streaming DRO and **CVaR DRO** to SASRec. CVaR needs no group labels and up-weights the highest-loss examples in each mini-batch, which makes it the closest analogue to "focus training on failures". The venue was not confirmed. — [arXiv 2501.18117](https://arxiv.org/html/2501.18117v1)
  - DRO over system exposure (arXiv 2312.07036, CIKM 2023 per its DOI 10.1145/3583780.3615492) targets exposure debiasing, not hard users. — [arXiv 2312.07036](https://arxiv.org/pdf/2312.07036)

### Inferences
- Boosting, CVaR and DRO all reweight the same history-derived training signal. If every history-only model fails on the same rows (as in the user's setup), reweighting towards those rows has nothing new to fit. The literature reports only aggregate or worst-group gains of modest size, and nothing on recovering a base model's misses.
- RouteRec is the most informative analogue. Oracle headroom exists only when the models really differ in their candidate sets. Even then, hard selection fails with one target per user. Item-level aggregation (candidate-union reranking) is what recovered some gain. The user's finding that every retriever hits the same rows is the stronger negative case: there the union adds almost nothing.
- Complementarity has to come from a different input (text, other users, other domains, LLM knowledge), not from a different loss on the same input.

### Gaps
- I found no paper that explicitly trains a recommender only on base-model misses and reports results on those misses. That absence is itself a finding.
- I did not verify a published per-user CF-vs-LLM gating study with per-group results. CoLLM (Zhang et al., arXiv 2310.19488, unverified here) reports warm/cold splits for an LLM+CF fusion, which is worth checking.
- I did not retrieve EMKD's numbers or a per-user-group breakdown.

## (b) New information sources for hard cases (text/metadata, semantic IDs, KG/LLM knowledge, cross-domain, user-user neighbors, intents)

### Takeaway
The strongest documented gains on hard rows come from two inputs: **item text and semantics** (cold and long-tail items) and **similar users retrieved via LLM embeddings of history** (long-tail users). LLM-ESR gives the clearest per-group numbers, on Amazon Beauty with SASRec: tail items H@10 0.087 → 0.226, tail users 0.427 → 0.558. Recformer reports cold-item gains on Amazon **Instruments**. Cross-domain methods need overlapped users. LLM-generated pseudo cross-domain items (LLMCDSR) are the proposed fix for non-overlapped users, but come with hallucination risk. Intent and multi-interest methods help sparsity on aggregate, but give no per-hard-user evidence.

### Cited Findings
- **LLM-ESR** (NeurIPS 2024, arXiv 2405.20646, verified).
  - Method: dual-view (LLM semantic + CF) item embeddings, plus retrieval-augmented self-distillation from the top-N similar users. Each user's history is turned into text, encoded by an LLM, and neighbours are found by cosine similarity; N = 10 is best.
  - Data: Yelp, Fashion and Beauty. Fashion has 4,722 items and **avg seq length 3.82**, similar to the user's 4.4. Beauty has avg length 7.57.
  - Groups: tail means not in the top 20% by popularity (items) or by sequence length (users).
  - Beauty + SASRec, H@10/N@10:
    - Overall: 0.4388/0.3030 → 0.5672/0.3713
    - Tail item: 0.0870/0.0649 → 0.2257/0.1108
    - Tail user: 0.4270/0.2941 → 0.5581/0.3643
    - All gains significant (p < 0.05).
  - Note: these are sampled-candidate metrics, which explains the high absolute values.
  - Sources: [arXiv HTML](https://arxiv.org/html/2405.20646); [NeurIPS 2024 poster](https://neurips.cc/virtual/2024/poster/93061)
- **Recformer** (KDD 2023, arXiv 2305.13731), on Amazon **Instruments**:
  - Full data, NDCG@10 / R@10:
    - SASRec 0.0634 / 0.0995
    - UniSRec 0.0785 / 0.1119
    - Recformer 0.0830 / 0.1052
  - Cold items (unseen in training), NDCG@10 in-set / cold:
    - SASRec 0.0669 / 0.0142
    - UniSRec 0.0715 / 0.0208
    - Recformer 0.0916 / 0.0315
  - Cold items, R@10 cold: SASRec 0.0309, UniSRec 0.0319, Recformer 0.0468.
  - Text models also lead in zero-shot and low-resource settings. The paper cites a 39.78% zero-shot NDCG@10 gain.
  - Source: [arXiv 2305.13731](https://arxiv.org/html/2305.13731)
- **UniSRec** (KDD 2022, arXiv 2206.05941, unverified here) is text-based universal SR. Its Instruments numbers above come from the Recformer paper.
- **TIGER / semantic IDs** (arXiv 2305.05065; NeurIPS 2023 per common citation, though the ar5iv page does not state the venue).
  - Data: Amazon Beauty, Sports and Toys (mean seq length about 8.3–8.9, median 6).
  - Recall@5: Beauty 0.0454 vs 0.0387 for S3-Rec (+17.3%); Toys 0.0521 vs 0.0443 (+12.5%).
  - NDCG@5 on Beauty: +29% over SASRec.
  - Cold start: 5% of test items were held out of training. TIGER with ε = 0.1 beats Semantic-KNN at all Recall@K. Exact values appear only in a figure.
  - Hierarchical codes share knowledge between semantically similar items.
  - Source: [ar5iv 2305.05065](https://ar5iv.labs.arxiv.org/html/2305.05065)
- **LC-Rec** (Zheng et al., ICDE 2024, arXiv 2311.09049, unverified here) also uses Amazon Instruments, Arts and Games with semantic IDs. I did not retrieve its numbers.
- **KAR** ("Towards Open-World Recommendation with Knowledge Augmentation from LLMs", RecSys 2024, arXiv 2306.10933, verified).
  - Method: the LLM generates user-preference reasoning (via factorization prompting) and item factual knowledge. A hybrid-expert adaptor turns this into augmentation vectors, which are precomputed.
  - Gains: about 1–1.5% AUC across 9 CTR models; +5.71% MAP@7 and +4.71% NDCG@7 on Amazon-Books reranking (PRM backbone); online A/B tests +7% (news) and +1.7% (music).
  - The follow-up REKI (arXiv 2408.10520) reports +1.99% for music, so the figures differ between the two papers.
  - These are CTR/rerank results with no hard-user breakdown found.
  - Sources: [arXiv 2306.10933](https://arxiv.org/pdf/2306.10933); [arXiv 2408.10520](https://arxiv.org/pdf/2408.10520)
- **LLM interest exploration** (Wang et al., Google, arXiv 2405.16363). An LLM generates *novel* interest clusters, and a transformer SR model is constrained to return items inside those clusters. This is a two-level "group then item" design. Live experiments report significant increases in novel-interest exploration and user enjoyment. The venue (RecSys 2024) appears only on third-party pages and was not confirmed. — [arXiv 2405.16363](https://arxiv.org/html/2405.16363v2)
- **Cross-domain.**
  - Tri-CDR (TOIS, arXiv 2304.05027) models source, target and mixed sequences. It is evaluated on 4 Amazon and 2 Douban domains, and the mixed sequence relies on overlapped users. — [arXiv 2304.05027](https://arxiv.org/pdf/2304.05027)
  - Overlapped users are a small fraction of all users, and non-overlapped users have no other-domain behaviour to borrow. — [arXiv 2410.13588](https://arxiv.org/abs/2410.13588v1)
  - **LLMCDSR** (TOIS 2025, DOI 10.1145/3715099) has an LLM generate pseudo cross-domain items from single-domain histories. A meta-learned Relevance-Aware Meta Recall Network keeps only high-quality pseudo items, to counter hallucination. It reports gains on two public datasets; I did not retrieve the numbers. — [PSU Pure](https://pure.psu.edu/en/publications/llmcdsr-enhancing-cross-domain-sequential-recommendation-with-lar/); [LLM-EDT arXiv 2511.19931](https://arxiv.org/pdf/2511.19931)
  - C2DSR (CIKM 2022, unverified here) was not retrieved.
- **Intent / latent interest.**
  - ICLRec (WWW 2022, arXiv 2202.02519) clusters sequence representations into latent intents and adds intent contrastive learning. It is evaluated on Amazon Sports, Beauty and Toys plus Yelp, and claims robustness to sparsity and noise.
  - A third-party summary reports 7.47–33.33% average HR/NDCG gains; this is not checked against the paper.
  - Sources: [ar5iv 2202.02519](https://ar5iv.arxiv.org/html/2202.02519); [summary](https://liner.com/review/intent-contrastive-learning-for-sequential-recommendation)
  - MIND/ComiRec were not retrieved.

### Inferences
- Of all the sources, only **text/semantic item content** and **LLM-embedded user neighbours** have documented per-group gains on rows where ID/history models are weak (cold items, tail items, short-history users).
  - LLM-ESR's tail-user gain on Beauty is large.
  - Its Fashion dataset (avg length 3.82, about 4.7K items) is the closest public analogue to Musical Instruments 5-core.
- Note the difference from the user's own neighbour retrievers, which are built from the user's *history*: LLM-ESR finds neighbours with **LLM text embeddings of the history**. This is a different similarity space, which may surface different users. Whether it fixes the *same* misses was not measured.
- Recformer's Instruments cold-item NDCG (0.0142 → 0.0315) shows that text closes part of the cold-item gap, but cold-item accuracy stays about 3x below in-set accuracy. Text adds information, but not enough to close the gap.
- The Google two-level design (LLM picks a novel cluster, CF picks the item within it) mirrors the user's finding that coarse groups are right on about 18% of misses but do not identify the item. In that work, the item is picked by a standard sequential model restricted to the cluster. No published offline accuracy on misses was found.
- Cross-domain sources depend on user overlap. With about 1% overlap in the user's local data, the only option the literature offers is LLM-generated pseudo-items (LLMCDSR), which is noisy.

### Gaps
- No source evaluates any of these information sources specifically on the *misses* of a strong base model (a conditional analysis). Per-group results (tail/cold) are the closest proxy.
- Not retrieved: numbers for UniSRec, LC-Rec, C2DSR, MIND/ComiRec, KG-based SR (e.g. KGAT/KSR), and LLMCDSR.
- I did not search temporal/trend signals specifically. TRec (arXiv 2009.05183) appeared in results but was not examined.

## Predictability ceiling of next-item prediction on sparse Amazon 5-core data

### Takeaway
I found no source that reports the fraction of Amazon 5-core test targets that never co-occur with the user's history. The closest evidence:
- A 2026 study shows that Amazon benchmarks are largely "shortcut-solvable" through local item-transition structure.
- Snippet-level evidence puts 1-hop transition-graph coverage of test targets at only about 9% on Beauty and CDs.
- Repeat/explore work shows that "explore" (novel) targets are far harder. That analysis uses session datasets, not Amazon.

### Cited Findings
- **Han et al., "An Embarrassingly Simple Graph Heuristic Reveals Shortcut-Solvable Benchmarks for Sequential Recommendation"** (arXiv 2605.07125, May 2026, preprint; authors include Jiliang Tang).
  - Method: start from the last 1–2 items, retrieve few-hop transition-graph candidates, and rank them by item-feature similarity.
  - Results: +38.10% NDCG@10 on Amazon Sports and +44.18% on CDs over the best baseline; competitive on 10 of 14 datasets.
  - Their explanation for the shortcut: low-branching local transitions, feature-smooth transitions, and weak dependence on long histories.
  - Coverage: a search snippet attributes **Cov@1 = 8.61% (Beauty) and 9.07% (CDs)** to this paper. These figures were not visible on the abstract page and should be checked in the full text.
  - Sources: [arXiv 2605.07125](https://arxiv.org/abs/2605.07125); [PDF](https://arxiv.org/pdf/2605.07125)
- **Li et al., "Repetition and Exploration in Sequential Recommendation"** (SIGIR 2023, DOI 10.1145/3539618.3591914).
  - Performance differs very strongly between repeat and explore targets, with explore much harder.
  - Neural SR models have a repetitive bias and fail to learn pure exploration.
  - The datasets are Diginetica and YOOCHOOSE (session data), not Amazon. I could not extract the numbers because the PDF was not parseable.
  - Sources: [ACM](https://dl.acm.org/doi/10.1145/3539618.3591914); [PDF](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/li-2023-repetition.pdf)
- A 2025 repeat-aware paper reports repeat ratios of 35.03% and 16.53% on its datasets. — [arXiv 2506.08531](https://arxiv.org/html/2506.08531v1)
- Amazon 5-core targets are almost never repeats, since 5-core review data rarely contains repeat reviews (background, unverified). Most Amazon targets are therefore "explore" targets.
- **RouteRec** supports a softer ceiling on MovieLens-1M: even the union of 5 diverse agents' top-10 lists contains the target for only 58.4% of users. — [arXiv 2607.09908](https://arxiv.org/abs/2607.09908)
- **Recformer** on Instruments: the best full-ranking R@10 is about 0.11, and cold-item R@10 is below 0.05. — [arXiv 2305.13731](https://arxiv.org/html/2305.13731)

### Inferences
- The model successes the literature reports on Amazon come largely from local transition structure and feature similarity to the last 1–2 items. Rows whose target lies outside this local neighbourhood are where all history-based models fail together. This matches the user's observation of identical failure sets.
- With 1-hop coverage around 9% (if confirmed) and a short average history (4.4), a large share of Instruments targets is probably unreachable from co-occurrence evidence at all. For those rows, only content/semantic similarity, LLM priors or population trends can contribute, and published cold/tail gains suggest these recover only a fraction of them.

### Gaps
- No published fraction of Amazon 5-core targets (Instruments or others) that never co-occur with any history item. This can be computed locally and should be: co-occurrence coverage, 1/2-hop transition coverage, and same-category rates for test targets.
- No formal entropy-based predictability bound (Song et al.-style) for Amazon review sequences was found.
- The Cov@1 figures need verification in the full text of arXiv 2605.07125.
