# Interview guide

Practical answers from this implementation. Do not claim human relevance, CTR, or production A/B results.

## Why PostgreSQL full-text search?

The catalog already lives in PostgreSQL. `tsvector` / `websearch_to_tsquery` / `ts_rank_cd` is a real lexical baseline with stemming and a GIN index, without adding Elasticsearch. Queries are bound as parameters, not concatenated SQL.

## Why FAISS?

Embeddings are 384-d MiniLM vectors. FAISS `IndexFlatIP` on L2-normalized vectors is exact inner-product / cosine search and is already fast on ~112k items.

## Why exact IndexFlatIP instead of HNSW for serving?

HNSW is implemented and measured. At `efSearch=128` it recovered 92% of exact neighbors and was faster, but it is not exact. Exact search median was about 18 ms. Serving stays exact so retrieval quality is not silently approximated.

## What does ANN Recall@10 mean?

It is the overlap between HNSW’s top-10 neighbors and exact FAISS top-10 neighbors on the same vectors. It is **not** search-relevance Recall@10.

## Why did keyword beat semantic?

Official search labels are synthetic title and attribute strings. Those queries are lexical by construction. Keyword NDCG@10 was 0.509 vs semantic 0.204. That is a useful engineering result, not a claim about human queries.

## Why did RankNet lose?

Pairwise RankNet accuracy was high, but list metrics (NDCG/MRR) on the frozen synthetic TEST split were worse than hybrid weighted and RRF. RankNet stays opt-in. A more complex ranker is not automatically better.

## Why hybrid recommendations?

On the same observed 2-core hold-out, hybrid-rec slightly beat content (R@10 0.0533 vs 0.0511) and recovered some cold items that pure CF cannot. The lift is small. Omitted API method remains content for compatibility; the demo requests hybrid explicitly.

## Why was CF weak?

All_Beauty is sparse. BPR-MF sees 10,618 items (~9% of the catalog). On hidden items with zero training degree, CF Recall@10 is 0 by construction. That is expected, not a training bug.

## How does cold start work?

Hybrid recommendations: no usable history → popularity; user outside CF → content + popularity; unknown user → 404. Personalized search: known user with no signal → ordinary search ranking; unknown user → 404. Search does not fall back to popularity.

## How does personalization avoid query drift?

It only permutes hybrid (or LTR) candidate IDs. Formula `Final = Q(1 + γP)` with `γ ≤ 0.30`. Tests assert the ID set is unchanged. Favorite products that search never retrieved cannot appear.

## What is NDCG?

Normalized Discounted Cumulative Gain. Relevant items higher in the list contribute more. We use binary relevance and macro-average. With one target, it mainly rewards how high that item is ranked.

## How did you prevent leakage?

Recommendation splits are time-aware leave-last-item-out. Popularity counts and CF pairs use TRAIN only. RankNet scaler is train-only. Personalization profiles use past history only. Inner tuning never used the external test. Search-log ranks are not treated as relevance labels.

## Limitations?

No human search labels. Synthetic queries favor keyword. Rec metrics are observed interactions on a 2-core, not preference. Hybrid rec candidate pools are bounded. CF does not cover the catalog. No production traffic.

## Résumé-safe bullets

- Built a hybrid product search and recommendation engine over 112,578 products using PostgreSQL full-text search, MiniLM/FAISS, PyTorch RankNet and BPR-MF, hybrid recommendation fusion, and bounded personalized reranking.
- Evaluated retrieval and recommendation systems with Precision@K, Recall@K, MRR, and NDCG@K under documented synthetic search and observed hold-out protocols; reported that keyword won the synthetic search test and RankNet did not beat hybrid fusion.
- Measured exact vs HNSW FAISS neighbor overlap (Recall@10 = 0.920 at efSearch=128) and kept exact IndexFlatIP as the serving index on this catalog scale.
