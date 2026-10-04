# Evaluation

This document summarizes the official offline comparison. It does not retrain models or change frozen metrics.

There is no single best model. Search, personalized search, recommendations, and ANN answer different questions. Their NDCG values are not comparable.

## Metrics

Precision@K, Recall@K, MRR, and NDCG@K for K = 5, 10, 20. Macro-average per query or user. Missing targets score 0. Duplicate ranked IDs keep the first occurrence. Implementation: `app/evaluation/metrics.py`.

Several protocols have exactly one relevant item. Then Recall@K is 0 or 1 per row, and Precision@K is 1/K on a hit.

## Label classes

| Family | Label class | Meaning |
| --- | --- | --- |
| Search | synthetic | Deterministic product-derived queries. Not human-issued. Not editorial judgments. |
| Personalized search | synthetic query / observed target | Query is synthetic; the user’s later interaction is observed. |
| Recommendations | observed | Held-out future interaction. Not rating, CTR, or preference. |
| ANN | neighbor overlap | Fraction of exact FAISS neighbors recovered by HNSW. Not search relevance. |

## Synthetic search (603 TEST queries)

Protocol: frozen `ltr-synthetic-v1` TEST. Same query IDs for every system. Hybrid/LTR share `candidate_k=100`. Keyword and semantic are independent retrievers.

| System | R@10 | MRR | NDCG@10 |
| --- | --- | --- | --- |
| Keyword | 0.653 | 0.471 | **0.509** |
| Semantic | 0.299 | 0.182 | 0.204 |
| Hybrid weighted | 0.609 | 0.422 | 0.461 |
| Hybrid RRF | 0.600 | 0.397 | 0.439 |
| Hybrid + RankNet | 0.562 | 0.366 | 0.404 |

Keyword won this synthetic title/attribute protocol. RankNet lost to both fusions. Weighted beat RRF on the same hybrid union. Coverage: keyword 82.4%, semantic 46.1%, hybrid/LTR 84.6%. Uncovered queries count as misses.

### Provenance for future search evaluations

New LTR builds sort eligible source products by product ID before seeded shuffling and record `source_order_policy`. Their manifests include exact SHA-256 checksums for `queries.json` and `candidates.parquet`. New RankNet builds validate those inputs before training and record their observed identity under `ltr_dataset_provenance`, together with model, scaler, and configuration checksums.

Official search evaluation validates declared input checksums and ranker/dataset linkage before evaluating queries. Output records the query and available candidate-file hashes, loaded ranker identity, actual semantic backend/FAISS type and metric, semantic catalog identity, and effective search configuration. Artifact directories use logical names rather than local absolute paths. Candidate-identity failures return nonzero without creating or replacing the comparison JSON; valid output retains atomic writes and overwrite protection.

Historical manifests without these declarations remain loadable with warnings and `legacy_unverified` provenance. Hashing their current bytes does not establish missing historical build identity or ranker/dataset linkage. Missing optional candidate parquet does not prevent search evaluation, but its identity remains unverified; declared mismatches are rejected. Historical E-012 metrics and artifacts were not regenerated. A future rebuild, even under the same dataset-version label, is identified by its own checksums.

## Personalized search (22,554 usable queries)

Unpersonalized hybrid RRF vs bounded personalization on identical candidate IDs (22,554/22,554). Candidate coverage 80.7%.

| | R@10 | MRR | NDCG@10 |
| --- | --- | --- | --- |
| Unpersonalized | 0.418 | 0.223 | 0.259 |
| Personalized | 0.465 | 0.275 | 0.312 |

Personalization stays opt-in. Queries remain synthetic.

## Recommendations (22,628 users)

Same `recsys-eval-v1` hold-out for popularity, content, CF, and hybrid-rec.

| System | R@10 | MRR | NDCG@10 |
| --- | --- | --- | --- |
| Popularity | 0.0190 | 0.0096 | 0.0097 |
| Content | 0.0511 | 0.0308 | 0.0345 |
| CF | 0.0186 | 0.0120 | 0.0125 |
| Hybrid-rec | **0.0533** | **0.0340** | **0.0370** |

CF catalog coverage 9.43%. On hidden items with training degree 0, CF R@10 = 0; content/hybrid recover some of them (R@10 0.037). Hybrid union target coverage 19.1%.

## Exact vs ANN

Verified HNSW vs `IndexFlatIP` on the same MiniLM vectors (80 queries, K=10). At `efSearch=128`, neighbor Recall@10 = **0.920**, median latency 0.493 ms vs exact 17.897 ms. Serving remains exact because the catalog is ~112k vectors and HNSW is not exact.

## What these numbers do not prove

Human search relevance, production CTR, conversion, A/B wins, customer satisfaction, or that HNSW matches exact search quality.
