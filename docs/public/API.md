# HTTP API

Base URL for local development: `http://127.0.0.1:8000`. Interactive OpenAPI: `/docs`.

`/health` does not check PostgreSQL. `/ready` does.

## GET /health

Process liveness.

```bash
curl -sS http://127.0.0.1:8000/health
```

## GET /ready

PostgreSQL ping plus semantic / RankNet / CF artifact probes. Optional ranker and CF may be `skipped` when not required.

```bash
curl -sS http://127.0.0.1:8000/ready
```

## GET /demo

HTML demonstration UI. Result cards emphasize product identity; scores, product IDs, and raw source strings sit in collapsed technical details. The page still calls the same `/search` and `/recommendations/*` APIs.

## POST /search

JSON body:

| Field | Default | Notes |
| --- | --- | --- |
| `query` | required | 1–512 characters after strip |
| `top_k` | 10 | 1–100 |
| `retrieval_mode` | `keyword` | `keyword` \| `semantic` \| `hybrid` |
| `fusion_method` | omitted | `rrf` \| `weighted`; hybrid only |
| `rerank_mode` | `none` | `ltr` requires hybrid |
| `personalization_mode` | `none` | `bounded` requires hybrid and `user_id` |
| `user_id` | omitted | max 256 characters |

```bash
curl -sS http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"baby shampoo","top_k":5,"retrieval_mode":"keyword"}'

curl -sS http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"leather conditioner","top_k":5,"retrieval_mode":"hybrid","fusion_method":"weighted"}'

curl -sS http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"sunscreen","top_k":5,"retrieval_mode":"hybrid","fusion_method":"weighted","rerank_mode":"ltr"}'
```

Invalid enums and blank queries return **422**. Unknown personalized `user_id` returns **404**. Missing semantic or RankNet artifacts for those modes return **503**. Scores are ranking scores, not probabilities.

## GET /products/{product_id}

Exact catalog lookup. **404** if missing.

## GET /recommendations/similar/{product_id}

Embedding neighbors of a product. `top_k` 1–100.

```bash
curl -sS 'http://127.0.0.1:8000/recommendations/similar/B006UNWU2W?top_k=5'
```

## GET /recommendations/user/{user_id}

| Query | Default | Notes |
| --- | --- | --- |
| `method` | `content` | `content` \| `cf` \| `hybrid` |
| `top_k` | 10 | 1–100 |

The demo sends `method=hybrid`. Invalid `method` → **422**. Unknown user → **404**. `method=cf` is strict (no popularity fallback). Hybrid uses documented cold-start fallbacks.

```bash
curl -sS 'http://127.0.0.1:8000/recommendations/user/USER_ID?method=hybrid&top_k=5'
```

## GET /recommendations/trending

Historical interaction-count popularity, not live traffic.

```bash
curl -sS 'http://127.0.0.1:8000/recommendations/trending?top_k=5'
```

## POST /interactions

Records a user–product event. Not used by the demo UI.
