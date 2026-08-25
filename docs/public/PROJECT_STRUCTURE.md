# Project structure

```text
app/                    FastAPI application
  api/                  HTTP routes including /demo
  search/               keyword, semantic, hybrid, LTR, personalization
  ranking/              features and RankNet
  recommendations/      content, popularity, BPR-MF, hybrid rec
  evaluation/           shared Precision/Recall/MRR/NDCG
  embeddings/           MiniLM encoding helpers
  db/ models/ schemas/  persistence and request models
  static/demo/          local HTML/CSS/JS demo
scripts/                dataset, ingest, train, evaluate, demo launcher
tests/                  unit and PostgreSQL integration tests
alembic/                schema migrations
docs/public/            public architecture, evaluation, API, dataset
```

Gitignored locally (never commit): `.env`, `data/raw/`, `data/processed/`, embeddings, FAISS indexes, `.pt` weights, evaluation dumps.

Runtime policy JSON lives under `app/search/policies/` and `app/recommendations/policies/`.
