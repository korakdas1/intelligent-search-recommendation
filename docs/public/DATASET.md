# Dataset

## Source

Amazon Reviews 2023, category **All_Beauty**, from McAuley Lab via Hugging Face (`McAuley-Lab/Amazon-Reviews-2023`). This project does **not** redistribute the raw files.

The dataset supplies product metadata and implicit interactions (reviews). It does **not** supply human search queries or search-relevance judgments.

## Role in this system

- Catalog text → keyword search document and MiniLM embeddings
- User–item events → recommendation history, popularity counts, BPR-MF training pairs
- A 2-core subset → official recommendation hold-out (`recsys-eval-v1`: 22,628 users / 57,114 pairs)

## Measured catalog after ingest

| Item | Count |
| --- | --- |
| Products | 112,578 |
| Users | 631,915 |
| Interactions | 694,170 |

The slice is sparse. Most users have a single event. Collaborative filtering is trained only on the filtered 2-core, so it cannot represent the full catalog (about 9.4% of items appear in the CF model).

## Normalization

Scripts download JSONL, convert to Parquet, clean titles/text, and ingest into PostgreSQL. Orphan interaction rows are skipped. Exact duplicates were dropped before Parquet.

## PostgreSQL (fresh machine)

This is manual. Example on Ubuntu:

```bash
sudo apt install -y postgresql postgresql-contrib
sudo systemctl enable --now postgresql
sudo -u postgres psql <<'SQL'
CREATE USER isre WITH PASSWORD 'change-me-locally';
CREATE DATABASE intelligent_search_recommendation OWNER isre;
CREATE DATABASE intelligent_search_recommendation_test OWNER isre;
GRANT ALL PRIVILEGES ON DATABASE intelligent_search_recommendation TO isre;
GRANT ALL PRIVILEGES ON DATABASE intelligent_search_recommendation_test TO isre;
SQL
sudo -u postgres psql -d intelligent_search_recommendation -c "GRANT ALL ON SCHEMA public TO isre;"
sudo -u postgres psql -d intelligent_search_recommendation_test -c "GRANT ALL ON SCHEMA public TO isre;"
```

Copy `.env.example` to `.env` and set `POSTGRES_PASSWORD` to match. Then:

```bash
alembic upgrade head
python scripts/download_dataset.py
python scripts/prepare_dataset.py
python scripts/ingest_database.py
```

## Limitations

- Beauty-only catalog; titles can be noisy
- Reviews are not search queries
- Observed hold-out ≠ stated preference
- Do not commit raw or processed dumps
