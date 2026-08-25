#!/usr/bin/env python3
"""Ingest processed Parquet products and interactions into PostgreSQL."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.core.events import DATASET_VERSION_FULL
from app.core.config import get_settings
from app.db.ingest import ingest_parquet
from app.db.session import get_engine, reset_engine

logger = logging.getLogger("ingest_database")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        default="data/processed/amazon_reviews_2023/all_beauty",
        help="Directory containing products.parquet and interactions.parquet",
    )
    parser.add_argument(
        "--dataset-version",
        default=DATASET_VERSION_FULL,
        help="Stable dataset_version primary key",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Directory for report.json (default artifacts/ingestion/<dataset-version>)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    processed = Path(args.processed_dir)
    report_dir = Path(args.report_dir) if args.report_dir else Path("artifacts/ingestion") / args.dataset_version
    settings = get_settings()
    logger.info("Ingesting %s as %s", processed, args.dataset_version)
    logger.info("Database host %s db %s", settings.postgres_host, settings.postgres_db)
    reset_engine()
    engine = get_engine()
    try:
        report = ingest_parquet(
            engine,
            processed,
            dataset_version=args.dataset_version,
            report_dir=report_dir,
        )
    except Exception:
        logger.exception("Ingestion failed")
        return 1
    print(json.dumps({k: report[k] for k in (
        "dataset_version",
        "products_read",
        "users_derived",
        "interactions_read",
        "interactions_valid",
        "orphan_interaction_rows",
        "orphan_product_ids",
        "final_table_counts",
        "elapsed",
    )}, indent=2))
    print("report", report_dir / "report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
