"""Ranking features against the PostgreSQL test database. No MiniLM download."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.repositories.products import get_products_by_ids
from app.embeddings.normalize import l2_normalize
from app.models.dataset import DatasetVersion
from app.models.product import Product
from app.ranking.feature_schema import FEATURE_COUNT, FEATURE_INDEX
from app.ranking.features import extract_features
from app.ranking.product_view import ProductView
from app.ranking.retrieval import collect_fused_candidates
from app.ranking.validate import validate_feature_batch
from app.search.faiss_index import build_flat_ip_index
from app.search.runtime import SemanticRuntime, reset_semantic_runtime, set_semantic_runtime

pytestmark = pytest.mark.postgres


class ScriptedEncoder:
    model_name = "fake-encoder"
    embedding_dim = 4
    device = "cpu"

    def encode(self, texts, *, batch_size: int = 32, show_progress: bool = False):
        del batch_size, show_progress
        rows = []
        for text in texts:
            if "leather" in text.lower():
                rows.append([0.0, 1.0, 0.0, 0.0])
            else:
                rows.append([1.0, 0.0, 0.0, 0.0])
        return l2_normalize(np.array(rows, dtype=np.float32))


def _seed(session: Session) -> None:
    session.add(
        DatasetVersion(
            dataset_version="search_v1",
            source_name="fixture",
            row_counts={},
            checksums={},
        )
    )
    session.add_all(
        [
            Product(
                product_id="PAAA",
                dataset_version="search_v1",
                title="Alpha Cream",
                category="All Beauty",
                brand="Acme",
                price=Decimal("1.00"),
                average_rating=Decimal("4.00"),
                rating_count=3,
                currency="USD",
            ),
            Product(
                product_id="PBBB",
                dataset_version="search_v1",
                title="Beta Leather Balm",
                category="All Beauty",
                brand="Howard",
                description=None,
                price=None,
                currency="USD",
            ),
            Product(
                product_id="PCCC",
                dataset_version="search_v1",
                title="Gamma Soap",
                category="All Beauty",
                currency="USD",
            ),
        ]
    )
    session.commit()


def _install_runtime() -> SemanticRuntime:
    embeddings = l2_normalize(np.eye(3, 4, dtype=np.float32))
    runtime = SemanticRuntime(
        artifact_version="test-semantic",
        dataset_version="search_v1",
        model_name="fake-encoder",
        embedding_dim=4,
        product_ids=np.array(["PAAA", "PBBB", "PCCC"], dtype="U32"),
        index=build_flat_ip_index(embeddings),
        backend="flat",
        embedding_manifest={"model_revision": "test"},
        index_manifest={},
        encoder=ScriptedEncoder(),
        skip_catalog_count_check=False,
    )
    set_semantic_runtime(runtime)
    return runtime


def test_batch_fetch_then_extract_preserves_order(db_session: Session) -> None:
    _seed(db_session)
    _install_runtime()
    try:
        fused, depth = collect_fused_candidates(
            db_session, "leather", candidate_k=3, top_k=3
        )
        assert depth >= 3
        product_ids = [row.product_id for row in fused]
        fetched = get_products_by_ids(db_session, product_ids)
        views = {pid: ProductView.from_product(product) for pid, product in fetched.items()}
        batch = extract_features("leather", fused, views, candidate_k=depth)
        validate_feature_batch(batch)
        assert batch.product_ids == tuple(product_ids)
        assert batch.values.shape == (len(product_ids), FEATURE_COUNT)
        leather = views["PBBB"]
        assert leather.description is None
        assert leather.price is None
        b_index = product_ids.index("PBBB")
        assert batch.values[b_index, FEATURE_INDEX["description_missing"]] == 1.0
        assert batch.values[b_index, FEATURE_INDEX["price_missing"]] == 1.0
        assert batch.values[b_index, FEATURE_INDEX["title_all_query_tokens"]] == 1.0
    finally:
        reset_semantic_runtime()


def test_hybrid_api_unchanged_by_ranking_package(
    client: TestClient, db_session: Session
) -> None:
    _seed(db_session)
    _install_runtime()
    try:
        response = client.post(
            "/search",
            json={
                "query": "leather",
                "top_k": 3,
                "retrieval_mode": "hybrid",
                "fusion_method": "rrf",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["retrieval_mode"] == "hybrid"
        assert "feature_version" not in body
        assert all(row["source"] == "hybrid" for row in body["results"])
    finally:
        reset_semantic_runtime()
