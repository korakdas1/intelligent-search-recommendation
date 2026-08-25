"""Search request validation without PostgreSQL."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.schemas.search import SearchRequest


def test_blank_query_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="   ")
    with pytest.raises(ValidationError):
        SearchRequest(query="")
    with pytest.raises(ValidationError):
        SearchRequest(query="\n\t")


def test_top_k_bounds() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="cream", top_k=0)
    with pytest.raises(ValidationError):
        SearchRequest(query="cream", top_k=101)
    parsed = SearchRequest(query=" cream ", top_k=10)
    assert parsed.query == "cream"
    assert parsed.top_k == 10
    assert parsed.retrieval_mode == "keyword"


def test_retrieval_mode_default_is_keyword() -> None:
    parsed = SearchRequest(query="leather")
    assert parsed.retrieval_mode == "keyword"
    assert parsed.fusion_method is None
    assert parsed.rerank_mode == "none"
    assert parsed.personalization_mode == "none"
    assert parsed.user_id is None


def test_retrieval_mode_semantic_accepted() -> None:
    parsed = SearchRequest(query="comfortable leather care", retrieval_mode="semantic")
    assert parsed.retrieval_mode == "semantic"
    assert parsed.fusion_method is None


def test_hybrid_mode_accepted() -> None:
    rrf = SearchRequest(query="cream", retrieval_mode="hybrid", fusion_method="rrf")
    weighted = SearchRequest(query="cream", retrieval_mode="hybrid", fusion_method="weighted")
    omitted = SearchRequest(query="cream", retrieval_mode="hybrid")
    assert rrf.fusion_method == "rrf"
    assert weighted.fusion_method == "weighted"
    assert omitted.fusion_method is None


def test_fusion_method_rejected_on_keyword_and_semantic() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="cream", retrieval_mode="keyword", fusion_method="rrf")
    with pytest.raises(ValidationError):
        SearchRequest(query="cream", retrieval_mode="semantic", fusion_method="weighted")


def test_invalid_fusion_method_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="cream", retrieval_mode="hybrid", fusion_method="ltr")


def test_search_fusion_method_on_keyword_http_422(client: TestClient) -> None:
    response = client.post("/search", json={"query": "cream", "fusion_method": "rrf"})
    assert response.status_code == 422


def test_search_invalid_fusion_method_http_422(client: TestClient) -> None:
    response = client.post(
        "/search",
        json={"query": "cream", "retrieval_mode": "hybrid", "fusion_method": "ltr"},
    )
    assert response.status_code == 422


def test_search_blank_query_http_422(client: TestClient) -> None:
    response = client.post("/search", json={"query": "  "})
    assert response.status_code == 422


def test_ltr_requires_hybrid() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="cream", retrieval_mode="keyword", rerank_mode="ltr")
    with pytest.raises(ValidationError):
        SearchRequest(query="cream", retrieval_mode="semantic", rerank_mode="ltr")
    parsed = SearchRequest(query="cream", retrieval_mode="hybrid", rerank_mode="ltr")
    assert parsed.rerank_mode == "ltr"


def test_invalid_rerank_mode_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="cream", retrieval_mode="hybrid", rerank_mode="magic")


def test_search_keyword_plus_ltr_http_422(client: TestClient) -> None:
    response = client.post(
        "/search",
        json={"query": "cream", "retrieval_mode": "keyword", "rerank_mode": "ltr"},
    )
    assert response.status_code == 422


def test_search_top_k_too_large_http_422(client: TestClient) -> None:
    response = client.post("/search", json={"query": "cream", "top_k": 1000})
    assert response.status_code == 422


def test_bounded_without_user_http_422(client: TestClient) -> None:
    response = client.post(
        "/search",
        json={
            "query": "cream",
            "retrieval_mode": "hybrid",
            "personalization_mode": "bounded",
        },
    )
    assert response.status_code == 422


def test_user_id_without_personalization_http_422(client: TestClient) -> None:
    response = client.post("/search", json={"query": "cream", "user_id": "U1"})
    assert response.status_code == 422


def test_keyword_personalization_http_422(client: TestClient) -> None:
    response = client.post(
        "/search",
        json={
            "query": "cream",
            "retrieval_mode": "keyword",
            "personalization_mode": "bounded",
            "user_id": "U1",
        },
    )
    assert response.status_code == 422


def test_query_longer_than_512_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="a" * 513)


def test_query_max_length_http_422(client: TestClient) -> None:
    response = client.post("/search", json={"query": "a" * 513})
    assert response.status_code == 422
