"""Prepare pipeline tests using in-memory / tmp JSONL fixtures."""

import json
from pathlib import Path

from app.data.prepare import normalize_interactions, normalize_products


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_normalize_products_drops_duplicates_and_missing_title(tmp_path: Path) -> None:
    path = tmp_path / "meta.jsonl"
    _write_jsonl(
        path,
        [
            {"parent_asin": "P1", "title": "One", "price": 1.5},
            {"parent_asin": "P1", "title": "One duplicate"},
            {"parent_asin": "P2", "title": "   "},
            {"parent_asin": "P3", "title": "Three", "description": ["Nice"]},
        ],
    )
    frame, rejects, raw_stats = normalize_products(path)
    assert "parent_asin" in raw_stats["all_keys"]
    assert set(frame["product_id"]) == {"P1", "P3"}
    assert rejects["duplicate_parent_asin"] == 1
    assert rejects["raw_rows"] == 4
    assert any("title" in reason for reason in rejects)
    assert raw_stats["rows"] == 4
    assert raw_stats["fields"]["title"]["present"] == 3
    assert raw_stats["fields"]["title"]["missing"] == 1


def test_normalize_interactions_drops_exact_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "reviews.jsonl"
    base = {
        "user_id": "U1",
        "parent_asin": "P1",
        "asin": "C1",
        "rating": 4.0,
        "timestamp": 1588687728923,
        "verified_purchase": True,
    }
    later = dict(base)
    later["timestamp"] = 1588687729923
    _write_jsonl(path, [base, base, later])
    frame, rejects, _raw_stats = normalize_interactions(path)
    assert len(frame) == 2
    assert rejects["exact_duplicate_user_item_time"] == 1
    assert bool(frame.iloc[0]["verified_purchase"]) is True
    assert (frame["event_type"] == "review").all()
