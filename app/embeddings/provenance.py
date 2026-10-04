"""Streaming identity of the ordered inputs to semantic encoding."""

from __future__ import annotations

import hashlib


class SemanticCatalogFingerprint:
    """SHA-256 over product IDs and their exact semantic text, in ID order.

    Callers supply rows in ascending PostgreSQL ``C`` collation order (UTF-8
    byte order), without duplicates. Each UTF-8 value is framed by its byte
    length as an unsigned 8-byte big-endian integer. The domain prefix fixes
    this algorithm's identity. Batch boundaries do not affect the digest;
    only one field's bytes need to be held in memory at a time.
    """

    def __init__(self) -> None:
        self._digest = hashlib.sha256(b"semantic-catalog-sha256-v1\x00")

    def update(self, product_id: str, semantic_text: str) -> None:
        for value in (product_id, semantic_text):
            encoded = value.encode("utf-8")
            self._digest.update(len(encoded).to_bytes(8, "big"))
            self._digest.update(encoded)

    def hexdigest(self) -> str:
        return self._digest.hexdigest()
