"""Stable semantic-embedding identifiers.

These are configuration defaults, not a substitute for the artifact manifest.
"""

from __future__ import annotations

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
SEMANTIC_TEXT_VERSION = "semantic-text-v1"
SEMANTIC_RETRIEVAL_MODE = "semantic"
SEMANTIC_SOURCE = "semantic"
EXACT_INDEX_TYPE = "IndexFlatIP"
ANN_INDEX_TYPE = "IndexHNSWFlat"
FAISS_METRIC = "inner_product"
VECTOR_DTYPE = "float32"
MAX_FIELD_CHARS = 4000

# Serving and artifact defaults. Serving may stay on exact search even after ANN exists.
DEFAULT_INDEX_BACKEND = "flat"
DEFAULT_BATCH_SIZE = 64
DEFAULT_HNSW_M = 32
DEFAULT_HNSW_EF_CONSTRUCTION = 200
DEFAULT_HNSW_EF_SEARCH = 64
