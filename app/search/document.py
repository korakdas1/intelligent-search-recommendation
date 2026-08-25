"""PostgreSQL full-text document expression.

The database owns ``products.search_document`` as a stored generated
column. Application code must not write the vector.
"""

from __future__ import annotations

SEARCH_CONFIG = "english"
GIN_INDEX_NAME = "ix_products_search_document"

# ``to_tsvector(regconfig, text)`` is IMMUTABLE, which PostgreSQL requires
# for generated columns. The text-config overload is only STABLE.
SEARCH_DOCUMENT_SQL = """
setweight(to_tsvector('english'::regconfig, coalesce(title, '')), 'A')
|| setweight(to_tsvector('english'::regconfig, coalesce(brand, '')), 'A')
|| setweight(to_tsvector('english'::regconfig, coalesce(description, '')), 'B')
|| setweight(
    to_tsvector(
        'english'::regconfig,
        coalesce(category, '') || ' ' || coalesce(subcategory, '')
    ),
    'C'
)
""".strip()

FIELD_WEIGHTS = {
    "A": ("title", "brand"),
    "B": ("description",),
    "C": ("category", "subcategory"),
}

PARSER_NAME = "websearch_to_tsquery"
RANKER_NAME = "ts_rank_cd"
RETRIEVAL_MODE = "keyword"
