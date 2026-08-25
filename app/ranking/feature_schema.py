"""Versioned ranking-feature schema. Order is the model contract."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

FEATURE_VERSION = "rank-features-v1"
FEATURE_DTYPE = "float32"


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    description: str
    formula: str
    source: str
    missing_behavior: str
    group: str
    dtype: str = FEATURE_DTYPE


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="keyword_present",
        description="1 if the product appeared in the keyword candidate list.",
        formula="1 if keyword_rank is not None else 0",
        source="hybrid candidate metadata",
        missing_behavior="0 when absent from keyword retrieval",
        group="retrieval",
    ),
    FeatureSpec(
        name="semantic_present",
        description="1 if the product appeared in the semantic candidate list.",
        formula="1 if semantic_rank is not None else 0",
        source="hybrid candidate metadata",
        missing_behavior="0 when absent from semantic retrieval",
        group="retrieval",
    ),
    FeatureSpec(
        name="retrieval_source_count",
        description="How many retrievers contributed this product (1 or 2).",
        formula="keyword_present + semantic_present",
        source="hybrid candidate metadata",
        missing_behavior="0 only if the candidate list is empty (should not occur)",
        group="retrieval",
    ),
    FeatureSpec(
        name="keyword_score",
        description="Raw PostgreSQL ts_rank_cd. Not a probability.",
        formula="keyword_score if present else 0",
        source="keyword retriever",
        missing_behavior="0 when keyword_present=0",
        group="retrieval",
    ),
    FeatureSpec(
        name="semantic_score",
        description="Cosine / inner product on L2-normalized MiniLM vectors.",
        formula="semantic_score if present else 0",
        source="semantic retriever",
        missing_behavior="0 when semantic_present=0",
        group="retrieval",
    ),
    FeatureSpec(
        name="keyword_rr",
        description="Reciprocal keyword rank. Rank 1 is better than rank 100.",
        formula="1 / keyword_rank if present else 0",
        source="keyword rank (1-based)",
        missing_behavior="0 when keyword_present=0",
        group="retrieval",
    ),
    FeatureSpec(
        name="semantic_rr",
        description="Reciprocal semantic rank.",
        formula="1 / semantic_rank if present else 0",
        source="semantic rank (1-based)",
        missing_behavior="0 when semantic_present=0",
        group="retrieval",
    ),
    FeatureSpec(
        name="keyword_rank_fraction",
        description="Keyword rank scaled by candidate_k (smaller is better).",
        formula="keyword_rank / candidate_k if present else 0",
        source="keyword rank, candidate_k",
        missing_behavior="0 when keyword_present=0",
        group="retrieval",
    ),
    FeatureSpec(
        name="semantic_rank_fraction",
        description="Semantic rank scaled by candidate_k (smaller is better).",
        formula="semantic_rank / candidate_k if present else 0",
        source="semantic rank, candidate_k",
        missing_behavior="0 when semantic_present=0",
        group="retrieval",
    ),
    FeatureSpec(
        name="rrf_score",
        description="Reciprocal Rank Fusion score. Reuses rrf_contribution; k0=60.",
        formula="1/(k0+kw_rank) + 1/(k0+sem_rank); missing side contributes 0",
        source="app.search.fusion.rrf_contribution",
        missing_behavior="missing retriever contributes 0",
        group="retrieval",
    ),
    FeatureSpec(
        name="weighted_score",
        description="Min-max weighted fusion at alpha=0.5 unless overridden.",
        formula="alpha * K_norm + (1-alpha) * S_norm; per-query min-max inside each list",
        source="app.search.fusion.fuse_weighted",
        missing_behavior="missing side normalized score is 0",
        group="retrieval",
    ),
    FeatureSpec(
        name="title_exact_phrase",
        description="1 if the normalized query phrase is a substring of the title.",
        formula="1 if normalize(query) is non-empty and in normalize(title) else 0",
        source="query, product.title",
        missing_behavior="0 if query or title has no remaining text",
        group="lexical",
    ),
    FeatureSpec(
        name="title_all_query_tokens",
        description="1 if every unique query token appears in the title tokens.",
        formula="1 if query tokens and all of them are in title tokens else 0",
        source="query tokens, title tokens",
        missing_behavior="0 if the query has no alphanumeric tokens",
        group="lexical",
    ),
    FeatureSpec(
        name="title_overlap_count",
        description="Count of unique query tokens also in the title.",
        formula="|unique(query_tokens) ∩ unique(title_tokens)|",
        source="query tokens, title tokens",
        missing_behavior="0 if either side has no tokens",
        group="lexical",
    ),
    FeatureSpec(
        name="title_overlap_ratio",
        description="Fraction of unique query tokens found in the title.",
        formula="title_overlap_count / |unique(query_tokens)|",
        source="query tokens, title tokens",
        missing_behavior="0 if the query has no alphanumeric tokens (no divide-by-zero)",
        group="lexical",
    ),
    FeatureSpec(
        name="brand_present",
        description="1 if the catalog brand field is non-empty. Brand is never inferred from store.",
        formula="1 if brand is a non-empty string else 0",
        source="products.brand",
        missing_behavior="0 when brand is null or blank",
        group="brand",
    ),
    FeatureSpec(
        name="brand_exact",
        description="1 if the normalized brand phrase appears in the normalized query.",
        formula="1 if brand_present and normalize(brand) in normalize(query) else 0",
        source="query, products.brand",
        missing_behavior="0 when brand is missing",
        group="brand",
    ),
    FeatureSpec(
        name="brand_overlap",
        description="Fraction of unique query tokens that also appear in the brand tokens.",
        formula="|Q ∩ B| / |Q|",
        source="query tokens, brand tokens",
        missing_behavior="0 when brand or query tokens are empty",
        group="brand",
    ),
    FeatureSpec(
        name="description_missing",
        description="1 if description is null or blank.",
        formula="1 if description empty else 0",
        source="products.description",
        missing_behavior="1 when missing (this IS the missingness flag)",
        group="missingness",
    ),
    FeatureSpec(
        name="title_token_count",
        description="Number of alphanumeric tokens in the title (repeats counted).",
        formula="len(tokenize(title))",
        source="product.title",
        missing_behavior="0 if title tokenizes to nothing",
        group="missingness",
    ),
    FeatureSpec(
        name="description_token_count",
        description="Number of alphanumeric tokens in the description.",
        formula="len(tokenize(description)) if description else 0",
        source="product.description",
        missing_behavior="0 when description is missing",
        group="missingness",
    ),
    FeatureSpec(
        name="average_rating",
        description="Catalog average rating. A prior, not query relevance. 0 when missing.",
        formula="float(average_rating) if present else 0",
        source="products.average_rating",
        missing_behavior="0 when missing (this catalog's hybrid candidates all had ratings in E-006)",
        group="catalog",
    ),
    FeatureSpec(
        name="log_rating_count",
        description="log1p of rating_count. Deterministic; not a fitted scaler.",
        formula="log1p(rating_count) if rating_count is not None and >= 0 else 0",
        source="products.rating_count",
        missing_behavior="0 when rating_count is null",
        group="catalog",
    ),
    FeatureSpec(
        name="price_missing",
        description="1 if price is null or not positive.",
        formula="1 if price is None or price <= 0 else 0",
        source="products.price",
        missing_behavior="1 when missing or non-positive",
        group="catalog",
    ),
    FeatureSpec(
        name="log_price",
        description="log1p(price) when price > 0. No catalog-median imputation.",
        formula="log1p(price) if price > 0 else 0",
        source="products.price",
        missing_behavior="0 when price_missing=1",
        group="catalog",
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
FEATURE_COUNT = len(FEATURE_NAMES)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
FEATURE_NAMES_SHA256 = hashlib.sha256("\n".join(FEATURE_NAMES).encode("utf-8")).hexdigest()


def feature_name_at(index: int) -> str:
    if index < 0 or index >= FEATURE_COUNT:
        raise IndexError(f"feature index {index} out of range 0..{FEATURE_COUNT - 1}")
    return FEATURE_NAMES[index]


def schema_document() -> dict:
    return {
        "feature_version": FEATURE_VERSION,
        "feature_count": FEATURE_COUNT,
        "dtype": FEATURE_DTYPE,
        "features": [asdict(spec) | {"index": index} for index, spec in enumerate(FEATURE_SPECS)],
        "excluded": {
            "interaction_popularity": (
                "Deferred from v1. Full-catalog interaction counts would leak future events "
                "into later time-aware splits. Cutoff-bounded counts were not added to v1."
            ),
            "user_features": "User affinity is applied by bounded personalized search, not these features.",
            "subcategory": "Field is unused / always empty in this catalog.",
            "category_match": (
                "Dropped from v1 after E-006: category_present was constant (this slice is "
                "almost entirely All Beauty) and category_overlap was ~99% zero."
            ),
            "rating_missing": (
                "Dropped from v1 after E-006: every profiled candidate had average_rating."
            ),
            "labels": "Labels are never stored inside the feature vector.",
        },
    }
