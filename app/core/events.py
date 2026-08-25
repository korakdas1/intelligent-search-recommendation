"""Interaction event type names.

These are labels, not relevance weights. Numeric maps belong in later
experiment configuration (D-013).
"""

EVENT_TYPES = ("view", "click", "add_to_cart", "purchase", "review")
EVENT_TYPE_SET = frozenset(EVENT_TYPES)
DEFAULT_CURRENCY = "USD"
DATASET_VERSION_FULL = "amazon2023_all_beauty_phase2-v1"
