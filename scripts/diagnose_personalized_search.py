#!/usr/bin/env python3
"""Print baseline vs personalized candidate IDs for one real request."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.session import get_session_factory
from app.search.personalization_eval import hybrid_baseline_candidates
from app.search.personalization_policy import load_personalization_policy
from app.search.personalization_service import personalize_baseline
from app.search.service import run_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--fusion-method", default="rrf")
    parser.add_argument("--candidate-k", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy = load_personalization_policy()
    candidate_k = args.candidate_k or policy.candidate_k
    factory = get_session_factory()
    with factory() as session:
        baseline, _collected = hybrid_baseline_candidates(
            session,
            query=args.query,
            fusion_method=args.fusion_method,
            candidate_k=candidate_k,
            top_k=args.top_k,
        )
        outcome = personalize_baseline(
            session,
            user_id=args.user_id,
            baseline=baseline,
            policy=policy,
        )
        base_ids = [row.product_id for row in baseline]
        pers_ids = [row.product_id for row in outcome.candidates]
        same = set(base_ids) == set(pers_ids) and len(base_ids) == len(pers_ids)
        anonymous = run_search(
            session,
            query=args.query,
            top_k=args.top_k,
            retrieval_mode="hybrid",
            fusion_method=args.fusion_method,
            log=False,
        )
        personalized = run_search(
            session,
            query=args.query,
            top_k=args.top_k,
            retrieval_mode="hybrid",
            fusion_method=args.fusion_method,
            personalization_mode="bounded",
            user_id=args.user_id,
            log=False,
        )
        payload = {
            "query": args.query,
            "user_id": args.user_id,
            "same_candidate_set": same,
            "candidate_count": len(base_ids),
            "baseline_top_k": [row.product_id for row in anonymous.results],
            "personalized_top_k": [row.product_id for row in personalized.results],
            "personalization_applied": personalized.personalization_applied,
            "signals": personalized.personalization_signals,
        }
        print(json.dumps(payload, indent=2))
        return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
