#!/usr/bin/env python3
"""Train BPR-MF (bpr-mf-v1) on recsys-eval-v1 TRAIN pairs.

Selects embedding dim on cf-tune-v1 validation only. Does not read or
optimize against the recsys-eval-v1 external test metrics.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from app.core.config import get_settings
from app.core.events import DATASET_VERSION_FULL
from app.db.repositories.artifacts import upsert_artifact_version
from app.db.session import get_session_factory, reset_engine
from app.embeddings.checksums import sha256_file
from app.recommendations.cf_artifacts import save_cf_bundle
from app.recommendations.cf_constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_L2,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    MODEL_TYPE,
    MODEL_VERSION,
    NEGATIVE_POLICY,
    SELECTION_METRIC,
    TUNE_PROTOCOL,
)
from app.recommendations.cf_data import (
    RecsysEvalIncompatibleError,
    assert_no_zero_train_item,
    build_mappings,
    external_train_pairs,
    load_recsys_eval_split,
    mapping_checksum,
    pairs_to_index_arrays,
    user_positive_sets,
)
from app.recommendations.cf_train import resolve_device, train_bpr_mf
from app.recommendations.cf_tune import assert_tune_leakage_free, build_cf_tune_split
from app.recommendations.constants import EVALUATION_VERSION


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-version", default=EVALUATION_VERSION)
    parser.add_argument("--model-version", default=MODEL_VERSION)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--embedding-dim", type=int, default=None, help="Skip dim sweep if set")
    parser.add_argument("--dims", default="32,64")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--regularization", type=float, default=DEFAULT_L2)
    parser.add_argument("--epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Tiny subset smoke run; not the official model")
    parser.add_argument("--pilot-users", type=int, default=256)
    parser.add_argument("--register-artifact", action="store_true")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def _pack_validation(users, mappings) -> tuple[list[int], list[int | None], list[list[int]], np.ndarray, np.ndarray]:
    user_indices: list[int] = []
    hidden_indices: list[int | None] = []
    seen_indices: list[list[int]] = []
    pair_users: list[int] = []
    pair_items: list[int] = []
    for user in users:
        if user.user_id not in mappings.user_to_index:
            continue
        user_index = mappings.user_to_index[user.user_id]
        hidden = mappings.product_to_index.get(user.hidden_product_id)
        seen = [
            mappings.product_to_index[product_id]
            for product_id in user.inner_train_product_ids
            if product_id in mappings.product_to_index
        ]
        user_indices.append(user_index)
        hidden_indices.append(hidden)
        seen_indices.append(seen)
        if hidden is not None:
            pair_users.append(user_index)
            pair_items.append(int(hidden))
    return (
        user_indices,
        hidden_indices,
        seen_indices,
        np.asarray(pair_users, dtype=np.int64),
        np.asarray(pair_items, dtype=np.int64),
    )


def main() -> int:
    args = parse_args()
    settings = get_settings()
    split_dir = Path(args.split_dir or Path(settings.artifacts_root) / "evaluation" / args.evaluation_version)
    out_dir = Path(args.out_dir or Path(settings.artifacts_root) / "models" / args.model_version)
    if out_dir.exists() and not args.force:
        print(f"refusing to overwrite {out_dir}; pass --force", file=sys.stderr)
        return 2
    try:
        split = load_recsys_eval_split(split_dir, require_frozen_identity=True)
    except RecsysEvalIncompatibleError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    started = time.perf_counter()
    users = list(split.users)
    if args.pilot:
        users = users[: max(8, args.pilot_users)]
        from app.recommendations.cf_data import RecsysEvalSplit

        split = RecsysEvalSplit(
            evaluation_version=split.evaluation_version,
            users=tuple(users),
            two_core_pairs=split.two_core_pairs,
            evaluation_users=len(users),
            hidden_checksum=split.hidden_checksum,
            catalog_size=split.catalog_size,
            directory=split.directory,
        )

    tune = build_cf_tune_split(split)
    assert_tune_leakage_free(split, tune)
    full_pairs = external_train_pairs(split)
    inner_pairs = list(tune.inner_train_pairs)
    inner_mappings = build_mappings(inner_pairs)
    assert_no_zero_train_item(inner_mappings, inner_pairs)
    inner_users, inner_items = pairs_to_index_arrays(inner_pairs, inner_mappings)
    inner_pos = user_positive_sets(inner_pairs, inner_mappings)
    val_pack = _pack_validation(tune.validation_users, inner_mappings)

    device = resolve_device(args.device)
    if args.embedding_dim is not None:
        dims = [int(args.embedding_dim)]
    else:
        dims = [int(part.strip()) for part in args.dims.split(",") if part.strip()]
    if not dims:
        dims = [DEFAULT_EMBEDDING_DIM]

    tuning_started = time.perf_counter()
    runs: list[dict[str, Any]] = []
    winner: dict[str, Any] | None = None
    for dim in dims:
        print(f"tuning dim={dim} on {TUNE_PROTOCOL} (external test unused)", flush=True)
        _model, summary = train_bpr_mf(
            n_users=inner_mappings.n_users,
            n_items=inner_mappings.n_items,
            embedding_dim=dim,
            train_user_index=inner_users,
            train_item_index=inner_items,
            user_positives=inner_pos,
            product_ids=inner_mappings.product_ids,
            val_user_indices=val_pack[0],
            val_hidden_item_indices=val_pack[1],
            val_seen_item_indices=val_pack[2],
            val_pairs_user=val_pack[3],
            val_pairs_item=val_pack[4],
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            l2=args.regularization,
            max_epochs=3 if args.pilot else args.epochs,
            patience=2 if args.pilot else args.patience,
            seed=args.seed,
            device=device,
        )
        ndcg = summary["best_validation_ndcg@10"]
        print(
            f"dim={dim} best_epoch={summary['best_epoch']} "
            f"val_NDCG@10={ndcg} val_R@10={summary['history'][summary['best_epoch']-1].get('validation_R@10') if summary['history'] else None}",
            flush=True,
        )
        run = {"embedding_dim": dim, "summary": summary}
        runs.append(run)
        if winner is None:
            winner = run
            continue
        win_ndcg = winner["summary"]["best_validation_ndcg@10"]
        if ndcg > win_ndcg + 1e-12 or (
            abs(ndcg - win_ndcg) <= 1e-12 and dim < winner["embedding_dim"]
        ):
            winner = run
    tuning_seconds = time.perf_counter() - tuning_started
    assert winner is not None
    selected_dim = int(winner["embedding_dim"])
    selected_epoch = int(winner["summary"]["best_epoch"])
    print(
        f"selected dim={selected_dim} best_epoch={selected_epoch} by validation {SELECTION_METRIC}",
        flush=True,
    )

    final_mappings = build_mappings(full_pairs)
    assert_no_zero_train_item(final_mappings, full_pairs)
    final_users, final_items = pairs_to_index_arrays(full_pairs, final_mappings)
    final_pos = user_positive_sets(full_pairs, final_mappings)
    retrain_started = time.perf_counter()
    print(
        f"final retrain dim={selected_dim} epochs={selected_epoch} "
        f"users={final_mappings.n_users} items={final_mappings.n_items} pairs={len(full_pairs)}",
        flush=True,
    )
    model, final_summary = train_bpr_mf(
        n_users=final_mappings.n_users,
        n_items=final_mappings.n_items,
        embedding_dim=selected_dim,
        train_user_index=final_users,
        train_item_index=final_items,
        user_positives=final_pos,
        product_ids=final_mappings.product_ids,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        l2=args.regularization,
        max_epochs=selected_epoch,
        patience=args.patience,
        seed=args.seed,
        device=device,
        fixed_epochs=selected_epoch,
    )
    retrain_seconds = time.perf_counter() - retrain_started
    user_checksum, item_checksum = mapping_checksum(final_mappings.user_ids, final_mappings.product_ids)
    catalog_size = split.catalog_size
    catalog_coverage = (
        None if catalog_size in (None, 0) else final_mappings.n_items / float(catalog_size)
    )
    config = {
        "model_version": args.model_version,
        "model_type": MODEL_TYPE,
        "embedding_dim": selected_dim,
        "n_users": final_mappings.n_users,
        "n_items": final_mappings.n_items,
        "loss": "bpr_softplus_plus_batch_l2",
        "optimizer": "adam",
        "learning_rate": args.learning_rate,
        "regularization": args.regularization,
        "batch_size": args.batch_size,
        "negatives_per_positive": 1,
        "negative_sampling": NEGATIVE_POLICY,
        "seed": args.seed,
        "recsys_eval_version": EVALUATION_VERSION,
        "tune_protocol": TUNE_PROTOCOL,
        "selection_metric": SELECTION_METRIC,
        "selected_best_epoch": selected_epoch,
        "no_user_item_bias": True,
        "implicit_unique_pairs": True,
        "external_test_unused_for_tuning": True,
    }
    counts = model.embedding_parameter_counts()
    artifact_manifest = {
        "model_version": args.model_version,
        "model_type": MODEL_TYPE,
        "recsys_eval_version": EVALUATION_VERSION,
        "dataset_version": DATASET_VERSION_FULL,
        "tune_protocol": TUNE_PROTOCOL,
        "hidden_checksum": split.hidden_checksum,
        "evaluation_users": len(users) if args.pilot else split.evaluation_users,
        "two_core_pairs": split.two_core_pairs,
        "training_user_count": final_mappings.n_users,
        "training_item_count": final_mappings.n_items,
        "train_unique_pair_count": len(full_pairs),
        "catalog_size": catalog_size,
        "catalog_coverage": catalog_coverage,
        "embedding_dim": selected_dim,
        "optimizer": "adam",
        "learning_rate": args.learning_rate,
        "regularization": args.regularization,
        "negative_sampling": NEGATIVE_POLICY,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "selected_best_epoch": selected_epoch,
        "final_epochs": final_summary["epochs_run"],
        "final_train_loss": final_summary["history"][-1]["train_loss"] if final_summary["history"] else None,
        "tuning_duration_sec": round(tuning_seconds, 3),
        "final_retrain_duration_sec": round(retrain_seconds, 3),
        "total_runtime_sec": round(time.perf_counter() - started, 3),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "parameter_count": counts,
        "user_mapping_checksum": user_checksum,
        "item_mapping_checksum": item_checksum,
        "git_commit": _git_commit(),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "pilot": bool(args.pilot),
        "note": (
            "External recsys-eval-v1 test was not used for hyperparameter selection. "
            "Historical Parquet 2-core (22,634 / 12,105 / 57,127) is not this population."
        ),
    }
    tuning_payload = {
        "protocol": TUNE_PROTOCOL,
        "selection_metric": SELECTION_METRIC,
        "external_test_unused": True,
        "validation_users": len(tune.validation_users),
        "inner_train_pairs": len(inner_pairs),
        "inner_train_only_users": tune.inner_train_only_users,
        "candidates": [
            {
                "embedding_dim": run["embedding_dim"],
                "best_epoch": run["summary"]["best_epoch"],
                "best_validation_ndcg@10": run["summary"]["best_validation_ndcg@10"],
                "best_validation_loss": run["summary"]["best_validation_loss"],
                "epochs_run": run["summary"]["epochs_run"],
                "history": run["summary"]["history"],
            }
            for run in runs
        ],
        "selected_embedding_dim": selected_dim,
        "selected_best_epoch": selected_epoch,
    }
    history_payload = {
        "tuning": tuning_payload,
        "final": final_summary["history"],
        "note": "Selected by validation NDCG@10. External test unused.",
    }

    staging = out_dir.parent / f".{out_dir.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    save_cf_bundle(
        staging,
        model=model.cpu(),
        user_ids=final_mappings.user_ids,
        product_ids=final_mappings.product_ids,
        config=config,
        manifest=artifact_manifest,
        history=history_payload,
        tuning=tuning_payload,
    )
    artifact_manifest["model_checksum"] = sha256_file(staging / "model_state.pt")
    (staging / "manifest.json").write_text(json.dumps(artifact_manifest, indent=2) + "\n", encoding="utf-8")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    staging.rename(out_dir)

    if args.register_artifact and not args.pilot:
        reset_engine()
        factory = get_session_factory()
        with factory() as session:
            upsert_artifact_version(
                session,
                artifact_id=f"cf:{args.model_version}",
                artifact_type="collaborative_filtering",
                version=args.model_version,
                dataset_version=DATASET_VERSION_FULL,
                embedding_model_name=None,
                embedding_dim=selected_dim,
                metric="bpr",
                path=str(out_dir),
                metadata={
                    "model_type": MODEL_TYPE,
                    "recsys_eval_version": EVALUATION_VERSION,
                    "training_user_count": final_mappings.n_users,
                    "training_item_count": final_mappings.n_items,
                    "train_unique_pair_count": len(full_pairs),
                },
            )
            session.commit()

    print(
        json.dumps(
            {
                "model_version": args.model_version,
                "selected_dim": selected_dim,
                "selected_best_epoch": selected_epoch,
                "train_users": final_mappings.n_users,
                "train_items": final_mappings.n_items,
                "train_pairs": len(full_pairs),
                "catalog_coverage": catalog_coverage,
                "parameter_count": counts,
                "device": str(device),
                "path": str(out_dir),
                "pilot": bool(args.pilot),
                "external_test_unused_for_tuning": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
