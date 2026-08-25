#!/usr/bin/env python3
"""Train a small RankNet ranker from a synthetic LTR dataset. Validation selects the checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from app.core.events import DATASET_VERSION_FULL
from app.db.repositories.artifacts import upsert_artifact_version
from app.db.session import get_session_factory, reset_engine
from app.embeddings.checksums import sha256_file
from app.ranking.artifacts import save_ranker_bundle
from app.ranking.constants import (
    DATASET_VERSION,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_NEGATIVES_PER_POSITIVE,
    DEFAULT_PATIENCE,
    MODEL_VERSION,
)
from app.ranking.feature_schema import FEATURE_COUNT, FEATURE_NAMES, FEATURE_NAMES_SHA256, FEATURE_VERSION
from app.ranking.preprocessing import fit_scaler
from app.ranking.train import resolve_device, train_ranknet


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=f"artifacts/ltr/{DATASET_VERSION}")
    parser.add_argument("--model-version", default=MODEL_VERSION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--negatives-per-positive", type=int, default=DEFAULT_NEGATIVES_PER_POSITIVE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-architecture-compare", action="store_true")
    parser.add_argument("--register-artifact", action="store_true")
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset)
    parquet_path = dataset_dir / "candidates.parquet"
    if not parquet_path.is_file():
        print(f"missing dataset parquet: {parquet_path}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir or f"artifacts/models/{args.model_version}")
    if out_dir.exists() and not args.force:
        print(f"refusing to overwrite {out_dir}; pass --force", file=sys.stderr)
        return 2

    frame = pd.read_parquet(parquet_path)
    for leaked in ("relevance", "source_product_id", "split", "candidate_position"):
        if leaked in FEATURE_NAMES:
            raise RuntimeError(f"{leaked} must not be a rank-features-v1 column")
    train_frame = frame[frame["split"] == "train"]
    if train_frame.empty:
        print("empty train split", file=sys.stderr)
        return 1
    scaler = fit_scaler(train_frame.loc[:, list(FEATURE_NAMES)].to_numpy())
    architectures = [((64, 32), "primary")]
    if not args.skip_architecture_compare:
        architectures = [((32,), "small"), ((64, 32), "primary")]

    device = str(resolve_device(args.device))
    started = time.perf_counter()
    runs = []
    winner = None
    for hidden, name in architectures:
        model, summary = train_ranknet(
            frame,
            scaler,
            hidden_sizes=hidden,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            device=device,
            negatives_per_positive=args.negatives_per_positive,
        )
        val_loss = summary["best_validation_loss"]
        runs.append({"name": name, "hidden_sizes": list(hidden), "summary": summary})
        print(
            f"{name} hidden={list(hidden)} best_epoch={summary['best_epoch']} "
            f"val_loss={val_loss} train_pairs={summary['train_pair_count']}",
            flush=True,
        )
        if winner is None or (val_loss is not None and val_loss < winner["summary"]["best_validation_loss"]):
            winner = {"name": name, "model": model, "summary": summary, "hidden": hidden}

    duration = time.perf_counter() - started
    assert winner is not None
    model = winner["model"]
    summary = winner["summary"]
    config = {
        "model_version": args.model_version,
        "model_type": "ranknet_mlp",
        "feature_version": FEATURE_VERSION,
        "feature_names_sha256": FEATURE_NAMES_SHA256,
        "input_dim": FEATURE_COUNT,
        "hidden_sizes": list(winner["hidden"]),
        "dropout": args.dropout,
        "loss": "ranknet_bce_with_logits",
        "optimizer": "adam",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "dataset_version": DATASET_VERSION,
        "negatives_per_positive": args.negatives_per_positive,
        "architecture_selected_by": "validation_ranknet_loss",
        "architecture_name": winner["name"],
    }
    dataset_manifest = {}
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.is_file():
        dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    tmp = out_dir.parent / f".{out_dir.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    history = {
        "runs": [
            {
                "name": run["name"],
                "hidden_sizes": run["hidden_sizes"],
                "best_epoch": run["summary"]["best_epoch"],
                "best_validation_loss": run["summary"]["best_validation_loss"],
                "epochs_run": run["summary"]["epochs_run"],
                "history": run["summary"]["history"],
            }
            for run in runs
        ],
        "selected": winner["name"],
        "note": "Selected by lowest validation RankNet loss. Test was not used.",
    }
    artifact_manifest = {
        "model_version": args.model_version,
        "model_type": "ranknet_mlp",
        "feature_version": FEATURE_VERSION,
        "feature_names_sha256": FEATURE_NAMES_SHA256,
        "dataset_version": dataset_manifest.get("dataset_version", DATASET_VERSION),
        "synthetic_label_version": DATASET_VERSION,
        "catalog_dataset_version": DATASET_VERSION_FULL,
        "split_seed": args.seed,
        "train_query_count": dataset_manifest.get("train_query_count"),
        "validation_query_count": dataset_manifest.get("validation_query_count"),
        "test_query_count": dataset_manifest.get("test_query_count"),
        "pair_count": summary["train_pair_count"],
        "architecture": list(winner["hidden"]),
        "dropout": args.dropout,
        "optimizer": "adam",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs_run": summary["epochs_run"],
        "best_epoch": summary["best_epoch"],
        "preprocessing": "train_only_standardize",
        "zero_variance_indices": list(scaler.zero_variance_indices),
        "git_commit": _git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "training_duration_sec": round(duration, 3),
        "parameter_count": model.parameter_count(),
        "architecture_compare": not args.skip_architecture_compare,
        "selected_architecture": winner["name"],
    }
    save_ranker_bundle(
        tmp,
        model=model,
        scaler=scaler,
        config=config,
        manifest=artifact_manifest,
        history=history,
    )
    artifact_manifest["model_checksum"] = sha256_file(tmp / "model_state.pt")
    artifact_manifest["scaler_checksum"] = sha256_file(tmp / "scaler.npz")
    (tmp / "manifest.json").write_text(json.dumps(artifact_manifest, indent=2) + "\n", encoding="utf-8")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)

    if args.register_artifact:
        reset_engine()
        factory = get_session_factory()
        with factory() as session:
            upsert_artifact_version(
                session,
                artifact_id=f"ranker:{args.model_version}",
                artifact_type="ranker",
                version=args.model_version,
                dataset_version=DATASET_VERSION_FULL,
                embedding_model_name=None,
                embedding_dim=FEATURE_COUNT,
                metric="ranknet",
                path=str(out_dir),
                metadata={
                    "feature_version": FEATURE_VERSION,
                    "parameter_count": model.parameter_count(),
                },
            )
            session.commit()

    print(
        json.dumps(
            {
                "model_version": args.model_version,
                "selected": winner["name"],
                "hidden_sizes": list(winner["hidden"]),
                "best_epoch": summary["best_epoch"],
                "epochs_run": summary["epochs_run"],
                "train_pair_count": summary["train_pair_count"],
                "best_validation_loss": summary["best_validation_loss"],
                "parameter_count": model.parameter_count(),
                "device": device,
                "duration_sec": round(duration, 3),
                "path": str(out_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
