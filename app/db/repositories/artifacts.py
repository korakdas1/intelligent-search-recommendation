"""Register derived ML artifacts in the existing artifact_versions table."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.models.artifact import ArtifactVersion


def upsert_artifact_version(
    session: Session,
    *,
    artifact_id: str,
    artifact_type: str,
    version: str,
    dataset_version: str | None,
    embedding_model_name: str | None,
    embedding_dim: int | None,
    metric: str | None,
    path: str | None,
    metadata: dict[str, Any] | None = None,
) -> ArtifactVersion:
    row = session.get(ArtifactVersion, artifact_id)
    if row is None:
        row = ArtifactVersion(artifact_id=artifact_id)
        session.add(row)
    row.artifact_type = artifact_type
    row.version = version
    row.dataset_version = dataset_version
    row.embedding_model_name = embedding_model_name
    row.embedding_dim = embedding_dim
    row.metric = metric
    row.path = path
    row.metadata_ = metadata or {}
    return row
