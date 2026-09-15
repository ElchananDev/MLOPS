"""רישום תוצאות הריצה ב-MLflow, כדי שניסויים יהיו ניתנים להשוואה."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
metadata = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-learning")

with mlflow.start_run():
    mlflow.log_params(
        {
            "split_seed": metadata["split_seed"],
            "model_version": metadata["model_version"],
            "model_C": metadata["best_parameters"]["model__C"],
            "sklearn_version": metadata["sklearn_version"],
        }
    )
    mlflow.set_tag("dataset_sha256", metadata["dataset_sha256"])
    mlflow.log_metrics(metrics)
    mlflow.log_artifacts(str(ARTIFACTS))

print("run logged; open the UI with: mlflow ui --backend-store-uri sqlite:///mlflow.db")
