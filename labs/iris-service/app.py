"""שירות חיזוי עם חוזה קלט/פלט מפורש, בדיקות בריאות ולוג שניתן לחבר לתוצאה בפועל."""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "model.joblib"

logger = logging.getLogger("iris-service")
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")


def log_event(**fields: Any) -> None:
    """שורת לוג אחת בפורמט JSON, כדי שאפשר יהיה לחפש ולצבור אותה."""
    logger.info(json.dumps(fields, ensure_ascii=False))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # המודל נטען פעם אחת בעליית השירות, ולא בכל בקשה.
    app.state.bundle = joblib.load(ARTIFACT)
    app.state.ready = True
    log_event(
        event="model_loaded",
        model_version=app.state.bundle["metadata"]["model_version"],
    )
    yield
    app.state.ready = False


app = FastAPI(
    title="Iris prediction service",
    version="1.0.0",
    lifespan=lifespan,
)


class Flower(BaseModel):
    """חוזה הקלט. ערך מחוץ לטווח או שדה לא מוכר נדחים לפני שהמודל רץ."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    sepal_length_cm: float = Field(gt=0, le=20)
    sepal_width_cm: float = Field(gt=0, le=20)
    petal_length_cm: float = Field(gt=0, le=20)
    petal_width_cm: float = Field(gt=0, le=20)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: האם התהליך חי."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    """Readiness: האם אפשר לשלוח לכאן תעבורה."""
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="model is not loaded")
    return {"status": "ready"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return app.state.bundle["metadata"]


@app.post("/predict")
def predict(flower: Flower) -> dict[str, Any]:
    started = time.perf_counter()

    bundle = app.state.bundle
    model = bundle["pipeline"]
    meta = bundle["metadata"]

    # סדר העמודות נלקח מהמטא-דאטה, לא מסדר השדות בבקשה.
    row = pd.DataFrame([flower.model_dump()], columns=meta["features"])

    class_id = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]

    class_probabilities = {
        meta["target_names"][int(label)]: float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }

    prediction_id = uuid4().hex
    latency_ms = (time.perf_counter() - started) * 1000

    # המזהה הזה הוא מה שיאפשר בהמשך לחבר תחזית לתוצאה האמיתית.
    log_event(
        event="prediction",
        prediction_id=prediction_id,
        model_version=meta["model_version"],
        prediction=meta["target_names"][class_id],
        confidence=round(max(class_probabilities.values()), 4),
        latency_ms=round(latency_ms, 2),
    )

    return {
        "prediction_id": prediction_id,
        "prediction": meta["target_names"][class_id],
        "probabilities": class_probabilities,
        "model_version": meta["model_version"],
    }
