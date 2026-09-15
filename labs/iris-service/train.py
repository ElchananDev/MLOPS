"""אימון מודל סיווג על מאגר Iris, ושמירת תוצרים ניתנים לשחזור.

התהליך:
1. טעינת נתונים והפרדת סט בדיקה שלא משתתף בבחירת ההגדרות.
2. מדידת Baseline על אותן חלוקות אימון.
3. חיפוש הגדרות עם Cross-Validation.
4. הערכה סופית אחת על סט הבדיקה.
5. שמירת המודל, המדדים והמטא-דאטה לתיקיית artifacts.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]

SPLIT_SEED = 11
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def build_search(cv: StratifiedKFold) -> GridSearchCV:
    """Pipeline של עיבוד + מודל, עטוף בחיפוש הגדרות."""
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid={"model__C": [0.1, 1.0, 10.0]},
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        refit=True,
    )


def main() -> None:
    dataset = load_iris()

    X = pd.DataFrame(dataset.data, columns=FEATURES)
    y = dataset.target

    # סט הבדיקה נשמר בצד לפני כל החלטה על הגדרות.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SPLIT_SEED,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)

    baseline_scores = cross_val_score(
        DummyClassifier(strategy="most_frequent"),
        X_train,
        y_train,
        scoring="f1_macro",
        cv=cv,
    )

    search = build_search(cv)
    search.fit(X_train, y_train)
    model = search.best_estimator_

    # הערכה סופית, אחרי שכל ההגדרות כבר נבחרו.
    predictions = model.predict(X_test)
    test_f1 = f1_score(y_test, predictions, average="macro")

    dataset_hash = hashlib.sha256(
        dataset.data.tobytes() + dataset.target.tobytes()
    ).hexdigest()

    metadata = {
        "model_version": uuid4().hex,
        "features": FEATURES,
        "target_names": dataset.target_names.tolist(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "dataset_sha256": dataset_hash,
        "split_seed": SPLIT_SEED,
        "best_parameters": search.best_params_,
    }

    metrics = {
        "baseline_cv_macro_f1": float(baseline_scores.mean()),
        "selected_cv_macro_f1": float(search.best_score_),
        "test_macro_f1": float(test_f1),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)

    joblib.dump(
        {"pipeline": model, "metadata": metadata},
        ARTIFACT_DIR / "model.joblib",
    )
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (ARTIFACT_DIR / "classification_report.txt").write_text(
        classification_report(
            y_test, predictions, target_names=dataset.target_names
        ),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
