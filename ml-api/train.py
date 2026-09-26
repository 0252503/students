"""Entrena un Random Forest para predecir pass_math y guarda el pipeline."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "students.csv"
MODEL_PATH = ROOT / "model" / "model.pkl"

FEATURES = [
    "gender",
    "ethnicity",
    "parental_education",
    "lunch",
    "test_prep",
]
TARGET = "pass_math"


def build_pipeline() -> Pipeline:
    encoder = ColumnTransformer(
        transformers=[
            (
                "cats",
                OneHotEncoder(handle_unknown="ignore"),
                FEATURES,
            )
        ]
    )
    return Pipeline(
        steps=[
            ("encoder", encoder),
            (
                "model",
                RandomForestClassifier(n_estimators=100, random_state=42),
            ),
        ]
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"Accuracy en holdout: {accuracy:.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()
