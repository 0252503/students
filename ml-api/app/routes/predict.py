from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

MODEL_PATH = Path(__file__).resolve().parents[2] / "model" / "model.pkl"
PASS_THRESHOLD = 0.5

pipeline = joblib.load(MODEL_PATH)

router = APIRouter()


class PredictRequest(BaseModel):
    gender: Literal["female", "male"]
    ethnicity: Literal["group A", "group B", "group C", "group D", "group E"]
    parental_education: Literal[
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree",
    ]
    lunch: Literal["standard", "free/reduced"]
    test_prep: Literal["none", "completed"]


class PredictResponse(BaseModel):
    pass_math: int
    probability: float
    result: str


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    frame = pd.DataFrame([payload.model_dump()])
    probability = float(pipeline.predict_proba(frame)[0][1])
    passes = probability > PASS_THRESHOLD
    return PredictResponse(
        pass_math=1 if passes else 0,
        probability=round(probability, 4),
        result="aprueba" if passes else "reprueba",
    )
