import pickle
from typing import Optional, Union
from fastapi import FastAPI
from pydantic import BaseModel
from preprocess import to_record

with open("model.bin", "rb") as f:
    bundle = pickle.load(f)
dv, model, keep_sets = bundle["dv"], bundle["model"], bundle["keep_sets"]

app = FastAPI(title="Supreme Court Case Outcome Prediction")
Val = Optional[Union[int, float, str]]


class Case(BaseModel):
    term: int
    naturalCourt: Val = None
    chief: Val = None
    petitioner: Val = None
    respondent: Val = None
    jurisdiction: Val = None
    adminAction: Val = None
    threeJudgeFdc: Val = None
    caseOrigin: Val = None
    caseSource: Val = None
    certReason: Val = None
    lcDisagreement: Val = None
    lcDisposition: Val = None
    lcDispositionDirection: Val = None


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"service": "Supreme Court Case Outcome Prediction",
            "docs": "/docs", "predict": "POST /predict"}


@app.post("/predict")
def predict(case: Case):
    rec = to_record(case.model_dump(), keep_sets)
    prob = float(model.predict_proba(dv.transform([rec]))[0, 1])
    return {"petitioner_win_probability": round(prob, 4),
            "prediction": "petitioner wins" if prob >= 0.5 else "petitioner loses"}