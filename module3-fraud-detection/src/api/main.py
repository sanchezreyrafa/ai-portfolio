from pathlib import Path
from fastapi import FastAPI
from xgboost import XGBClassifier
from .schemas import Transaction, PredictResponse
from .pipeline import run_inference

_MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'model.json'

model = XGBClassifier()
model.load_model(str(_MODEL_PATH))

app = FastAPI()

@app.post("/predict", response_model=PredictResponse)
def predict(data: Transaction):
    return run_inference(data.model_dump(), model)