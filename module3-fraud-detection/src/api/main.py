from fastapi import FastAPI
from xgboost import XGBClassifier
from .schemas import Transaction, PredictResponse
from .pipeline import run_inference

#Get the trained model when starting the app
_MODEL_PATH = '../models/model.json'
model = XGBClassifier()
model.load_model(_MODEL_PATH)

app = FastAPI()

@app.post("/predict", response_model=PredictResponse)
def predict(data: Transaction):
    return run_inference(data.model_dump(), model)