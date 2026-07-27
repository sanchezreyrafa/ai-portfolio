import sys
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocessing import (
    encode_boolean_known_features,
    encode_simple_boolean_features,
    encode_card_features,
    encode_match_status,
    encode_tf_features,
    extract_browser,
    extract_os,
)

THRESHOLD = 0.3


def run_inference(data: dict, model: XGBClassifier) -> dict:
    df = pd.DataFrame([data])

    # --- preprocessing (mirrors 02_preprocessing.ipynb) ---
    # TODO: apply transformations

    proba = float(model.predict_proba(df)[:, 1][0])
    return {"is_fraud": proba >= THRESHOLD, "probability": proba}