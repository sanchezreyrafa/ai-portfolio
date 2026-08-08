import sys
import json
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
    derive_has_identity,
    extract_browser,
    extract_os,
)

THRESHOLD = 0.3

_MODELS_DIR = Path(__file__).parent.parent.parent / 'models'

with open(_MODELS_DIR / 'cols_to_drop.json') as f:
    _COLS_TO_DROP: list[str] = json.load(f)

with open(_MODELS_DIR / 'schema.json') as f:
    _NUMERIC_COLUMNS = [col for col, dtype in json.load(f).items() if dtype in ('float64', 'int64')]


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # A single-row request DataFrame infers 'object' dtype for any numeric field
    # that's null, since a lone None can't be inferred as float/int. Coerce these
    # explicitly so they become proper NaN-bearing numeric columns instead of
    # falling into the object dtype bucket.
    for col in _NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create derived features before their source columns are dropped
    if 'id_01' in df.columns and 'has_identity' not in df.columns:
        df = derive_has_identity(df)

    if 'TransactionDT' in df.columns and 'TransactionDTDays' not in df.columns:
        df['TransactionDTDays'] = df['TransactionDT'] / 86400

    if 'DeviceInfo' in df.columns and 'DeviceOS' not in df.columns:
        df['DeviceOS'] = df['DeviceInfo'].apply(extract_os)

    if 'id_31' in df.columns and 'DeviceBrowser' not in df.columns:
        df['DeviceBrowser'] = df['id_31'].apply(extract_browser)

    # Drop all columns tracked in cols_to_drop.json
    df = df.drop(columns=[c for c in _COLS_TO_DROP if c in df.columns])

    # Encodings — same order as 02_preprocessing.ipynb
    df = encode_boolean_known_features(df, ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9'], 'T')
    df = encode_card_features(df, 'card4', ['visa', 'mastercard', 'american express', 'discover'], False)
    df = encode_card_features(df, 'card6', ['debit', 'credit'], False)
    df = encode_card_features(df, 'ProductCD', ['W', 'H', 'C', 'S', 'R'], False)
    df = encode_card_features(df, 'DeviceType', ['desktop', 'mobile'], False)
    df = encode_card_features(df, 'DeviceBrowser', ['safari', 'firefox', 'ie', 'chrome'], False)
    df = encode_card_features(df, 'DeviceOS', ['windows', 'android', 'ios', 'mac'], False)
    df = encode_simple_boolean_features(df, ['id_12', 'id_28', 'id_16', 'id_29'], 'Found')
    df = encode_boolean_known_features(df, ['id_15'], 'Found')
    df = encode_match_status(df)
    df = encode_tf_features(df, ['id_35', 'id_36', 'id_37', 'id_38'])
    df = encode_card_features(df, 'M4', ['M0', 'M1', 'M2'], False)

    return df


def run_inference(data: dict, model: XGBClassifier) -> dict:
    df = pd.DataFrame([data])
    df = _preprocess(df)
    proba = float(model.predict_proba(df)[:, 1][0])
    return {'is_fraud': proba >= THRESHOLD, 'probability': proba}
