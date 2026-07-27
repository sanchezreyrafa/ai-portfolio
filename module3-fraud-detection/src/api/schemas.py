import json
from pydantic import BaseModel, create_model
from typing import Optional

_SCHEMA_PATH = '../models/schema.json'


def create_request_model():
    with open(_SCHEMA_PATH) as f:
        schema = json.load(f)

    field_definitions = {}
    for col, dtype in schema.items():
        if col in ('isFraud', 'TransactionID'):
            continue
        if dtype == 'float64':
            field_definitions[col] = (Optional[float], None)
        elif dtype == 'int64':
            field_definitions[col] = (Optional[int], None)
        else:
            field_definitions[col] = (Optional[str], None)

    return create_model('Transaction', **field_definitions)


Transaction = create_request_model()


class PredictResponse(BaseModel):
    is_fraud: bool
    probability: float