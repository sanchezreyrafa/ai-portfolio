import json
from pathlib import Path
from pydantic import BaseModel, create_model
from typing import Optional

_SCHEMA_PATH = Path(__file__).parent.parent.parent / 'models' / 'schema.json'

_REQUIRED_FIELDS = ('TransactionAmt', 'TransactionDT')


def create_request_model():
    with open(_SCHEMA_PATH) as f:
        schema = json.load(f)

    field_definitions = {}
    for col, dtype in schema.items():
        if col in ('isFraud', 'TransactionID'):
            continue
        if dtype == 'float64':
            field_type, default = float, None
        elif dtype == 'int64':
            field_type, default = int, None
        else:
            field_type, default = str, None

        if col in _REQUIRED_FIELDS:
            field_definitions[col] = (field_type, ...)
        else:
            field_definitions[col] = (Optional[field_type], default)

    return create_model('Transaction', **field_definitions)


Transaction = create_request_model()


class PredictResponse(BaseModel):
    is_fraud: bool
    probability: float