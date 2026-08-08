import json
from pathlib import Path

import pytest

_SCHEMA_PATH = Path(__file__).parent.parent / 'models' / 'schema.json'


@pytest.fixture
def full_payload():
    """A payload with every schema field populated with a type-appropriate dummy value."""
    schema = json.loads(_SCHEMA_PATH.read_text())
    payload = {}
    for col, dtype in schema.items():
        if col in ('isFraud', 'TransactionID'):
            continue
        if dtype == 'float64':
            payload[col] = 100.0
        elif dtype == 'int64':
            payload[col] = 1
        else:
            payload[col] = 'W'
    return payload
