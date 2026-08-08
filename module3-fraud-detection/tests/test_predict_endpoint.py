import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestPredictEndpoint:
    def test_full_payload_returns_a_valid_prediction(self, full_payload):
        response = client.post('/predict', json=full_payload)

        assert response.status_code == 200
        body = response.json()
        assert isinstance(body['is_fraud'], bool)
        assert 0.0 <= body['probability'] <= 1.0

    def test_is_fraud_matches_the_030_threshold(self, full_payload):
        response = client.post('/predict', json=full_payload)
        body = response.json()

        assert body['is_fraud'] == (body['probability'] >= 0.3)

    def test_empty_payload_is_rejected_with_422(self):
        response = client.post('/predict', json={})

        assert response.status_code == 422

    @pytest.mark.parametrize('required_field', ['TransactionAmt', 'TransactionDT'])
    def test_missing_required_field_is_rejected_with_422(self, full_payload, required_field):
        del full_payload[required_field]

        response = client.post('/predict', json=full_payload)

        assert response.status_code == 422

    def test_wrong_field_type_is_rejected_with_422(self, full_payload):
        full_payload['TransactionAmt'] = 'not-a-number'

        response = client.post('/predict', json=full_payload)

        assert response.status_code == 422

    @pytest.mark.parametrize('dropped_field', ['id_01', 'DeviceInfo', 'id_31'])
    def test_missing_source_column_for_a_derived_feature_does_not_crash(self, full_payload, dropped_field):
        """These fields feed has_identity / DeviceOS / DeviceBrowser.
        A client omitting them must not break inference."""
        del full_payload[dropped_field]

        response = client.post('/predict', json=full_payload)

        assert response.status_code == 200

    def test_only_required_fields_present_does_not_crash(self):
        """The realistic case: most transactions have no identity data at all (~76% per EDA)."""
        response = client.post('/predict', json={'TransactionAmt': 100.0, 'TransactionDT': 86400})

        assert response.status_code == 200