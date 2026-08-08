import pandas as pd

from src.api.main import model
from src.api.pipeline import _preprocess


class TestFeatureParity:
    """Guards against the class of bug we just fixed: a column that _preprocess()
    still emits (or drops) but the trained model doesn't expect, silently corrupting
    every prediction instead of failing loudly."""

    def test_preprocess_output_matches_the_trained_model_features(self, full_payload):
        df = pd.DataFrame([full_payload])
        processed = _preprocess(df)

        assert set(processed.columns) == set(model.get_booster().feature_names)