import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    derive_has_identity,
    encode_boolean_known_features,
    encode_card_features,
    encode_match_status,
    encode_simple_boolean_features,
    encode_tf_features,
    extract_browser,
    extract_os,
    high_corr_list,
    split_dataframe,
)


class TestHighCorrList:
    def test_flags_the_second_column_of_a_correlated_pair(self):
        df = pd.DataFrame({
            'V1': [1, 2, 3, 4, 5],
            'V2': [1, 2, 3, 4, 5],  # perfectly correlated with V1
            'V3': [5, 1, 4, 2, 3],  # uncorrelated
        })
        result = high_corr_list({'V1-V3': [1, 2, 3]}, df)
        assert result == {'V2'}

    def test_no_correlation_above_threshold_returns_empty_set(self):
        df = pd.DataFrame({
            'V1': [1, 2, 3, 4, 5],
            'V2': [5, 1, 4, 2, 3],
        })
        result = high_corr_list({'V1-V2': [1, 2]}, df)
        assert result == set()


class TestEncodeBooleanKnownFeatures:
    def test_splits_into_known_and_value_columns_and_drops_source(self):
        df = pd.DataFrame({'M1': ['T', 'F', None]})
        result = encode_boolean_known_features(df, ['M1'], 'T')

        assert 'M1' not in result.columns
        assert list(result['M1_known']) == [True, True, False]
        assert list(result['M1_value']) == [True, False, False]


class TestEncodeSimpleBooleanFeatures:
    def test_maps_true_value_and_drops_source(self):
        df = pd.DataFrame({'id_12': ['Found', 'NotFound', None]})
        result = encode_simple_boolean_features(df, ['id_12'], 'Found')

        assert 'id_12' not in result.columns
        assert list(result['id_12_value']) == [True, False, False]


class TestSplitDataframe:
    def test_splits_chronologically_into_60_20_20(self):
        df = pd.DataFrame({
            'TransactionDTDays': list(range(10)),
            'value': list(range(10)),
        })
        df_train, df_val, df_test = split_dataframe(df)

        assert len(df_train) == 6
        assert len(df_val) == 2
        assert len(df_test) == 2
        # chronological, non-overlapping, in order
        assert df_train['TransactionDTDays'].max() < df_val['TransactionDTDays'].min()
        assert df_val['TransactionDTDays'].max() < df_test['TransactionDTDays'].min()

    def test_input_out_of_order_is_sorted_before_splitting(self):
        df = pd.DataFrame({'TransactionDTDays': [3, 1, 4, 0, 2]})
        df_train, _, _ = split_dataframe(df)
        assert list(df_train['TransactionDTDays']) == sorted(df_train['TransactionDTDays'])


class TestEncodeCardFeatures:
    def test_one_hot_encodes_known_categories_and_buckets_the_rest_as_other(self):
        df = pd.DataFrame({'card4': ['visa', 'mastercard', 'unknown_brand', None]})
        result = encode_card_features(df, 'card4', ['visa', 'mastercard'], drop_first=False)

        assert 'card4' not in result.columns
        assert list(result['card4_visa']) == [True, False, False, False]
        assert list(result['card4_mastercard']) == [False, True, False, False]
        assert list(result['card4_other']) == [False, False, True, True]

    def test_drop_first_removes_the_first_category_column(self):
        df = pd.DataFrame({'card6': ['debit', 'credit']})
        result = encode_card_features(df, 'card6', ['debit', 'credit'], drop_first=True)

        assert 'card6_debit' not in result.columns
        assert 'card6_credit' in result.columns


class TestExtractBrowser:
    @pytest.mark.parametrize('raw,expected', [
        ('Safari 12.0', 'safari'),
        ('Mobile Safari', 'safari'),
        ('Firefox 60.0', 'firefox'),
        ('IE 11.0 for desktop', 'ie'),
        ('chrome 68.0 for android', 'chrome'),
        ('Samsung/SM-G531H', 'other'),
    ])
    def test_extracts_known_browsers_case_insensitively(self, raw, expected):
        assert extract_browser(raw) == expected

    def test_missing_value_returns_unknown(self):
        assert extract_browser(np.nan) == 'unknown'


class TestExtractOs:
    @pytest.mark.parametrize('raw,expected', [
        ('Windows 10', 'windows'),
        ('SM-G531H Build/MMB29T', 'android'),
        ('ALE-L23 Build/HuaweiALE-L23', 'android'),
        ('moto g(6) play Build/OPP27.61-38', 'android'),
        ('iOS 12.1.2', 'ios'),
        ('iPhone', 'ios'),
        ('MacOS', 'mac'),
        ('Linux', 'other'),
    ])
    def test_extracts_known_os_case_insensitively(self, raw, expected):
        assert extract_os(raw) == expected

    def test_missing_value_returns_unknown(self):
        assert extract_os(None) == 'unknown'


class TestEncodeTfFeatures:
    def test_maps_t_f_to_booleans_and_preserves_na(self):
        df = pd.DataFrame({'id_35': ['T', 'F', None]})
        result = encode_tf_features(df, ['id_35'])

        assert result['id_35'].tolist()[:2] == [True, False]
        assert pd.isna(result['id_35'].tolist()[2])


class TestEncodeMatchStatus:
    def test_extracts_numeric_status_from_prefixed_string(self):
        df = pd.DataFrame({'id_34': ['match_status:2', 'match_status:-1', None]})
        result = encode_match_status(df)

        assert result['id_34'].tolist()[0] == 2.0
        assert result['id_34'].tolist()[1] == -1.0
        assert pd.isna(result['id_34'].tolist()[2])


class TestDeriveHasIdentity:
    def test_flags_rows_with_a_non_null_source_column(self):
        df = pd.DataFrame({'id_01': [0.0, np.nan, -5.0, np.nan]})
        result = derive_has_identity(df)

        assert list(result['has_identity']) == [1, 0, 1, 0]

    def test_uses_the_given_source_column(self):
        df = pd.DataFrame({'other_col': [1.0, np.nan]})
        result = derive_has_identity(df, source_col='other_col')

        assert list(result['has_identity']) == [1, 0]
