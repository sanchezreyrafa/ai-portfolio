import pandas as pd


def high_corr_list(feature_group: dict, df: pd.DataFrame) -> set :
    cols_to_drop = set()
    for group_name, indices in feature_group.items():
        cols = [f'V{i}' for i in indices]
        corr_matrix = df[cols].corr()
        # Get values with high correlation (>0.9) excluding diagonal

        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                val = corr_matrix.iloc[i, j]
                if abs(float(val)) > 0.9:
                    cols_to_drop.add(cols[j])
    return cols_to_drop

def encode_boolean_known_features(df, features, true_value):
    for feat in features :
        df[f'{feat}_known'] = df[f'{feat}'].notna()
        df[f'{feat}_value'] = df[f'{feat}'] == true_value
        df.drop(columns=[f'{feat}'], inplace=True)
    return df

def encode_simple_boolean_features(df, features, true_value):
    for feat in features :
        df[f'{feat}_value'] = df[f'{feat}'] == true_value
        df.drop(columns=[f'{feat}'], inplace=True)
    return df

def encode_m_features(df: pd.DataFrame) -> pd.DataFrame:
    for i in range(1,10) :
        if i != 4 :
            df[f'M{i}_known'] = df[f'M{i}'].notna()
            df[f'M{i}_value'] = df[f'M{i}'] == "T"
            df.drop(columns=[f'M{i}'], inplace=True)
    return df

def split_dataframe(df: pd.DataFrame):
    df.sort_values(by=['TransactionDTDays'], inplace=True)
    train_idx = int(len(df) * 0.60)
    val_idx = int(len(df) * 0.80)

    df_train = df.iloc[:train_idx]
    df_val = df.iloc[train_idx:val_idx]
    df_test = df.iloc[val_idx:]
    return df_train, df_val, df_test


def encode_card_features(df, column, valid_categories, drop_first) -> pd.DataFrame:
    df[f'{column}_other'] = ~df[column].isin(valid_categories)

    for category in valid_categories:
        df[f'{column}_{category}'] = df[column] == category

    if drop_first:
        df.drop(columns=[f'{column}_{valid_categories[0]}'], inplace=True)
    df.drop(columns=[column], inplace=True)
    return df

def extract_browser(browser_info):
    if pd.isna(browser_info):
        return 'unknown'
    browser_info = browser_info.lower()
    if 'safari' in browser_info:
        return 'safari'
    if 'firefox' in browser_info:
        return 'firefox'
    if 'ie' in browser_info or 'iexplorer' in browser_info or 'explorer' in browser_info:
        return 'ie'
    if 'chrome' in browser_info:
        return 'chrome'
    return 'other'

def encode_tf_features(df, features):
    for feat in features:
        df[feat] = df[feat].map({'T': True, 'F': False})
    return df

def encode_match_status(df, column='id_34'):
    df[column] = pd.to_numeric(df[column].str.split(':').str[-1], errors='coerce')
    return df

def extract_os(device_info):
    if pd.isna(device_info):
        return 'unknown'
    device_info = device_info.lower()
    if 'windows' in device_info:
        return 'windows'
    if 'android' in device_info or 'sm-' in device_info or 'ale-l23' in device_info or 'moto g' in device_info:
        return 'android'
    if 'ios' in device_info or 'iphone' in device_info or 'ipad' in device_info:
        return 'ios'
    if 'mac' in device_info:
        return 'mac'
    return 'other'