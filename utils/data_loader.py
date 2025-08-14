import pickle
from collections import defaultdict

import pandas as pd
from rich import print
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from . import code, root_path

TARGET_FEATURE = ['hibp', 'majdysrh', 'myocisch']


def get_type(c, verbose=True):
    try:
        return code.get(c).get('type')
    except AttributeError:
        if c == 'NG':
            return 'categorical'
        try:
            return code.get(c.split('_')[0]).get('type')
        except AttributeError:
            if c != 'target' and verbose:
                print(f'{c} is not in codebook')
        return None


def get_code_object(c):
    return code.get(c.split('_')[0])


def change_column_data_type(df, verbose=True):
    for c in df.columns:
        t = get_type(c, verbose)
        if t == 'categorical':
            df[c] = df[c].astype('category')
        elif t == 'numeric':
            df[c] = pd.to_numeric(df[c], errors='coerce')
        elif t == 'text':
            df[c] = df[c].astype('str')
    return df


def apply_filter(df, filters):
    for f in filters:
        if f[1] == '=':
            df = df[df[f[0]] == f[2]]
        elif f[1] == '<':
            df = df[df[f[0]] < f[2]]
        elif f[1] == '>':
            df = df[df[f[0]] > f[2]]
    return df


def get_categorical_columns(df, exclude_target=True):
    categorical_features = df.select_dtypes(include=['category', 'object']).columns
    if exclude_target:
        categorical_features = list(set(categorical_features) - {'target'})
    return categorical_features


def load_scaler(scaler_id=None):
    if scaler_id is None:
        scaler_path = root_path / 'temp/scaler.pkl'
    else:
        scaler_path = root_path / 'results/scalers' / f'{scaler_id}.pkl'
    return ColumnScaler.load(scaler_path)


def scale_df(df, train, scaler_id=None):
    if scaler_id is None:
        scaler_path = root_path / 'temp/scaler.pkl'
    else:
        scaler_path = root_path / 'results/scalers' / f'{scaler_id}.pkl'
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    if train and not scaler_path.exists():
        scaler = ColumnScaler()
        scaler.fit(df)
        scaler.save(scaler_path)
        return scaler.transform(df)
    else:
        scaler = ColumnScaler.load(scaler_path)
        return scaler.transform(df)


def threshold_time_features(df, time_list):
    time_drop_cols = defaultdict(list)
    for c in df.columns:
        if c == 'target' or c == 'NG' or c in ['hibp', 'majdysrh', 'myocisch', 'id']:
            continue
        code_object = get_code_object(c)
        assert code_object is not None, f'{c} is not in codebook'
        assert code_object.get('time') in [0, 1, 2, 3], f'{c} time should be in [0, 1, 2, 3], got {code.get(c)}'
        code_time = code_object.get('time')
        if code_time not in time_list:
            time_drop_cols[code_time].append(c)
    for k, v in time_drop_cols.items():
        df = df.drop(columns=v, axis=1)
        # print(f"Dropped {len(v)} features with time == {k}, dropped features: {v}")
    return df


def preload_vo2_df(
    train, group, drop_columns, filters, fold, file_name, hold_out_folder, full_load=False, verbose=True
):
    df = pd.read_csv(root_path / file_name, low_memory=False)
    df.drop('date', inplace=True, axis=1, errors='ignore')
    df = change_column_data_type(df, verbose)

    if not full_load:
        with open(root_path / hold_out_folder / f'split_{fold}.pkl', 'rb') as f:
            hold_out = pickle.load(f)
            key = 'train' if train else 'test'
            df = df[df['id'].isin(hold_out[key])]

    cat_col = set(get_categorical_columns(df))
    cat_col.remove('id')
    df[list(cat_col)] = df[list(cat_col)].astype(int)
    ## Group filter
    assert group in ['NG', 'OG', 'NG+OG']
    normal_mask = df['NG'] == 1
    disease_mask = (df['hibp'] == 1) | (df['majdysrh'] == 1) | (df['myocisch'] == 1)
    abnormal_mask = ~normal_mask & ~disease_mask
    drop_columns.extend(['NG'])

    assert (normal_mask & disease_mask & abnormal_mask).sum() == 0, 'Overlapping mask'
    if group == 'NG':
        mask = normal_mask
    elif group == 'OG':
        mask = disease_mask | abnormal_mask
    elif group == 'NG+OG':
        mask = normal_mask | disease_mask | abnormal_mask
    df = df[mask]
    df.drop(columns=drop_columns, inplace=True, errors='ignore')

    df['target'] = df['vo2pkg']
    ## Custom filter
    df = apply_filter(df, filters)
    return df


def load_vo2_df(train, group, drop_columns, filters, fold, file_name, hold_out_folder, scaler_id=None, full_load=False):
    df = preload_vo2_df(train, group, drop_columns, filters, fold, file_name, hold_out_folder, full_load=full_load)

    unscaled_df = df.copy(deep=True)
    scaled_df = scale_df(df, train, scaler_id=scaler_id)
    scaled_df = change_column_data_type(scaled_df)
    return scaled_df, unscaled_df


class ColumnScaler(BaseEstimator, TransformerMixin):
    columns: list

    def __init__(self, scaler=None, drop=['target']):
        self.scaler = StandardScaler() if scaler is None else scaler
        self.drop = drop

    def fit(self, X, y=None):
        self.columns = X.select_dtypes(include=['float64']).columns
        self.columns = list(set(self.columns) - set(self.drop))
        self.scaler.fit(X[self.columns])
        return self

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def transform(self, X):
        X[self.columns] = self.scaler.transform(X[self.columns])
        return X

    def tranform_df(self, X):
        intersec = [col for col in self.columns if col in X.columns]

        mean_map = dict(zip(self.scaler.feature_names_in_, self.scaler.mean_))
        scale_map = dict(zip(self.scaler.feature_names_in_, self.scaler.scale_))

        for col in intersec:
            X[col] = (X[col] - mean_map[col]) / scale_map[col]

        return X
