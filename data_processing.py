import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def get_data_summary(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    numeric_summary = df[numeric_cols].describe()
    categorical_summary = df[categorical_cols].describe()
    return numeric_summary, categorical_summary, numeric_cols, categorical_cols


def detect_missing_values(df):
    return df.columns[df.isnull().any()].tolist()


def handle_missing_values(df, strategy_dict):
    df_copy = df.copy()
    for col, strategy in strategy_dict.items():
        if col not in df_copy.columns:
            continue
        if strategy == "Drop":
            df_copy = df_copy.dropna(subset=[col])
        elif strategy == "Mean":
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        elif strategy == "Median":
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        elif strategy == "Mode":
            mode = df_copy[col].mode()
            if not mode.empty:
                df_copy[col] = df_copy[col].fillna(mode[0])
        else:
            df_copy[col] = df_copy[col].fillna(strategy)
    return df_copy


def detect_task_type(target_data):
    if pd.api.types.is_numeric_dtype(target_data):
        return "Regression" if len(target_data.unique()) > 20 else "Classification"
    return "Classification"


def encode_categorical(df, columns=None):
    df_copy = df.copy()
    if columns is None:
        columns = df_copy.select_dtypes(exclude=[np.number]).columns
    le_dict = {}
    for col in columns:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col].astype(str))
        le_dict[col] = le
    return df_copy, le_dict


def prepare_data(df, feature_cols, target_col=None):
    X = df[feature_cols].copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].mean())
    X = pd.get_dummies(X)

    if target_col:
        y = df[target_col].copy()
        if y.isnull().any():
            if pd.api.types.is_numeric_dtype(y):
                y = y.fillna(y.mean())
            else:
                y = y.fillna(y.mode()[0])
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=df.index)
        return X, y
    return X


def normalize_data(X, scaler=None, method='Standard'):
    if method is None:
        return X, None
    scalers = {'Standard': StandardScaler, 'MinMax': MinMaxScaler, 'Robust': RobustScaler}
    if method not in scalers:
        raise ValueError(f"Unknown normalization method: {method}")
    if scaler is not None:
        return scaler.transform(X), scaler
    scaler_obj = scalers[method]()
    return scaler_obj.fit_transform(X), scaler_obj


def split_and_scale(X, y=None, test_size=0.2, normalization="Standard"):
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train_scaled, scaler = normalize_data(X_train, method=normalization)
        X_test_scaled, _ = normalize_data(X_test, scaler=scaler, method=normalization)
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    else:
        X_scaled, scaler = normalize_data(X, method=normalization)
        return X_scaled, scaler


def apply_pca(X, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X), pca
