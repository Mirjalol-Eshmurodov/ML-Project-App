import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# load csv and excel file

def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

# separate numerical and categorical data and provide sumary

def get_data_summary(df):
    numeric_cols = df.select_dtypes(include = [np.number]).columns
    categorical_cols = df.select_dtypes(exclude = [np.number]).columns

    numeric_summary = df[numeric_cols].describe()
    categorical_summary = df[categorical_cols].describe()

    return numeric_summary, categorical_summary, numeric_cols, categorical_cols
# detect missing values
def detect_missing_values(df):
    return df.columns[df.isnull().any()].tolist()

# handling missing values based on user strategy
def handle_missing_values(df, strategy_dict):
    df_copy = df.copy()
    for col, strategy in strategy_dict.items():
        if strategy == "Mean":
    imp = SimpleImputer(strategy='mean')
    df_copy[col] = imp.fit_transform(df_copy[[col]]).flatten()
elif strategy == "Median":
    imp = SimpleImputer(strategy='median')
    df_copy[col] = imp.fit_transform(df_copy[[col]]).flatten()
elif strategy == "Mode":
    imp = SimpleImputer(strategy='most_frequent')
    df_copy[col] = imp.fit_transform(df_copy[[col]]).flatten()
    return df_copy

# smart detection of task type

def detect_task_type(target_data):
    if pd.api.types.is_numeric_dtype(target_data):
        unique_values = target_data.unique()
        return "Regression" if len(unique_values) >20 else "Classification"
    return "Classification"

#encode-categorical

def encode_categorical(df, columns = None):
    df_copy = df.copy()
    if columns is None:
        columns = df_copy.select_dtypes(exclude = [np.number]).columns

    le_dict = {}
    for col in columns:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col].astype(str))
        le_dict[col] = le
    return df_copy, le_dict

#prepare data for encoding and imputation
def prepare_data(df, feature_cols, target_col = None):
    X = df[feature_cols].copy()
    numeric_cols = X.select_dtypes(include = [np.number]).columns
    if len(numeric_cols) > 0:
        imp = SimpleImputer(strategy='mean')
        X[numeric_cols] = imp.fit_transform(X[numeric_cols])
    X = pd.get_dummies(X) #one-hot encoding

    if target_col:
        y = df[target_col]
        if y.isnull().any():
            if pd.api.types.is_numeric_dtype(y):
                imp = SimpleImputer(strategy='mean')
            else:
                imp = SimpleImputer(strategy='most_frequent')
            y = pd.Series(imp.fit_transform(y.values.reshape(-1,1)).flatten(), index = y.index)
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y)
            y = pd.Series(y, index=df[target_col].index)
        
        return X, y
    return X

def normalize_data(X, scaler=None, method='Standard'):
    if method is None:
        return X, None          
    if method == 'Standard':
        scaler_obj = StandardScaler()
    elif method == 'MinMax':
        scaler_obj = MinMaxScaler()
    elif method == 'Robust':
        scaler_obj = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if scaler is not None:
        return scaler.transform(X), scaler
    
    # Otherwise fit and transform
    scaled = scaler_obj.fit_transform(X)
    return scaled, scaler_obj

#split and normalize the data
def split_and_scale(X, y=None, test_size=0.2, normalization="Standard"):
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        X_train_scaled, scaler = normalize_data(X_train, method=normalization)
        X_test_scaled, _ = normalize_data(X_test, scaler=scaler, method=normalization)
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    else:
        X_scaled, scaler = normalize_data(X, method=normalization)
        return X_scaled, scaler
    
def apply_pca(X, n_components):
    pca = PCA(n_components = n_components)
    return pca.fit_transform(X), pca
