from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

def get_supervised_models(task_type):
    if task_type == "Regression":
        return {"Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "XGBoost": xgb.XGBRegressor()}
    return {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": xgb.XGBClassifier()
    }


#train supervised model
def train_supervised(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)

#train multiple models and return scores
def train_multiple_models(models, X_train, y_train, X_test, y_test, task_type):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if task_type == 'Regression':
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse) #RMSE
            results[name] = {"model": model, "score": rmse, "y_pred":y_pred}
        else:
            score = accuracy_score(y_test, y_pred)
            results[name] = {"model":model, "score":score, "y_pred": y_pred}
    return results

def get_unsupervised_models():
    return {
        "KMeans": KMeans,
        "DBSCAN": DBSCAN
    }

def train_unsupervised(model_class, X_scaled, params):
    if model_class == KMeans:
        cluster_model = model_class(n_clusters=params['n_clusters'], random_state=42, n_init=10)
    elif model_class == DBSCAN:
        cluster_model = model_class(eps=params['eps'], min_samples=params['min_samples'])
    return cluster_model.fit_predict(X_scaled)