from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def calculate_metrics(task_type, y_test, y_pred):
    if task_type == "Regression":
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)  # ← SQRT QOSHILDI!
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            "RMSE": rmse,
            "MAE": mae,
            "MSE": mse,
            "R² Score": r2
        }
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred)
    }