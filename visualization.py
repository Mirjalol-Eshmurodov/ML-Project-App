import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_processing import encode_categorical

#Plot correlation matrix for selected fetures and target only
def plot_correlation_matrix(df, feature_cols, target_col, st):
    
    #select categorical columns if any
    corr_cols = feature_cols + [target_col]
    df_subset = df[corr_cols]

    #Encode categorical columns if any
    categorical_cols = df_subset.select_dtypes(exclude = [np.number]).columns
    if len(categorical_cols)>0:
        df_encoded, _ = encode_categorical(df_subset, categorical_cols)
    else:
        df_encoded = df_subset

    #calculate correlation
    corr_df = df_encoded.corr()

    #plot
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr_df, annot =True, cmap = 'coolwarm', center=0 )
    plt.title("Correlation Matrix (Selected Features and Target)")
    st.pyplot(fig)

def plot_regression_results(y_test, y_pred, st):
    fig, ax = plt.subplots()
    plt.scatter(y_test, y_pred, alpha = 0.5)
    plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title("Actual vs Predicted")
    st.pyplot(fig)

def plot_classification_results(y_test, y_pred, st):
    fig, ax = plt.subplots()
    sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt='d')
    plt.title('Confusion Matrix')
    st.pyplot(fig)


#plot radar chart
def plot_model_comparison(results, task_type, st):
    categories = list(results.keys())
    values = [results[name]['score'] for name in categories]

    if task_type == 'Regression':
        max_score = max(values)
        values = [max_score / v if v !=0 else max_score for v in values] # Higher is better
    values += values[:1]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize = (6,6), subplot_kw = dict(polar = True))
    ax.fill(angles, values, color = 'blue', alpha =0.25)
    ax.plot(angles, values, color = 'blue', linewidth = 2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(f"Model Comparison ({'RMSE' if task_type == 'Regression' else 'Accuracy'})")
    st.pyplot(fig)

#plot feature
def plot_feature_importance(model, X, st):
    if hasattr(model, 'feature_importances_'):
        fig, ax = plt.subplots()
        importance = pd.DataFrame({
            'feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        sns.barplot(x = 'Importance', y = 'feature', data = importance)
        plt.title("Feature Importance")
        st.pyplot(fig)

#plot clustering results
def plot_clusters(df, feature_cols, clusters, st):
    df_copy = df.copy()
    df_copy['Cluster'] = clusters
    viz_cols = feature_cols[:4] if len(feature_cols) >4 else feature_cols
    viz_cols = viz_cols + ['Cluster']
    fig = sns.pairplot(df_copy[viz_cols], hue= 'Cluster', palette='deep')
    st.pyplot(fig.figure)