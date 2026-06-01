import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_processing import (load_data, get_data_summary, detect_missing_values, handle_missing_values, detect_task_type, prepare_data, split_and_scale, encode_categorical, apply_pca)
from visualization import plot_feature_importance, plot_regression_results, plot_classification_results, plot_correlation_matrix, plot_model_comparison, plot_clusters
from utils import calculate_metrics
from models import get_supervised_models, train_supervised, train_multiple_models, get_unsupervised_models, train_unsupervised
#page configuration
st.set_page_config(page_title = "ML Project App", layout="wide")
st.title("Machine Learning Project App🚀")
st.write("Upload your dataset and build your ML models with ease! 📊")

#uploading
uploaded_file = st.file_uploader("Upload CSV or Excel file 🗂️", type = ["csv", "xlsx"])


if uploaded_file:
    df = load_data(uploaded_file)
    st.write("Raw Data Preview 👀:")
    st.dataframe(df.head())

    st.subheader("Dataset Summary 📈")
    numeric_summary, categorical_summary, numeric_cols, categorical_cols = get_data_summary(df)
    st.write("Numerical Features:")
    st.dataframe(numeric_summary)
    st.write("Categorical Features:")
    st.dataframe(categorical_summary)

    #handling missing values
    missing_columns = detect_missing_values(df)
    if missing_columns:
        st.subheader("Missing Values Detected ⚠️")
        st.write(f"Columns with missing values: {missing_columns}")


        missing_strategies={}
        for col in missing_columns:
            strategy = st.selectbox(
            f"How to handle missing value in {col}? 🤔",
            ["Mean", "Median", "Mode", "Drop", "Custom Value"],
            key = f"missing_{col}"            
            )
            if strategy == "Custom Value":
                custom_value = st.text_input(f"Enter value for {col}", key = f"custom_{col}")
                missing_strategies[col] = custom_value if custom_value else 0
            else:
                missing_strategies[col] = strategy

        if st.button("Apply Missing Value Handling ✅"):
            df = handle_missing_values(df, missing_strategies)
            st.write("Updated Dataset Preview 🔄:")
            st.dataframe(df.head())

# normalization options

    normalization_method = st.selectbox(
        "Select Normalization Method 🛠️",
        ["Standard (Z-score)", "MinMax (0-1)", "Robust", "None"],
        index = 0
    )

    norm_dict = {
        "Standard (Z-score)": "Standard",
        "MinMax (0-1)": "MinMax",
        "Robust":"Robust",
        "None": None
    }
    learning_type = st.selectbox("Select Learning Type 🎓", ["Supervised", "Unsupervised"])
    columns = df.columns.tolist()

    if learning_type == "Supervised":
        target_col = st.selectbox("Select Target Column 🎯", columns)
        detected_task = detect_task_type(df[target_col])
        st.write(f"Detected Task Type: {detected_task} 🧠")

        task_type = st.radio("Confirm Task Type ✅",
                             ["Regression", "Classification"],
                             index =0 if detected_task == "Regression" else 1)
        
        feature_cols = st.multiselect("Select Feature Columns 🗒️",
                                      [col for col in columns if col != target_col],
                                      default = [col for col in columns if col != target_col])
        
# correlation visualisation
        if st.button("Show Correlation Matrix 🌐"):
            st.subheader("Feature_Target Correlation 🌐")
            plot_correlation_matrix(df, feature_cols, target_col, st)

        test_size = st.slider("Test Split Ratio 𐄷", 0.1, 0.5, 0.2, 0.05)
        model_options = get_supervised_models(task_type)
        selected_models = st.multiselect("Select Models to Train 🏋🏻‍♂️",
                                         list(model_options.keys()),
                                         default = list(model_options.keys()))
        
        train_mode = st.radio("Training Mode", ["Single Model", "Compare Multiple Models"])
        
        if st.button("Train Model(s) 🚀"):
            try:
                X, y = prepare_data(df, feature_cols, target_col)
                X_train, X_test, y_train, y_test, scaler = split_and_scale(
                    X, y, test_size, norm_dict[normalization_method]
                )

                if train_mode == "Single Model":
                    if len(selected_models) != 1:
                        st.error("Please, select exactly one model for Single Model mode ❌")
                    else:
                        model = model_options[selected_models[0]]
                        y_pred = train_supervised(model, X_train, y_train, X_test)

                        st.subheader("Results 🎉")
                        plot_feature_importance(model, X, st)
                        metrics = calculate_metrics(task_type, y_test, y_pred)
                        for metric, value, in metrics.items():
                            st.write(f"{metric}: {value}")
                        if task_type == "Regression":
                            plot_regression_results(y_test, y_pred, st)
                        else:
                            plot_classification_results(y_test, y_pred, st)
                else:
                    if not selected_models:
                        st.error("Please select at least one model to compare ❌")
                    else:
                        models_to_train = {name: model_options[name] for name in selected_models}
                        results = train_multiple_models(models_to_train, X_train, y_train, X_test, y_test, task_type)
                        
                        st.subheader("Model Comparison 🎉")
                        plot_model_comparison(results, task_type, st)

                        for name, result in results.items():
                            st.write(f"{name} Score ({'RMSE' if task_type == 'Regression' else 'Accuracy'}): {result['score']:.4f}")
                            plot_feature_importance (result['model'], X, st)
                            if task_type == 'Regression':
                                plot_regression_results(y_test, result['y_pred'], st)
                            else:
                                plot_classification_results(y_test, result['y_pred'], st)
            except ValueError as e:
                st.error(f"Error: {str(e)} ❌")
                st.write("Please ensure all missing values are handled properly.")

    else:
        feature_cols = st.multiselect("Select Feature Columns 🗒️", columns, default = columns)

        use_pca = st.checkbox("Apply PCA for Dimensionality Reduction 🧩")
        if use_pca:
            n_components = st.slider("Number of PCA Components 📉", 2, min(len(feature_cols), 10), 2)

        model_options = get_unsupervised_models()
        selected_model = st.selectbox("Select Clustering Algorithm 🏷️", list(model_options.keys()))

        if selected_model == "KMeans":
            n_clusters = st.slider("Number of Clusters (K) 🔢", 2, 10, 3)
            params = {'n_clusters':n_clusters}
        else:
            eps = st.slider("Epsilon (eps) for DBSCAN 🔍", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("Minimum Samples for DBSCAN 🧮", 2, 10, 5)
            params = {'eps':eps, 'min_samples':min_samples}

        if st.button("Run Clustering 🚀"):
            try:
                df_subset = df[feature_cols]
                categorical_cols_subset = df_subset.select_dtypes(exclude=[np.number]).columns
                
                if len(categorical_cols_subset) > 0:
                    df_encoded, _ = encode_categorical(df_subset)
                else:
                    df_encoded = df_subset
                X = prepare_data(df_encoded, feature_cols)
                X_scaled, scaler = split_and_scale(X, normalization = norm_dict[normalization_method])
                
                if use_pca:
                    X_scaled, pca = apply_pca(X_scaled, n_components)
                    st.write(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_} 📊")
                clusters = train_unsupervised(model_options[selected_model], X_scaled, params)

                st.subheader("Clustering Results 🎉")
                plot_clusters(df, feature_cols, clusters, st)
                st.write("Number of samples per cluster:")
                cluster_counts = pd.Series(clusters).value_counts()
                st.write(cluster_counts)
                if -1 in cluster_counts.index:
                    st.write("Note: -1 indicates noise points in DBSCAN")
            except ValueError as e:
                st.error(f"Error: {str(e)} ❌")
                st.write("Please ensure all parameters are handled properly.")

st.sidebar.title("Instructions 📚")
st.sidebar.write("""
1. Upload your CSV or Excel file 🗂️
2. Handle any missing values if detected ⚠️
3. Choose normalization method 🛠️
4. Select learning type (Supervised/Unsupervised) 🎓
5. For Supervised: View correlation, then train single or multiple models 🏋
6. For Unsupervised: Configure clustering parameters 🏷️            
""")
st.sidebar.header("Requirements 🛠️")
st.sidebar.code("pip install streamlit pandas numpy sklearn seaborn matplotlib xgboost")
