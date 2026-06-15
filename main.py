import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_processing import (load_data, get_data_summary, detect_missing_values,
                              handle_missing_values, detect_task_type, prepare_data,
                              split_and_scale, encode_categorical, apply_pca)
from visualization import (plot_feature_importance, plot_regression_results,
                            plot_classification_results, plot_correlation_matrix,
                            plot_model_comparison, plot_clusters)
from utils import calculate_metrics
from models import (get_supervised_models, train_supervised, train_multiple_models,
                    get_unsupervised_models, train_unsupervised)

st.set_page_config(page_title="ML Studio", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0f1117; color: #e2e8f0; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1400px; }

.top-bar {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.top-bar::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.top-bar h1 {
    font-size: 2.2rem; font-weight: 700; color: #f1f5f9;
    margin: 0 0 0.5rem 0; letter-spacing: -0.02em;
}
.top-bar h1 span { color: #818cf8; }
.top-bar p { font-size: 1rem; color: #94a3b8; margin: 0; }

.tag {
    display: inline-flex; align-items: center;
    background: #1e293b; border: 1px solid #334155;
    border-radius: 8px; padding: 0.4rem 0.9rem;
    font-size: 0.78rem; font-weight: 600; color: #818cf8;
    letter-spacing: 0.05em; text-transform: uppercase;
    margin-bottom: 0.75rem;
}

.card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 12px; padding: 1.5rem 2rem; margin: 1.25rem 0;
}
.card-title {
    font-size: 1rem; font-weight: 600; color: #f1f5f9;
    margin: 0 0 1rem 0;
}

.stat {
    flex: 1; min-width: 140px;
    background: #0f172a; border: 1px solid #334155;
    border-radius: 10px; padding: 1rem 1.25rem;
}
.stat-label {
    font-size: 0.72rem; font-weight: 600; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.4rem;
}
.stat-value {
    font-size: 1.6rem; font-weight: 700; color: #818cf8;
    font-family: 'JetBrains Mono', monospace;
}

.badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.3rem 0.75rem; border-radius: 999px;
    font-size: 0.78rem; font-weight: 500;
}
.badge-blue { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }
.badge-green { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }

.note {
    background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.25);
    border-left: 3px solid #6366f1; border-radius: 8px;
    padding: 0.85rem 1.2rem; color: #a5b4fc; font-size: 0.88rem;
}
.alert {
    background: rgba(251,146,60,0.08); border: 1px solid rgba(251,146,60,0.25);
    border-left: 3px solid #fb923c; border-radius: 8px;
    padding: 0.85rem 1.2rem; color: #fdba74; font-size: 0.88rem;
}

.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #0f172a !important; border: 1px solid #334155 !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
}
.stSlider > div > div > div { background: #6366f1 !important; }
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 0.9rem !important; padding: 0.6rem 2rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.35) !important;
}
.stRadio label {
    background: #0f172a; border: 1px solid #334155;
    border-radius: 8px; padding: 0.4rem 1rem; color: #94a3b8;
}
.stDataFrame { border-radius: 10px; overflow: hidden; }

[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b !important;
}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    'figure.facecolor': '#1e293b',
    'axes.facecolor': '#0f172a',
    'axes.edgecolor': '#334155',
    'axes.labelcolor': '#94a3b8',
    'text.color': '#e2e8f0',
    'xtick.color': '#64748b',
    'ytick.color': '#64748b',
    'grid.color': '#1e293b',
    'grid.alpha': 0.8,
})

with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem 0;">
        <div style="font-size:1.3rem; font-weight:700; color:#f1f5f9;">⚡ ML Studio</div>
        <div style="font-size:0.78rem; color:#475569; margin-top:0.2rem;">by Mirjalol Eshmurodov</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.72rem; font-weight:600; color:#475569; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.75rem;">Steps</div>', unsafe_allow_html=True)

    for num, label in [("01", "Upload dataset"), ("02", "Handle missing values"),
                        ("03", "Normalization"), ("04", "Learning type"), ("05", "Train & evaluate")]:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.75rem; padding:0.5rem 0.75rem;
                    border-radius:8px; margin-bottom:0.3rem; background:#1e293b;">
            <span style="font-size:0.65rem; font-weight:700; color:#6366f1;
                         font-family:'JetBrains Mono',monospace;">{num}</span>
            <span style="font-size:0.82rem; color:#94a3b8;">{label}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr style="border:none; border-top:1px solid #1e293b; margin:1.5rem 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.72rem; font-weight:600; color:#475569; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">Stack</div>', unsafe_allow_html=True)
    for lib in ["Python", "scikit-learn", "XGBoost", "Streamlit", "pandas"]:
        st.markdown(f'<div style="font-size:0.78rem; color:#64748b; padding:0.2rem 0;">· {lib}</div>', unsafe_allow_html=True)

st.markdown("""
<div class="top-bar">
    <h1>Machine Learning <span>Studio</span></h1>
    <p>Upload any dataset — train, evaluate, and compare ML models without writing code.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="tag">01 — Upload</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"], label_visibility="collapsed")

if not uploaded_file:
    st.markdown('<div class="note">Drop a <strong>.csv</strong> or <strong>.xlsx</strong> file above to get started.</div>', unsafe_allow_html=True)
    st.stop()

df = load_data(uploaded_file)
numeric_summary, categorical_summary, numeric_cols, categorical_cols = get_data_summary(df)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Dataset Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, label, val in zip([c1, c2, c3, c4],
                             ["Rows", "Columns", "Numeric", "Categorical"],
                             [f"{df.shape[0]:,}", df.shape[1], len(numeric_cols), len(categorical_cols)]):
    with col:
        st.markdown(f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-value">{val}</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Preview data"):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("Feature statistics"):
    t1, t2 = st.tabs(["Numeric", "Categorical"])
    with t1:
        st.dataframe(numeric_summary, use_container_width=True)
    with t2:
        st.dataframe(categorical_summary, use_container_width=True)

missing_columns = detect_missing_values(df)

if missing_columns:
    st.markdown('<div class="tag">02 — Missing Values</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="alert">Missing values detected in: <strong>{", ".join(missing_columns)}</strong></div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    missing_strategies = {}
    for i in range(0, len(missing_columns), 2):
        row = st.columns(2)
        for j, col in enumerate(missing_columns[i:i+2]):
            with row[j]:
                strategy = st.selectbox(f"`{col}`", ["Mean", "Median", "Mode", "Drop", "Custom Value"], key=f"missing_{col}")
                if strategy == "Custom Value":
                    val = st.text_input(f"Value for {col}", key=f"custom_{col}")
                    missing_strategies[col] = val if val else 0
                else:
                    missing_strategies[col] = strategy

    if st.button("Apply"):
        df = handle_missing_values(df, missing_strategies)
        st.markdown('<div class="note">Done — missing values handled.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="tag">03 — Settings</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

ca, cb = st.columns(2)
with ca:
    normalization_method = st.selectbox("Normalization", ["Standard (Z-score)", "MinMax (0-1)", "Robust", "None"])
with cb:
    learning_type = st.selectbox("Learning Type", ["Supervised", "Unsupervised"])

st.markdown('</div>', unsafe_allow_html=True)

norm_map = {"Standard (Z-score)": "Standard", "MinMax (0-1)": "MinMax", "Robust": "Robust", "None": None}
columns = df.columns.tolist()

if learning_type == "Supervised":
    st.markdown('<div class="tag">04 — Supervised</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        target_col = st.selectbox("Target Column", columns)
        detected_task = detect_task_type(df[target_col])
        color = "badge-green" if detected_task == "Classification" else "badge-blue"
        st.markdown(f'<span class="badge {color}">Detected: {detected_task}</span>', unsafe_allow_html=True)
    with c2:
        task_type = st.radio("Task Type", ["Regression", "Classification"],
                              index=0 if detected_task == "Regression" else 1, horizontal=True)

    st.markdown('<hr style="border:none; border-top:1px solid #1e293b; margin:1rem 0;">', unsafe_allow_html=True)

    feature_cols = st.multiselect("Feature Columns",
                                   [c for c in columns if c != target_col],
                                   default=[c for c in columns if c != target_col])

    c3, c4 = st.columns(2)
    with c3:
        test_size = st.slider("Test split", 0.1, 0.5, 0.2, 0.05)
    with c4:
        train_mode = st.radio("Mode", ["Single Model", "Compare Multiple Models"], horizontal=True)

    model_options = get_supervised_models(task_type)
    selected_models = st.multiselect("Models", list(model_options.keys()), default=list(model_options.keys()))

    st.markdown('</div>', unsafe_allow_html=True)

    btn1, btn2, _ = st.columns([1, 1, 3])
    with btn1:
        show_corr = st.button("Correlation Matrix")
    with btn2:
        train_btn = st.button("Train ⚡")

    if show_corr:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Correlation Matrix</div>', unsafe_allow_html=True)
        plot_correlation_matrix(df, feature_cols, target_col, st)
        st.markdown('</div>', unsafe_allow_html=True)

    if train_btn:
        try:
            with st.spinner("Training..."):
                X, y = prepare_data(df, feature_cols, target_col)
                X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, test_size, norm_map[normalization_method])

            st.markdown('<div class="tag">05 — Results</div>', unsafe_allow_html=True)

            if train_mode == "Single Model":
                if len(selected_models) != 1:
                    st.markdown('<div class="alert">Select exactly one model for single model mode.</div>', unsafe_allow_html=True)
                else:
                    model = model_options[selected_models[0]]
                    y_pred = train_supervised(model, X_train, y_train, X_test)
                    metrics = calculate_metrics(task_type, y_test, y_pred)

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="card-title">{selected_models[0]}</div>', unsafe_allow_html=True)

                    mcols = st.columns(len(metrics))
                    for i, (metric, value) in enumerate(metrics.items()):
                        with mcols[i]:
                            v = f"{value:.4f}" if isinstance(value, float) else str(value)
                            st.markdown(f'<div class="stat"><div class="stat-label">{metric}</div><div class="stat-value">{v}</div></div>', unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    plot_feature_importance(model, X, st)
                    if task_type == "Regression":
                        plot_regression_results(y_test, y_pred, st)
                    else:
                        plot_classification_results(y_test, y_pred, st)
                    st.markdown('</div>', unsafe_allow_html=True)

            else:
                if not selected_models:
                    st.markdown('<div class="alert">Select at least one model.</div>', unsafe_allow_html=True)
                else:
                    with st.spinner("Comparing models..."):
                        models_to_train = {n: model_options[n] for n in selected_models}
                        results = train_multiple_models(models_to_train, X_train, y_train, X_test, y_test, task_type)

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-title">Model Comparison</div>', unsafe_allow_html=True)
                    plot_model_comparison(results, task_type, st)

                    metric_label = "RMSE" if task_type == "Regression" else "Accuracy"
                    sorted_results = sorted(results.items(), key=lambda x: x[1]['score'],
                                            reverse=(task_type == "Classification"))

                    for rank, (name, result) in enumerate(sorted_results):
                        medal = ["🥇", "🥈", "🥉"][rank] if rank < 3 else ""
                        st.markdown(f"""
                        <div style="background:#0f172a; border:1px solid #334155; border-radius:10px;
                                    padding:1rem 1.25rem; margin:0.5rem 0;
                                    display:flex; align-items:center; justify-content:space-between;">
                            <span style="font-weight:600; color:#f1f5f9;">{medal} {name}</span>
                            <span style="font-family:'JetBrains Mono',monospace; color:#818cf8; font-weight:600;">
                                {metric_label}: {result['score']:.4f}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                    for name, result in results.items():
                        with st.expander(f"{name} — details"):
                            plot_feature_importance(result['model'], X, st)
                            if task_type == "Regression":
                                plot_regression_results(y_test, result['y_pred'], st)
                            else:
                                plot_classification_results(y_test, result['y_pred'], st)

        except ValueError as e:
            st.markdown(f'<div class="alert">{str(e)}</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="tag">04 — Unsupervised</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    feature_cols = st.multiselect("Feature Columns", columns, default=columns)

    c1, c2 = st.columns(2)
    with c1:
        use_pca = st.checkbox("Apply PCA")
        if use_pca:
            n_components = st.slider("Components", 2, min(len(feature_cols), 10), 2)
    with c2:
        model_options = get_unsupervised_models()
        selected_model = st.selectbox("Algorithm", list(model_options.keys()))

    if selected_model == "KMeans":
        n_clusters = st.slider("Clusters (K)", 2, 10, 3)
        params = {'n_clusters': n_clusters}
    else:
        c3, c4 = st.columns(2)
        with c3:
            eps = st.slider("Epsilon", 0.1, 2.0, 0.5, 0.1)
        with c4:
            min_samples = st.slider("Min samples", 2, 10, 5)
        params = {'eps': eps, 'min_samples': min_samples}

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Run ⚡"):
        try:
            with st.spinner("Clustering..."):
                df_subset = df[feature_cols]
                cat_cols = df_subset.select_dtypes(exclude=[np.number]).columns
                df_encoded = encode_categorical(df_subset)[0] if len(cat_cols) > 0 else df_subset
                X = prepare_data(df_encoded, feature_cols)
                X_scaled, _ = split_and_scale(X, normalization=norm_map[normalization_method])
                if use_pca:
                    X_scaled, pca = apply_pca(X_scaled, n_components)
                    st.markdown(f'<div class="note">PCA variance explained: {[f"{v:.1%}" for v in pca.explained_variance_ratio_]}</div>', unsafe_allow_html=True)
                clusters = train_unsupervised(model_options[selected_model], X_scaled, params)

            st.markdown('<div class="tag">05 — Results</div>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)

            cluster_counts = pd.Series(clusters).value_counts()
            n_found = len([c for c in cluster_counts.index if c != -1])
            noise = cluster_counts.get(-1, 0)

            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(f'<div class="stat"><div class="stat-label">Clusters</div><div class="stat-value">{n_found}</div></div>', unsafe_allow_html=True)
            with rc2:
                st.markdown(f'<div class="stat"><div class="stat-label">Noise points</div><div class="stat-value">{noise}</div></div>', unsafe_allow_html=True)

            plot_clusters(df, feature_cols, clusters, st)
            st.dataframe(cluster_counts.rename("Count").to_frame(), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        except ValueError as e:
            st.markdown(f'<div class="alert">{str(e)}</div>', unsafe_allow_html=True)
