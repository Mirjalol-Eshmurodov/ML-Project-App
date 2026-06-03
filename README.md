# ML Project App 🚀

An interactive machine learning web application built with Streamlit. Upload any CSV or Excel dataset and train ML models without writing code.

## Features

- **Supervised Learning** — Regression and Classification
  - Linear/Logistic Regression, Random Forest, XGBoost
  - Single model or multi-model comparison
  - Feature importance, confusion matrix, actual vs predicted plots
- **Unsupervised Learning** — Clustering
  - KMeans and DBSCAN
  - PCA dimensionality reduction
  - Cluster visualization via pairplot
- **Data Preprocessing**
  - Missing value handling (mean, median, mode, drop, custom)
  - Normalization (Standard, MinMax, Robust)
  - Automatic categorical encoding
  - Correlation matrix visualization

## Installation

```bash
git clone https://github.com/Mirjalol-Eshmurodov/ML-Project-App.git
cd ML-Project-App
pip install -r requirements.txt
streamlit run main.py
```

## Usage

1. Upload a CSV or Excel file
2. Handle missing values if detected
3. Select normalization method
4. Choose Supervised or Unsupervised learning
5. Select features, target, and model(s) → Train

## Tech Stack

Python · Streamlit · scikit-learn · XGBoost · pandas · matplotlib · seaborn
