# ML Project App 🚀

An interactive, no-code machine learning web application built with Streamlit. Upload any CSV or Excel dataset and train, evaluate, and compare ML models — no coding required.

> **Demo dataset:** Tested on the Titanic dataset (Kaggle) — XGBoost achieved **84% accuracy**, Random Forest **82%**, Logistic Regression **80%** in multi-model comparison mode.

---

## Features

### Supervised Learning
- **Task auto-detection** — automatically identifies Regression vs Classification from your target column
- **Models:** Logistic/Linear Regression, Random Forest, XGBoost
- **Single model** or **multi-model comparison** with side-by-side metrics
- Feature importance plots, confusion matrix, actual vs predicted charts

### Unsupervised Learning
- **Clustering:** KMeans and DBSCAN with configurable parameters
- **Dimensionality reduction:** PCA before clustering (optional)
- Cluster visualisation via pairplot

### Data Preprocessing
- Missing value handling: Mean, Median, Mode, Drop, or Custom Value — per column
- Normalisation: StandardScaler, MinMaxScaler, RobustScaler — fitted on train set only (no leakage)
- Automatic categorical encoding
- Correlation matrix visualisation

---

## Project Structure

```
ML-Project-App/
├── main.py              # Streamlit UI — all user interaction
├── data_processing.py   # Loading, cleaning, encoding, splitting, scaling
├── models.py            # Model definitions, training (supervised & unsupervised)
├── visualization.py     # All plots (feature importance, confusion matrix, clusters)
└── utils.py             # Metrics calculation
```

Each module is fully independent — ML logic can be imported and reused in Jupyter notebooks or Kaggle without modification.

---

## Quickstart

```bash
git clone https://github.com/Mirjalol-Eshmurodov/ML-Project-App.git
cd ML-Project-App
pip install -r requirements.txt
streamlit run main.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How to Use

1. Upload a CSV or Excel file
2. Handle missing values (per-column strategy)
3. Choose a normalisation method
4. Select **Supervised** or **Unsupervised** learning
5. For Supervised: pick target column, confirm task type, select features and models → Train
6. For Unsupervised: select features, configure KMeans/DBSCAN parameters → Cluster

---

## Tech Stack

`Python` · `Streamlit` · `scikit-learn` · `XGBoost` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

---

## Design Decisions

- **Scaler leakage prevention** — StandardScaler/MinMaxScaler/RobustScaler is fitted on training data only, then applied to the test set via the same fitted object
- **Modular architecture** — UI layer (`main.py`) is fully decoupled from ML logic, enabling reuse without Streamlit
- **Parallel model training** — multiple models trained and evaluated in a single pass, results displayed side-by-side

---

## Author

**Mirjalol Eshmurodov** — Data Scientist & ML Engineer  
[LinkedIn](https://www.linkedin.com/in/mirjalol-eshmurodov-1ab434355) · [GitHub](https://github.com/Mirjalol-Eshmurodov) · [Medium](https://medium.com/@mirjaloleshmurodov1017)
