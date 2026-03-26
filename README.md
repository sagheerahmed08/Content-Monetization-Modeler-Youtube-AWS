# Content Monetization Modeler — YouTube Ad Revenue Predictor

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.47.1-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![AWS S3](https://img.shields.io/badge/AWS-S3-FF9900?logo=amazon-aws)](https://aws.amazon.com/s3/)

A production-grade Streamlit application that predicts YouTube ad revenue using machine learning models trained on engagement metrics. Built with AWS S3 for model and data storage, scikit-learn pipelines for reproducible training, and automated hyperparameter tuning.

---

## Live App

Deployed on Streamlit Community Cloud — models and data stored on AWS S3 (no local files required).

---

## Features

- **EDA Page** — Missing value heatmaps, correlation matrix, distribution plots, outlier detection, key revenue driver insights
- **Model Training** — Automated RidgeCV/LassoCV hyperparameter tuning, 6-model comparison (LinearRegression, Ridge, Lasso, PassiveAggressiveRegressor, RandomForest, XGBoost), cross-validation R² ranking, automatic best model selection, S3 upload
- **Feature Importance** — Horizontal bar chart for tree (`feature_importances_`) and linear (`|coef_|` normalized) models
- **Revenue Prediction** — Physics-based revenue cap (`MAX_RPM = $20/1000 views`), ±1 RMSE confidence interval, per-country currency display (INR/USD/GBP/CAD), live exchange rates, input validation, CSV export
- **CI/CD** — GitHub Actions runs pytest on every push to main

---

## Architecture

```
User Browser
    │
    ▼
Streamlit App (4 pages)
    │
    ├── AWS S3 ── Raw CSV / Cleaned CSV / Model .joblib / results.csv
    └── ExchangeRate API (live USD → INR / GBP / CAD)
```

**Data flow:**
1. Raw CSV → preprocessing pipeline (dedup, imputation, feature engineering) → cleaned CSV saved to S3
2. Model Training page loads cleaned CSV, trains all models with CV tuning, uploads each `.joblib` + `results.csv` to S3
3. Prediction page loads selected model from S3, applies physics-based revenue cap, shows confidence interval

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.47 |
| ML | scikit-learn 1.7, XGBoost 3.0 |
| Data | pandas, numpy, scipy |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Storage | AWS S3 (boto3) |
| CI/CD | GitHub Actions (pytest on push) |
| Tests | pytest (21 unit tests) |

---

## Project Structure

```
app/
├── 1_Home.py                   # Landing page
├── pages/
│   ├── 2_EDA.py                # Exploratory Data Analysis
│   ├── 3_Model.py              # Model training & visualization
│   └── 4_Prediction.py         # Revenue prediction
├── preprocessing.py            # Data cleaning pipeline
├── config.py                   # S3 bucket / key constants
├── requirements.txt
└── tests/
    ├── test_preprocessing.py   # 7 tests for preprocessing logic
    └── test_prediction_logic.py # 14 tests for validation & revenue cap
.github/workflows/tests.yml     # CI: run pytest on push to main
reference/                      # Legacy scripts (not part of app)
```

---

## Setup

### Prerequisites

- Python 3.10+
- AWS account with an S3 bucket
- Streamlit Community Cloud account (for deployment)

### Local Development

```bash
git clone https://github.com/sagheerahmed08/Content-Monetization-Modeler-Youtube-AWS
cd Content-Monetization-Modeler-Youtube-AWS/app

pip install -r requirements.txt

# Create Streamlit secrets file
mkdir -p .streamlit
cat > .streamlit/secrets.toml << 'EOF'
[aws]
aws_access_key_id = "YOUR_KEY"
aws_secret_access_key = "YOUR_SECRET"
region = "eu-north-1"
EOF

streamlit run 1_Home.py
```

### AWS S3 Structure

```
s3://your-bucket/
├── Data/
│   ├── Raw/youtube_ad_revenue_dataset.csv
│   └── Cleaned/youtube_ad_revenue_dataset_cleaned.csv
└── models/
    ├── BestModel.joblib
    ├── LinearRegression.joblib
    ├── Ridge.joblib
    ├── Lasso.joblib
    ├── PassiveAggressiveRegressor.joblib
    ├── RandomForest.joblib
    ├── XGBoost.joblib
    └── results.csv
```

Update `app/config.py` with your bucket name. Models are populated automatically after clicking "Train Model".

### Running Tests

```bash
cd app
pytest tests/ -v
```

---

## Models

| Model | Configuration |
|---|---|
| LinearRegression | Baseline (no regularisation) |
| Ridge | Alpha auto-tuned via RidgeCV (50 log-spaced alphas, 5-fold CV) |
| Lasso | Alpha auto-tuned via LassoCV (50 log-spaced alphas, 5-fold CV) |
| PassiveAggressiveRegressor | Online learning baseline, max_iter=1000 |
| RandomForest | 150 estimators, max_depth=15 |
| XGBoost | 150 estimators, max_depth=10, lr=0.1 |

Best model selected by cross-validated R² and saved as `BestModel.joblib`.

---

## Revenue Prediction Logic

Raw model output is clipped to a physics-based cap:

```python
predicted = clip(raw_prediction, 0, views × MAX_RPM / 1000)
# MAX_RPM = $20.00 — maximum realistic revenue per 1000 views
```

This prevents nonsensical outputs (e.g., $280 for 1 view). Confidence interval is displayed as ±1 RMSE sourced from the stored `results.csv`.

---

## Input Features

| Feature | Type | Description |
|---|---|---|
| views | numeric | Total video views |
| likes | numeric | Total likes |
| comments | numeric | Total comments |
| watch_time_minutes | numeric | Aggregate watch time |
| video_length_minutes | numeric | Video duration |
| subscribers | numeric | Channel subscriber count |
| category | categorical | Entertainment / Gaming / Education / Music / News |
| device | categorical | Mobile / Tablet / TV / Desktop |
| country | categorical | IN / US / CA / UK |

Derived automatically: `engagement_rate = (likes + comments) / views`, `avg_watch_time_per_view = watch_time_minutes / views`.

---

## Author

**Sagheer Ahmed**
- GitHub: [@sagheerahmed08](https://github.com/sagheerahmed08)
- Project: [Content-Monetization-Modeler-Youtube-AWS](https://github.com/sagheerahmed08/Content-Monetization-Modeler-Youtube-AWS)