# src/modeling.py
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# Optional: XGBoost - import dynamically so the module isn't required at static-analysis time
import importlib
try:
    _xgb = importlib.import_module("xgboost")
    XGBRegressor = getattr(_xgb, "XGBRegressor")
    has_xgb = True
except Exception:
    # xgboost not installed or failed to import; downstream code will skip XGBoost model
    has_xgb = False

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent  # Project root
DATA_FILE = BASE_DIR / "app" / "Data" / "Cleaned" / "youtube_ad_revenue_dataset_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ----------------------------
# Load cleaned data
# ----------------------------
df = pd.read_csv(DATA_FILE)
print(f"Cleaned data loaded: {df.shape}")

# ----------------------------
# Features and target
# ----------------------------
num_features = [
    'views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes',
    'subscribers', 'engagement_rate', 'avg_watch_time_per_view'
]
cat_features = ['category', 'device', 'country']

# Derived features
df['engagement_rate'] = (df['likes'] + df['comments']) / df['views'].replace(0, 1)
df['avg_watch_time_per_view'] = df['watch_time_minutes'] / df['views'].replace(0, 1)

X = df[num_features + cat_features]
y = df['ad_revenue_usd'].fillna(0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# ----------------------------
# Models for comparison
# ----------------------------
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'RandomForest': RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    )
}

if has_xgb:
    models['XGBoost'] = XGBRegressor(
        n_estimators=200, random_state=42, n_jobs=1, objective='reg:squarederror'
    )

# ----------------------------
# Train, cross-validate, save models
# ----------------------------
results = {}

for name, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODEL_DIR / f"{name}.joblib")
    print(f"💾 Saved {name} model at {MODEL_DIR / f'{name}.joblib'}")
    
    scores = cross_val_score(pipe, X_train, y_train, scoring='r2', cv=5, n_jobs=-1)
    results[name] = {'cv_r2_mean': np.mean(scores), 'cv_r2_std': np.std(scores)}
    print(f"📊 {name}: CV R² = {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Save comparison results to CSV for Streamlit app
perf_df = pd.DataFrame([
    {"Model": k, "CV_R2_Mean": v['cv_r2_mean'], "CV_R2_STD": v['cv_r2_std']}
    for k, v in results.items()
])
perf_df.to_csv(MODEL_DIR / "results.csv", index=False)
print(f"📂 Model comparison results saved at {MODEL_DIR / 'results.csv'}")

# ----------------------------
# Fit best model
# ----------------------------
best_name = max(results, key=lambda k: results[k]['cv_r2_mean'])
best_model = models[best_name]
pipe_best = Pipeline([('preprocessor', preprocessor), ('model', best_model)])
pipe_best.fit(X_train, y_train)

# ----------------------------
# Evaluate on test set
# ----------------------------
y_pred = pipe_best.predict(X_test)

def eval_metrics(y_true, y_pred):
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }

metrics = eval_metrics(y_test, y_pred)
print(f"\n🏆 Best model: {best_name}")
print(f"📈 Test set performance: {metrics}")

# ----------------------------
# Save best model
# ----------------------------
joblib.dump(pipe_best, MODEL_DIR / "best_model.joblib")
print(f"💾 Best model saved at {MODEL_DIR / 'best_model.joblib'}")

# ----------------------------
# Feature importance for tree models
# ----------------------------
tree_models = ['RandomForest', 'XGBoost']
if best_name in tree_models:
    model_step = pipe_best.named_steps['model']
    ohe = pipe_best.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_names = ohe.get_feature_names_out(cat_features)
    feature_names = num_features + list(cat_names)
    
    importances = model_step.feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    print("\n🌟 Top 20 feature importances:")
    print(fi.head(20))
    
    fi.head(20).plot.barh(figsize=(10,6))
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.show()
