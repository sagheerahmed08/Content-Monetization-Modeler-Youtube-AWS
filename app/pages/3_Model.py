import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import boto3

# ----------------------------
# Optional XGBoost
# ----------------------------
try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    has_xgb = True
except ImportError:
    has_xgb = False

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="YouTube Ad Revenue Predictor", page_icon="📊", layout="wide")
st.title("📊 YouTube Ad Revenue Predictor")

# ---------- AWS S3 Configuration ----------
S3_BUCKET = "youtube-ad-revenue-app-sagheer"
CLEAN_KEY = "Data/Cleaned/youtube_ad_revenue_dataset_cleaned.csv"
MODEL_PREFIX = "models"

s3 = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
    region_name=st.secrets["aws"]["region"]
)

# ---------- Helper Functions ----------
def eval_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred))
    }

def upload_model_to_s3(model, bucket, key, is_xgb=False):
    buffer = io.BytesIO()
    if is_xgb:
        model.save_model(buffer)
        buffer.seek(0)
        s3.upload_fileobj(buffer, bucket, key)
    else:
        joblib.dump(model, buffer)
        buffer.seek(0)
        s3.upload_fileobj(buffer, bucket, key)

def load_model_from_s3(bucket, key, is_xgb=False):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        bytestream = io.BytesIO(obj["Body"].read())
        if is_xgb and has_xgb:
            model = XGBRegressor()
            model.load_model(bytestream)
            return model
        else:
            return joblib.load(bytestream)
    except:
        return None

# ---------- Load Dataset ----------
try:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=CLEAN_KEY)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    st.success("✅ Cleaned dataset loaded successfully from S3")
except Exception as e:
    st.error(f"❌ Failed to load data from S3: {e}")
    st.stop()

st.write("### Dataset Preview")
st.dataframe(df.head())

# ---------- Feature Selection ----------
target_col = "ad_revenue_usd"
X = df.drop(columns=[target_col])
y = df[target_col]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# ---------- Preprocessing ----------
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["⚙️ Train Model", "📊 Model Evaluation"])

# ---------- TAB 1: TRAIN ----------
with tab1:
    st.subheader("🚀 Train Models")

    if st.button("🧠 Train Models"):
        with st.spinner("Training models... ⏳"):
            X_processed = preprocessor.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
            }

            if has_xgb:
                models["XGBoost"] = XGBRegressor(
                    n_estimators=100,
                    random_state=42,
                    objective='reg:squarederror',
                    learning_rate=0.1,
                    max_depth=6,
                    verbosity=0
                )

            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                upload_model_to_s3(model, S3_BUCKET, f"{MODEL_PREFIX}/{name}.joblib", is_xgb=(name=="XGBoost"))
                scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=5)
                results[name] = {"cv_r2_mean": np.mean(scores), "cv_r2_std": np.std(scores)}

            perf_df = pd.DataFrame([{"Model": k, "CV_R2_Mean": v['cv_r2_mean'], "CV_R2_STD": v['cv_r2_std']} for k, v in results.items()])
            st.success("✅ Models trained and uploaded to S3 successfully!")
            st.dataframe(perf_df)

# ---------- TAB 2: EVALUATION ----------
with tab2:
    st.subheader("📈 Model Evaluation")
    X_processed = preprocessor.fit_transform(X)
    cols = st.columns(2)

    model_files = ["LinearRegression.joblib", "Ridge.joblib", "Lasso.joblib", "RandomForest.joblib"]
    if has_xgb:
        model_files.append("XGBoost.joblib")

    for idx, model_path in enumerate(model_files):
        model_name = model_path.replace(".joblib", "")
        is_xgb = (model_name == "XGBoost")
        model = load_model_from_s3(S3_BUCKET, f"{MODEL_PREFIX}/{model_path}", is_xgb=is_xgb)
        if model is None:
            st.warning(f"{model_name} not found in S3")
            continue

        y_pred = model.predict(X_processed)
        metrics = eval_metrics(y, y_pred)

        with cols[idx % 2]:
            st.markdown(f"### {model_name}")
            c1, c2, c3 = st.columns(3)
            c1.metric("R²", f"{metrics['r2']:.3f}")
            c2.metric("MAE", f"{metrics['mae']:.2f}")
            c3.metric("RMSE", f"{metrics['rmse']:.2f}")

            fig, ax = plt.subplots()
            sns.scatterplot(x=y, y=y_pred, alpha=0.6, ax=ax)
            lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
            ax.plot(lims, lims, 'r--')
            ax.set_xlabel("Actual Revenue")
            ax.set_ylabel("Predicted Revenue")
            ax.set_title(f"{model_name} — Actual vs Predicted")
            st.pyplot(fig)

        if (idx + 1) % 2 == 0:
            cols = st.columns(2)
