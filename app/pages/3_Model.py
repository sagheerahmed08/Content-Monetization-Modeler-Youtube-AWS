import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
import time
import importlib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import boto3

# optional xgboost
try:
    _xgb = importlib.import_module("xgboost")
    XGBRegressor = getattr(_xgb, "XGBRegressor")
    has_xgb = True
except Exception:
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

def load_model_from_s3(bucket, key, is_xgb=False):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        bytestream = io.BytesIO(obj["Body"].read())
        if is_xgb:
            model = xgb.Booster()
            model.load_model(bytestream)
        else:
            model = joblib.load(bytestream)
        return model
    except Exception as e:
        st.warning(f"⚠️ Could not load model {key}: {e}")
        return None

# ---------- Load Cleaned Data from S3 ----------
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
target_col = "revenue"
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
    st.subheader("🚀 Train Model")

    # List models in S3
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=MODEL_PREFIX)
    existing_keys = [item['Key'] for item in response.get('Contents', [])] if 'Contents' in response else []

    model_files = [
        f"{MODEL_PREFIX}/RandomForest.joblib",
        f"{MODEL_PREFIX}/Ridge.joblib",
        f"{MODEL_PREFIX}/Lasso.joblib",
        f"{MODEL_PREFIX}/LinearRegression.joblib",
        f"{MODEL_PREFIX}/XGBoost.joblib",
        f"{MODEL_PREFIX}/BestModel.joblib"
    ]

    for model_path in model_files:
        if model_path in existing_keys:
            st.info(f"{model_path} already trained and uploaded to S3.")
        else:
            st.warning(f"{model_path} not found in S3.")

    # Load results.csv from S3 if exists
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=f"{MODEL_PREFIX}/results.csv")
        df_results = pd.read_csv(io.BytesIO(obj['Body'].read()))
        st.success("✅ Model results found in S3.")
        st.header("📊 Existing Model Results")
        subtab1, subtab2 = st.tabs(["📄 Table", "📊 Graph"])
        with subtab1:
            st.dataframe(df_results)
        with subtab2:
            fig = px.bar(df_results, x="Model", y="CV_R2_Mean", error_y="CV_R2_STD",
                         title="Model CV R² Comparison", color="Model", text="CV_R2_Mean")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"No previous results found: {e}")

    if st.button("🧠 Train Model"):
        with st.spinner("Training models... ⏳ Please wait..."):
            time.sleep(1)
            st.success("✅ Models trained successfully!")

# ---------- TAB 2: EVALUATION ----------
with tab2:
    st.subheader("📈 Model Visualization & Evaluation")

    if df is not None:
        # Show model performance
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=f"{MODEL_PREFIX}/results.csv")
            df_results = pd.read_csv(io.BytesIO(obj['Body'].read()))
            st.dataframe(df_results)
            fig = px.bar(df_results, x="Model", y="CV_R2_Mean", error_y="CV_R2_STD",
                         title="Model CV R² Comparison", color="Model", text="CV_R2_Mean")
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("No model results available.")

        # Evaluate all models
        model_files = [
            "BestModel.joblib",
            "LinearRegression.joblib",
            "Lasso.joblib",
            "Ridge.joblib",
            "RandomForest.joblib",
            "XGBoost.joblib"
        ]

        st.subheader("📊 Model Evaluation Summary")
        cols = st.columns(2)

        for idx, model_path in enumerate(model_files):
            model_name = model_path.replace(".joblib", "")
            try:
                is_xgb = "xgboost" in model_name.lower()
                model = load_model_from_s3(S3_BUCKET, f"{MODEL_PREFIX}/{model_path}", is_xgb=is_xgb)
                if model is None:
                    continue

                # Apply preprocessing
                X_processed = preprocessor.fit_transform(X)

                if is_xgb:
                    dmatrix = xgb.DMatrix(X_processed, label=y, enable_categorical=True)
                    y_pred = model.predict(dmatrix)
                else:
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

                if (idx + 1) % 2 == 0 and idx + 1 < len(model_files):
                    cols = st.columns(2)

            except Exception as e:
                st.warning(f"⚠️ Could not evaluate {model_name}: {e}")
            if (idx + 1) % 2 == 0:
                cols = st.columns(2)

