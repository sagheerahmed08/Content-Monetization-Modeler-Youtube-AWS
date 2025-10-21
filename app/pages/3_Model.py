# pages/3_Model.py
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

# ----------------------------
# S3 Config
# ----------------------------
S3_BUCKET = "youtube-ad-revenue-app-sagheer"
CLEAN_KEY = "Data/Cleaned/youtube_ad_revenue_dataset_cleaned.csv"
MODEL_PREFIX = "models"

# create s3 client using secrets (must exist in Streamlit secrets)
s3 = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
    region_name=st.secrets["aws"]["region"]
)

# ----------------------------
# Utility Functions
# ----------------------------
@st.cache_data
def load_csv_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))

def upload_model_to_s3(pipe, bucket, key):
    """
    Save pipeline (preprocessor + model) to S3 using joblib.
    We ALWAYS save the pipeline with joblib to avoid shape mismatches.
    """
    buffer = io.BytesIO()
    joblib.dump(pipe, buffer)
    buffer.seek(0)
    s3.upload_fileobj(buffer, bucket, key)

def load_model_from_s3(bucket, key):
    """
    Load pipeline from S3 (joblib).
    Returns the deserialized object or None on error.
    """
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return joblib.load(io.BytesIO(obj['Body'].read()))
    except Exception as e:
        st.warning(f"Could not load {key} from S3: {e}")
        return None

def eval_metrics(y_true, y_pred):
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }

# ----------------------------
# Load Dataset
# ----------------------------
try:
    df = load_csv_from_s3(S3_BUCKET, CLEAN_KEY)
    st.success(f"✅ Dataset loaded from S3: {df.shape}")
except Exception as e:
    st.error(f"❌ Failed to load dataset from S3: {e}")
    df = None

# ----------------------------
# Feature Engineering & columns
# ----------------------------
if df is not None:
    # Derived features
    df['engagement_rate'] = (df['likes'].fillna(0) + df['comments'].fillna(0)) / df['views'].replace(0, 1)
    df['avg_watch_time_per_view'] = df['watch_time_minutes'].fillna(0) / df['views'].replace(0, 1)

    # Define features used in the app (9 raw features)
    num_features = ['views', 'comments', 'video_length_minutes', 'subscribers', 'engagement_rate', 'avg_watch_time_per_view']
    cat_features = ['category', 'device', 'country']

    X = df[num_features + cat_features].copy()
    y = df['ad_revenue_usd'].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    # placeholders so code later doesn't break
    num_features = ['views', 'comments', 'video_length_minutes', 'subscribers', 'engagement_rate', 'avg_watch_time_per_view']
    cat_features = ['category', 'device', 'country']
    X = pd.DataFrame(columns=num_features + cat_features)
    y = pd.Series(dtype=float)

# ----------------------------
# Preprocessor defined at module scope (so available later)
# ----------------------------
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2 = st.tabs(["🚀 Train Model", "📈 Model Visualization & Evaluation"])

# ============================
# TAB 1: TRAINING
# ============================
with tab1:
    st.subheader("🚀 Train Model")

    # show which models exist in S3
    try:
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=MODEL_PREFIX)
        existing_keys = [item['Key'] for item in resp.get('Contents', [])] if resp.get('Contents') else []
    except Exception as e:
        existing_keys = []
        st.warning(f"Could not list S3 models: {e}")

    expected_paths = [
        f"{MODEL_PREFIX}/RandomForest.joblib",
        f"{MODEL_PREFIX}/Ridge.joblib",
        f"{MODEL_PREFIX}/Lasso.joblib",
        f"{MODEL_PREFIX}/LinearRegression.joblib",
        f"{MODEL_PREFIX}/XGBoost.joblib",
        f"{MODEL_PREFIX}/BestModel.joblib"
    ]
    for p in expected_paths:
        if p in existing_keys:
            st.info(f"✅ {p} already uploaded.")
        else:
            st.warning(f"⚠️ {p} not found in S3.")

    # results.csv (if present)
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=f"{MODEL_PREFIX}/results.csv")
        df_results = pd.read_csv(io.BytesIO(obj['Body'].read()))
        st.success("✅ Model results found in S3.")
        st.header("📊 Existing Model Results")
        subtab1, subtab2 = st.tabs(["📋 Table", "📊 Chart"])
        with subtab1:
            st.dataframe(df_results, use_container_width=True)
        with subtab2:
            fig = px.bar(df_results, x="Model", y="CV_R2_Mean", error_y="CV_R2_STD",
                         title="Model CV R² Comparison", color="Model", text="CV_R2_Mean")
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("No previous results.csv in S3 (or cannot access).")

    # training button
    st.subheader("🧠 Train & Upload Models")
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=1.0, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    }
    if has_xgb:
        # treat XGBoost like other models inside a pipeline
        models['XGBoost'] = XGBRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror',
            learning_rate=0.1,
            max_depth=6
        )

    if st.button("🧠 Train Models"):
        if df is None:
            st.error("No dataset loaded; cannot train.")
        else:
            with st.spinner("Training models... ⏳ Please wait..."):
                results = {}
                for name, model in models.items():
                    st.write(f"🔹 Training {name}...")
                    # create pipeline that includes preprocessor AND model
                    pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
                    pipe.fit(X_train, y_train)

                    # upload pipeline to S3 (always saving pipeline via joblib)
                    upload_model_to_s3(pipe, S3_BUCKET, f"{MODEL_PREFIX}/{name}.joblib")
                    st.success(f"{name} uploaded to S3")

                    # cross-validation (use pipeline)
                    try:
                        scores = cross_val_score(pipe, X_train, y_train, scoring='r2', cv=5, n_jobs=-1)
                        results[name] = {'cv_r2_mean': np.mean(scores), 'cv_r2_std': np.std(scores)}
                        st.write(f"{name}: CV R² = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
                    except Exception as e:
                        st.warning(f"CV failed for {name}: {e}")
                        results[name] = {'cv_r2_mean': float('nan'), 'cv_r2_std': float('nan')}

                # pick best by CV mean (ignore NaN)
                valid_results = {k: v for k, v in results.items() if not np.isnan(v['cv_r2_mean'])}
                if valid_results:
                    best_name = max(valid_results, key=lambda k: valid_results[k]['cv_r2_mean'])
                else:
                    best_name = max(results, key=lambda k: (results[k]['cv_r2_mean'] if not np.isnan(results[k]['cv_r2_mean']) else -np.inf))

                st.success(f"🏆 Best Model: {best_name}")

                # load best model pipeline from S3 and save as BestModel.joblib (pipeline)
                best_pipe = load_model_from_s3(S3_BUCKET, f"{MODEL_PREFIX}/{best_name}.joblib")
                if best_pipe is not None:
                    upload_model_to_s3(best_pipe, S3_BUCKET, f"{MODEL_PREFIX}/BestModel.joblib")
                    st.success("BestModel.joblib uploaded to S3")

                # save results.csv to S3
                perf_df = pd.DataFrame([{"Model": k, "CV_R2_Mean": v['cv_r2_mean'], "CV_R2_STD": v['cv_r2_std']} for k, v in results.items()])
                csv_buffer = io.StringIO()
                perf_df.to_csv(csv_buffer, index=False)
                s3.put_object(Bucket=S3_BUCKET, Key=f"{MODEL_PREFIX}/results.csv", Body=csv_buffer.getvalue())
                st.success("🎉 Training finished and results.csv uploaded!")

# ============================
# TAB 2: VISUALIZATION
# ============================
with tab2:
    st.subheader("📈 Model Visualization & Evaluation")

    if df is None:
        st.warning("No data loaded – please upload/correct S3 dataset.")
    else:
        # show results.csv if present
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=f"{MODEL_PREFIX}/results.csv")
            df_results = pd.read_csv(io.BytesIO(obj['Body'].read()))
            st.dataframe(df_results, use_container_width=True)
            fig = px.bar(df_results, x="Model", y="CV_R2_Mean", error_y="CV_R2_STD", title="Model CV R² Comparison", color="Model", text="CV_R2_Mean")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("No results.csv available.")

        # Evaluate saved models
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
            with st.expander(f"{model_name} evaluation", expanded=False):
                # load model (pipeline)
                model = load_model_from_s3(S3_BUCKET, f"{MODEL_PREFIX}/{model_path}")
                if model is None:
                    st.warning(f"{model_name} is not available in S3.")
                    continue

                try:
                    # For saved pipelines we can call predict directly on raw X
                    y_pred = model.predict(X)
                    metrics = eval_metrics(y, y_pred)

                    # Display in layout column
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
                except Exception as e:
                    st.warning(f"⚠️ Could not evaluate {model_name}: {e}")

            # new row every 2 models
            if (idx + 1) % 2 == 0:
                cols = st.columns(2)

