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
import tempfile
import os

# ----------------------------
# Optional XGBoost
# ----------------------------
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

def upload_model_to_s3(pipe, bucket, key, is_xgb=False):
    """Upload pipeline/model to S3 (supports XGBoost)"""
    buffer = io.BytesIO()
    if is_xgb and has_xgb:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
            booster = pipe.named_steps['model'].get_booster()
            booster.save_model(tmp_file.name)
            tmp_file.close()
            with open(tmp_file.name, "rb") as f:
                buffer.write(f.read())
            os.remove(tmp_file.name)
    else:
        joblib.dump(pipe, buffer)
    buffer.seek(0)
    s3.upload_fileobj(buffer, bucket, key)

def load_model_from_s3(bucket, key, is_xgb=False):
    obj = s3.get_object(Bucket=bucket, Key=key)
    if is_xgb and has_xgb:
        model = XGBRegressor()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
            tmp_file.write(obj['Body'].read())
            tmp_file.flush()
            model.load_model(tmp_file.name)
        os.remove(tmp_file.name)
        return model
    else:
        return joblib.load(io.BytesIO(obj['Body'].read()))

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

if df is not None:
    df['engagement_rate'] = (df['likes'] + df['comments']) / df['views'].replace(0,1)
    df['avg_watch_time_per_view'] = df['watch_time_minutes'] / df['views'].replace(0,1)

    num_features = ['views', 'comments','video_length_minutes','subscribers','engagement_rate','avg_watch_time_per_view']
    cat_features = ['category','device','country']

    X = df[num_features + cat_features]
    y = df['ad_revenue_usd'].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2 = st.tabs(["Model Training", "Model Visualization"])

# ============================
# TAB 1: MODEL TRAINING
# ============================
with tab1:
    st.subheader("🚀 Train Model")

    # --- Check existing models in S3 ---
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=MODEL_PREFIX)
        existing_keys = [item['Key'] for item in response.get('Contents', [])]

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
                st.info(f"✅ {model_path} already uploaded to S3.")
            else:
                st.warning(f"⚠️ {model_path} not found in S3.")
    except Exception as e:
        st.error(f"❌ Failed to list S3 models: {e}")

    # --- Load results.csv ---
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=f"{MODEL_PREFIX}/results.csv")
        df_results = pd.read_csv(io.BytesIO(obj['Body'].read()))
        st.success("📊 Found existing results in S3")
        st.dataframe(df_results)

        fig = px.bar(df_results, x="Model", y="CV_R2_Mean", error_y="CV_R2_STD",
                     title="Model CV R² Comparison", color="Model", text="CV_R2_Mean")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("No results.csv found in S3 yet.")

    st.subheader("🧠 Train & Upload Models")

    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(random_state=42, alpha=1.0),
        'Lasso': Lasso(random_state=42, alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    }
    if has_xgb:
        models['XGBoost'] = XGBRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=1,
            objective='reg:squarederror',
            learning_rate=0.1,
            max_depth=6
        )

    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('num', num_pipeline, num_features), ('cat', cat_pipeline, cat_features)])

    results = {}

    if st.button("🧠 Train Model"):
        with st.spinner("Training models... ⏳ Please wait..."):
            for name, model in models.items():
                st.write(f"🔹 Training {name}...")
                pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
                pipe.fit(X_train, y_train)

                upload_model_to_s3(pipe, S3_BUCKET, f"{MODEL_PREFIX}/{name}.joblib", is_xgb=(name=='XGBoost'))
                st.success(f"{name} uploaded to S3")

                scores = cross_val_score(pipe, X_train, y_train, scoring='r2', cv=5, n_jobs=-1)
                results[name] = {'cv_r2_mean': np.mean(scores), 'cv_r2_std': np.std(scores)}
                st.write(f"{name}: CV R² = {np.mean(scores):.4f} ± {np.std(scores):.4f}")

            best_name = max(results, key=lambda k: results[k]['cv_r2_mean'])
            st.success(f"🏆 Best Model: {best_name} (CV R² = {results[best_name]['cv_r2_mean']:.4f})")

            best_model = load_model_from_s3(S3_BUCKET, f"{MODEL_PREFIX}/{best_name}.joblib", is_xgb=(best_name=='XGBoost'))
            upload_model_to_s3(best_model, S3_BUCKET, f"{MODEL_PREFIX}/BestModel.joblib", is_xgb=(best_name=='XGBoost'))

            perf_df = pd.DataFrame([{"Model": k, "CV_R2_Mean": v['cv_r2_mean'], "CV_R2_STD": v['cv_r2_std']} for k,v in results.items()])
            csv_buffer = io.StringIO()
            perf_df.to_csv(csv_buffer, index=False)
            s3.put_object(Bucket=S3_BUCKET, Key=f"{MODEL_PREFIX}/results.csv", Body=csv_buffer.getvalue())
            st.success("🎉 Training and upload completed!")

# ============================
# TAB 2: MODEL VISUALIZATION
# ============================
with tab2:
    st.subheader("📈 Model Visualization & Evaluation")

    if df is not None:
        # Show model performance table
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=f"{MODEL_PREFIX}/results.csv")
            df_results = pd.read_csv(io.BytesIO(obj['Body'].read()))
            st.dataframe(df_results)
            fig = px.bar(
                df_results, x="Model", y="CV_R2_Mean", error_y="CV_R2_STD",
                title="Model CV R² Comparison", color="Model", text="CV_R2_Mean"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
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
                is_xgb = has_xgb and "xgboost" in model_name.lower()
                model = load_model_from_s3(S3_BUCKET, f"{MODEL_PREFIX}/{model_path}", is_xgb=is_xgb)

                if model is None:
                    st.warning(f"{model_name} could not be loaded.")
                    continue

                # Predict directly — no preprocessing here, since pipeline already includes it
                y_pred = model.predict(X)

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

