import streamlit as st
import pandas as pd
import joblib
import requests
import importlib
import io
import boto3
import plotly.express as px

#Try importing XGBoost
try:
    _xgb = importlib.import_module("xgboost")
    XGBRegressor = getattr(_xgb, "XGBRegressor")
    has_xgb = True
except Exception:
    has_xgb = False

# Streamlit Config
st.set_page_config(
    page_title="Prediction | YouTube Ad Revenue Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Predict YouTube Ad Revenue")

# AWS S3 Configuration
S3_BUCKET = "youtube-ad-revenue-app-sagheer"
CLEAN_KEY = "Data/Cleaned/youtube_ad_revenue_dataset_cleaned.csv"
MODEL_PREFIX = "models"

import boto3
import streamlit as st

s3 = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
    region_name="eu-north-1"
)

# Get USD to INR conversion
@st.cache_data
def get_usd_to_inr():
    try:
        res = requests.get(
            "https://v6.exchangerate-api.com/v6/01dd1651e64cb8df5a89b465/latest/USD"
        ).json()
        return res["conversion_rates"]["INR"]
    except:
        return 88.70

# Load model directly from S3
def load_model_from_s3(bucket, key):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        model_bytes = io.BytesIO(obj["Body"].read())
        model = joblib.load(model_bytes)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model from S3: {e}")
        return None

# 🧾 User Input
def user_input_features():
    col1, col2, col3 = st.columns(3)
    views = col1.number_input("Views", 0, 10_000_000, 10_000)
    likes = col2.number_input("Likes", 0, 1_000_000, 500)
    comments = col3.number_input("Comments", 0, 1_000_000, 50)
    watch_time = col1.number_input("Watch Time (min)", 0.0, 1_000_000.0, 2000.0)
    length = col2.number_input("Video Length (min)", 0.1, 120.0, 10.0)
    subs = col3.number_input("Subscribers", 0, 10_000_000, 10000)
    category = col1.selectbox("Category", ["Entertainment", "Gaming", "Education", "Music", "News"])
    device = col2.selectbox("Device", ["Mobile", "Tablet", "TV", "Desktop"])
    country = col3.selectbox("Country", ["IN", "US", "CA", "UK"])

    df = pd.DataFrame({
        "views": [views],
        "likes": [likes],
        "comments": [comments],
        "watch_time_minutes": [watch_time],
        "video_length_minutes": [length],
        "subscribers": [subs],
        "category": [category],
        "device": [device],
        "country": [country],
    })
    df["engagement_rate"] = (df["likes"] + df["comments"]) / df["views"].replace(0, 1)
    df["avg_watch_time_per_view"] = df["watch_time_minutes"] / df["views"].replace(0, 1)
    return df

# ---------------------------------------------------------------------
# 🚀 Main Prediction Logic
# ---------------------------------------------------------------------
usd_to_inr = get_usd_to_inr()
st.success(f"💱 Current Exchange Rate: 1 USD = ₹{usd_to_inr:.2f} INR")

select_model = st.selectbox(
    "Select Model",
    ["BestModel", "LinearRegression", "Ridge", "Lasso", "RandomForest", "XGBoost"],
    key="model_select"
)

model_key = f"{MODEL_PREFIX}/{select_model}.joblib"
model = load_model_from_s3(S3_BUCKET, model_key)

df = user_input_features()
st.subheader("📋 Input Features")
st.dataframe(df, use_container_width=True, hide_index=True)

if model:
    pred = model.predict(df)[0]
    pred_inr = pred * usd_to_inr

    st.success(f"Predicted Ad Revenue: ₹{pred_inr:,.2f} (≈ ${pred:,.2f})")

    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👀 Views", int(df['views'].iloc[0]))
    col2.metric("👍 Likes", int(df['likes'].iloc[0]))
    col3.metric("⏱ Watch Time", f"{df['watch_time_minutes'].iloc[0]} min")
    col4.markdown(
        f"""
        <div style="
            background-color:#d4edda;
            color:#155724;
            padding:10px;
            border-radius:5px;
            text-align:center;
            font-size:1.2em;
            font-weight:bold;">
            💰 Predicted Revenue<br>₹{pred_inr:,.2f}
            <span style="font-size:0.8em;">(${pred:,.2f})</span>
        </div>
        """,
        unsafe_allow_html=True
)


    st.subheader("📊 Feature Insights")
    st.bar_chart(df[['views', 'likes', 'comments', 'watch_time_minutes',
                     'subscribers', 'engagement_rate', 'avg_watch_time_per_view']].T)

    # Export CSV
    export_df = df.copy()
    export_df["predicted_revenue_usd"] = pred
    export_df["predicted_revenue_inr"] = pred_inr
    csv_data = export_df.to_csv(index=False).encode("utf-8")

    if st.download_button(
        "📥 Download Prediction Data",
        data=csv_data,
        file_name="prediction.csv",
        mime="text/csv"
    ):
        st.success("✅ Prediction data downloaded successfully!")

# ---------------------------------------------------------------------
# 📈 Model Performance Comparison
# ---------------------------------------------------------------------
st.subheader("📈 Model Performance Comparison")

try:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=f"{MODEL_PREFIX}/results.csv")
    perf_df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        perf_df,
        x="Model",
        y="CV_R2_Mean",
        error_y="CV_R2_STD",
        title="Model Cross-Validation R² Comparison",
        text="CV_R2_Mean"
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.info(f"ℹ️ Model performance data not found or cannot be accessed.\n{e}")
