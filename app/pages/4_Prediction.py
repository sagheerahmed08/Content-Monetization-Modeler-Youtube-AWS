import streamlit as st
import pandas as pd
import pathlib
import joblib
import requests
import importlib
import plotly.express as px
try:
    _xgb = importlib.import_module("xgboost")
    XGBRegressor = getattr(_xgb, "XGBRegressor")
    has_xgb = True
except Exception:
    # xgboost not installed or failed to import; downstream code will skip XGBoost model
    has_xgb = False
st.set_page_config(page_title="Prediction | YouTube Ad Revenue Predictor", page_icon="📊",layout="wide")

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

st.title("📊     Predict YouTube Ad Revenue")
@st.cache_data
def get_usd_to_inr():
    try:
        res = requests.get("https://v6.exchangerate-api.com/v6/01dd1651e64cb8df5a89b465/latest/USD").json()
        return res["conversion_rates"]["INR"]
    except:
        return 88.70
def load_model(name):
    model_file = MODEL_DIR / f"{name}.joblib"
    if not model_file.exists():
        st.error(f"❌ Model file not found: {model_file}")
        return None
    return joblib.load(model_file)
def user_input_features():
    col1,col2,col3 = st.columns(3)
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

usd_to_inr = get_usd_to_inr()

select_model = st.selectbox("Select Model", ["best_model", "LinearRegression", "Ridge", "Lasso", "RandomForest", "XGBoost"], key="model_select")
model = load_model(select_model)
st.success(f"💱 Current Exchange Rate: 1 USD = ₹{usd_to_inr:.2f} INR")

df = user_input_features()
st.subheader("Input Features")  
st.dataframe(df, use_container_width=True, hide_index=True)


model = load_model(select_model)
pred = model.predict(df)[0]
pred_inr = pred * usd_to_inr

st.success(f"Predicted Ad Revenue: ₹{pred_inr:,.2f} (≈ ${pred:,.2f})")

col1, col2, col3, col4 = st.columns(4)
col1.metric("👀 Views",df['views'])
col2.metric("👍 Likes",df['likes'])
col3.metric(f"⏱ Watch Time ",f"{df['watch_time_minutes'].iloc[0]} mins")
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
        <span style="font-size:0.8em; color:#155724;">(${pred:,.2f})</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.subheader("📊 Feature Insights")
st.bar_chart(df[['views', 'likes', 'comments', 'watch_time_minutes', 'subscribers','engagement_rate','avg_watch_time_per_view']].T)

# Export predictions into CSV
export_df = df.copy()
export_df['predicted_revenue_usd'] = pred
export_df['predicted_revenue_inr'] = pred_inr   

# CSV download
csv_data = export_df.to_csv(index=False).encode('utf-8')
s=st.download_button(
        "📥 Download Prediction data",
        data=csv_data,
        file_name="prediction.csv",
        mime="text/csv"
)
if s:
    st.success("✅ Prediction data downloaded successfully!")

# Model Performance Comparison
st.subheader("📈 Model Performance Comparison")
results_file = MODEL_DIR / "results.csv"
if results_file.exists():
    perf_df = pd.read_csv(results_file)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    fig = px.bar(perf_df, x="Model",
                 y="CV_R2_Mean", 
                 error_y="CV_R2_STD",
                 title="Model Cross-Validation R² Comparison",
                 text="CV_R2_Mean")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Model comparison file not found. Train models to generate performance metrics.")