import streamlit as st
import pandas as pd
import pathlib
import joblib
import numpy as np
import plotly.express as px
from io import BytesIO
import requests

st.set_page_config(
    page_title="YouTube Ad Revenue Predictor",
    layout="wide",
    page_icon="📊"
)

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

st.title("🎬 YouTube Ad Revenue Predictor")
st.markdown("An interactive dashboard to predict YouTube ad revenue and compare model performance.")

# Load pipeline
def load_pipeline(model_name="best_model"):
    model_file = MODEL_DIR / f"{model_name}.joblib"
    if not model_file.exists():
        st.error(f"❌ Model file not found: {model_file}")
        return None
    return joblib.load(model_file)

# User Input Section
def user_input_features():
    st.sidebar.header("📥 Input Parameters")
    views = st.sidebar.number_input("Views", min_value=0, value=10000, step=100)
    likes = st.sidebar.number_input("Likes", min_value=0, value=500, step=10)
    comments = st.sidebar.number_input("Comments", min_value=0, value=50, step=5)
    watch_time_minutes = st.sidebar.number_input("Watch Time (minutes)", min_value=0.0, value=2000.0, step=100.0)
    video_length_minutes = st.sidebar.number_input("Video Length (minutes)", min_value=0.1, value=10.0, step=0.5)
    subscribers = st.sidebar.number_input("Channel Subscribers", min_value=0, value=10000, step=100)
    category = st.sidebar.selectbox("Category", ["Entertainment", "Gaming", "Education", "Music", "News"])
    device = st.sidebar.selectbox("Device", ["Mobile", "Tablet", "TV", "Desktop"])
    country = st.sidebar.selectbox("Country", ["IN", "US", "CA", "UK"])

    data = {
        'views': views,
        'likes': likes,
        'comments': comments,
        'watch_time_minutes': watch_time_minutes,
        'video_length_minutes': video_length_minutes,
        'subscribers': subscribers,
        'category': category,
        'device': device,
        'country': country
    }
    features = pd.DataFrame(data, index=[0])
    
# Derived features
    features['engagement_rate'] = (features['likes'] + features['comments']) / features['views'].replace(0, 1)
    features['avg_watch_time_per_view'] = features['watch_time_minutes'] / features['views'].replace(0, 1)

    st.write("### 📝 Input Summary")
    st.dataframe(features, use_container_width=True, hide_index=True)
    return features

input_df = user_input_features()


# Model selection

model_files = [f.stem for f in MODEL_DIR.glob("*.joblib")]
model_name = st.sidebar.selectbox("📌 Select Model", sorted(model_files))

pipeline = load_pipeline(model_name)

#Currency Conversion
try:
    url = "https://api.fastforex.io/fetch-all?api_key=4eac62afbf-e22e0da3e1-t42ixg"
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    data = response.json()
    usd_to_inr = data.get("results", {}).get("INR", 88.73)
    st.sidebar.success(f"💱 1 USD = ₹{usd_to_inr:.2f} INR (Live rate)")
except Exception as e:
    usd_to_inr = 88.73
    st.sidebar.warning(f"⚠️ Using fallback rate: 1 USD = ₹{usd_to_inr}")
    
# Prediction & Dashboard
  
if pipeline:
    prediction = pipeline.predict(input_df)[0]
    if prediction < 0:
        prediction = 0  
    prediction_inr = prediction * usd_to_inr
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👀 Views",input_df['views'])
    col2.metric("👍 Likes",input_df['likes'])
    col3.metric(f"⏱ Watch Time ",f"{input_df['watch_time_minutes'].iloc[0]} mins")
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
        💰 Predicted Revenue<br>₹{prediction_inr:,.2f}
        <span style="font-size:0.8em; color:#155724;">(${prediction:,.2f})</span>
    </div>
    """,
    unsafe_allow_html=True
)
    st.subheader("📊 Feature Insights")
    st.bar_chart(input_df[['views', 'likes', 'comments', 'watch_time_minutes', 'subscribers']].T)

    # Export predictions into CSV
    export_df = input_df.copy()
    export_df['predicted_revenue_usd'] = prediction

    # CSV download
    csv_data = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Prediction data",
        data=csv_data,
        file_name="prediction.csv",
        mime="text/csv"
    )

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
