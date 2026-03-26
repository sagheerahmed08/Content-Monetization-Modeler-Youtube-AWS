import streamlit as st
import pandas as pd
import joblib
import requests
import importlib
import io
import boto3
import plotly.express as px
import os
from datetime import datetime

# load model from model folder
def load_model_from_model(model_path, key):
    """Load model from model fodler and cache it in memory."""
    try:
        model = joblib.load(os.path.join(model_path, key))
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model from model folder: {e}")
        return None
    
def user_input_features():
    col1, col2, col3 = st.columns(3)
    views = col1.number_input("Views", 0, 10_000_000, 10_000)
    likes = col2.number_input("Likes", 0, 1_000_000, 500)
    comments = col3.number_input("Comments", 0, 1_000_000, 50)
    watch_time = col1.number_input("Watch Time (min)", 0.0, 1_000_000.0, 2000.0)
    length = col2.number_input("Video Length (min)", 0.1, 120.0, 10.0)
    subs = col3.number_input("Subscribers", 0, 10_000_000, 10000)
    category = col1.selectbox("Category", ["Entertainment", "Gaming", "Education", "Music", "Lifestyle", "Tech"])
    device = col2.selectbox("Device", ["Mobile", "Tablet", "TV", "Desktop"])
    country = col3.selectbox("Country", ["IN", "US", "CA", "UK", "AU", "DE"])

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

select_model = st.selectbox(
        "Select Model",["AdaBoostRegressor","BayesianRidge", "ElasticNet",
        "ExtraTreesRegressor","GradientBoostingRegressor","HuberRegressor",
        "Lasso","LinearRegression","PassiveAggressiveRegressor",
        "RandomForestRegressor","Ridge","SGDRegressor",
        "TheilSenRegressor","XGBRegressor"],
        key="model_select")
model_key = f"{select_model}.pkl"
model = load_model_from_model("model", model_key)
input_df = user_input_features()

pred = model.predict(input_df)
st.subheader("Predicted Ad Revenue (USD)")
st.write(f"${pred[0]:.2f} USD")
