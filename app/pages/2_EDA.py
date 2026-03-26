import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import boto3
import io
from scipy.stats import skew, kurtosis

from config import S3_BUCKET, RAW_KEY, CLEAN_KEY

st.set_page_config(page_title="EDA | YouTube Ad Revenue Predictor", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")

s3 = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
    region_name=st.secrets["aws"]["region"],
)


@st.cache_data
def load_data_from_s3():
    """Load raw and cleaned datasets from S3."""
    raw_obj = s3.get_object(Bucket=S3_BUCKET, Key=RAW_KEY)
    clean_obj = s3.get_object(Bucket=S3_BUCKET, Key=CLEAN_KEY)
    raw = pd.read_csv(io.BytesIO(raw_obj["Body"].read()))
    clean = pd.read_csv(io.BytesIO(clean_obj["Body"].read()))
    return raw, clean


try:
    raw_df, clean_df = load_data_from_s3()
except Exception as e:
    st.error(f"Failed to load data from S3: {e}")
    st.stop()

# Overview
st.subheader("Dataset Overview")
col1, col2 = st.columns(2)
with col1:
    st.header("Before Cleaning")
    st.warning(f"{raw_df.shape[0]} Rows × {raw_df.shape[1]} Columns")
with col2:
    st.header("After Cleaning")
    st.success(f"{clean_df.shape[0]} Rows × {clean_df.shape[1]} Columns")

tab1, tab2 = st.tabs(["Visualize Data", "Raw Data"])
with tab1:
    st.header("Missing Values Comparison")
    sample_raw = raw_df.sample(min(500, len(raw_df)))
    sample_clean = clean_df.sample(min(500, len(clean_df)))

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        sns.heatmap(sample_raw.isna(), cbar=False, ax=ax1, cmap="Reds")
        ax1.set_title("Raw Data Missing Values (sample)")
        st.pyplot(fig1, use_container_width=True)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        sns.heatmap(sample_clean.isna(), cbar=False, ax=ax2, cmap="Greens")
        ax2.set_title("Clean Data Missing Values (sample)")
        st.pyplot(fig2, use_container_width=True)

with tab2:
    common_cols = raw_df.columns.intersection(clean_df.columns)
    missing_df = pd.DataFrame({
        "Column": common_cols,
        "Raw Missing": raw_df[common_cols].isna().sum().values,
        "Clean Missing": clean_df[common_cols].isna().sum().values,
    })
    st.dataframe(missing_df, use_container_width=True, hide_index=True)


# Category Insights
st.header("📊 Category-Based Insights")


@st.cache_data
def compute_category_summaries(df):
    return df.groupby("category")[["likes", "comments", "watch_time_minutes"]].sum().reset_index()


raw_summary = compute_category_summaries(raw_df)
clean_summary = compute_category_summaries(clean_df)

tab4, tab5 = st.tabs(["Visualize Data", "Raw Data"])
with tab4:
    tab_likes, tab_comments, tab_watch = st.tabs(["Likes", "Comments", "Watch Time"])
    with tab_likes:
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.pie(raw_summary, values="likes", names="category",
                          title="Likes per Category before Cleaning", hover_name="category")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.pie(clean_summary, values="likes", names="category",
                          title="Likes per Category after Cleaning", hover_name="category")
            st.plotly_chart(fig2, use_container_width=True)
    with tab_comments:
        col1, col2 = st.columns(2)
        with col1:
            fig3 = px.pie(raw_summary, values="comments", names="category",
                          title="Comments per Category before Cleaning", hover_name="category")
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            fig4 = px.pie(clean_summary, values="comments", names="category",
                          title="Comments per Category after Cleaning", hover_name="category")
            st.plotly_chart(fig4, use_container_width=True)
    with tab_watch:
        col1, col2 = st.columns(2)
        with col1:
            fig5 = px.pie(raw_summary, values="watch_time_minutes", names="category",
                          title="Watch Time per Category before Cleaning", hover_name="category")
            st.plotly_chart(fig5, use_container_width=True)
        with col2:
            fig6 = px.pie(clean_summary, values="watch_time_minutes", names="category",
                          title="Watch Time per Category after Cleaning", hover_name="category")
            st.plotly_chart(fig6, use_container_width=True)

with tab5:
    cols_of_interest = ["likes", "comments", "watch_time_minutes"]
    raw_miss = raw_df.groupby("category")[cols_of_interest].apply(lambda x: x.isna().sum())
    clean_miss = clean_df.groupby("category")[cols_of_interest].apply(lambda x: x.isna().sum())
    missing_detail = pd.DataFrame({
        "Category": raw_miss.index,
        "Raw Missing (likes)": raw_miss["likes"].values,
        "Clean Missing (likes)": clean_miss["likes"].values,
        "Raw Missing (comments)": raw_miss["comments"].values,
        "Clean Missing (comments)": clean_miss["comments"].values,
        "Raw Missing (watch time)": raw_miss["watch_time_minutes"].values,
        "Clean Missing (watch time)": clean_miss["watch_time_minutes"].values,
    })
    total_row = missing_detail.drop(columns="Category").sum().to_frame().T
    total_row.insert(0, "Category", "Total")
    missing_detail = pd.concat([missing_detail, total_row], ignore_index=True)
    st.dataframe(missing_detail, use_container_width=True, hide_index=True)


# Numeric Summary & Stats
st.header("📈 Summary Statistics")
numeric_cols = clean_df.select_dtypes(include=["int64", "float64"]).columns
summary_df = clean_df[numeric_cols].describe().T
summary_df["Skewness"] = clean_df[numeric_cols].apply(skew)
summary_df["Kurtosis"] = clean_df[numeric_cols].apply(kurtosis)
st.dataframe(summary_df, use_container_width=True)


# Correlation Heatmap
st.header("🔗 Feature Correlation")
corr = clean_df[numeric_cols].corr()
fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", title="Feature Correlation Heatmap")
st.plotly_chart(fig_corr, use_container_width=True)


# Target Distribution
st.header("🎯 Target Distribution (Ad Revenue USD)")
column = st.selectbox(
    "Select the column to visualize",
    options=clean_df.columns,
    index=clean_df.columns.get_loc("ad_revenue_usd"),
)

col1, col2 = st.columns(2)
with col1:
    if column in raw_df.columns:
        fig = px.histogram(raw_df, x=column, nbins=80, title=f"{column} Distribution before Cleaning")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"{column} not found in raw data.")
with col2:
    fig = px.histogram(clean_df, x=column, nbins=80, title=f"{column} Distribution after Cleaning")
    st.plotly_chart(fig, use_container_width=True)


# Outlier Detection
st.header("📦 Outlier Detection (Boxplots)")
selected_col = st.selectbox("Select feature for outlier visualization", options=numeric_cols)
fig = px.box(clean_df, x=selected_col, title=f"Boxplot for {selected_col}")
st.plotly_chart(fig, use_container_width=True)


# Feature vs Target Relationship
st.header("📈 Feature vs Target Relationship")
target_col = "ad_revenue_usd"
feature = st.selectbox("Select Feature", options=[c for c in numeric_cols if c != target_col])
fig = px.scatter(clean_df, x=feature, y=target_col, trendline="ols",
                 title=f"{feature} vs Ad Revenue (USD)")
st.plotly_chart(fig, use_container_width=True)


# Key Insights
st.header("💡 Key Insights")
top_corr = (
    corr["ad_revenue_usd"]
    .drop("ad_revenue_usd")
    .abs()
    .sort_values(ascending=False)
)
top_feature = top_corr.index[0]
top_value = top_corr.iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric("Strongest Revenue Driver", top_feature, f"r = {top_value:.3f}")
col2.metric("Avg Ad Revenue (USD)", f"${clean_df['ad_revenue_usd'].mean():.2f}")
col3.metric("Median Ad Revenue (USD)", f"${clean_df['ad_revenue_usd'].median():.2f}")

skew_val = skew(clean_df["ad_revenue_usd"].dropna())
st.info(
    f"The revenue distribution is {'right' if skew_val > 0 else 'left'}-skewed "
    f"(skewness = {skew_val:.2f}), meaning a small number of videos earn "
    f"disproportionately high revenue. Focus on maximizing **{top_feature}** "
    f"to improve monetisation potential."
)

st.success("EDA completed successfully.")
