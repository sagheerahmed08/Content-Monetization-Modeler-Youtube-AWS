import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px

# Streamlit Config
st.set_page_config(
    page_title="📊 EDA | YouTube Ad Revenue Predictor",
    page_icon="📊",
    layout="wide"
)
st.title("📊 Exploratory Data Analysis")


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(BASE_DIR, "Data", "Raw", "youtube_ad_revenue_dataset.csv")
CLEAN = os.path.join(BASE_DIR, "Data", "Cleaned", "youtube_ad_revenue_dataset_cleaned.csv")

@st.cache_data
def load_data():
    raw = pd.read_csv(RAW)
    clean = pd.read_csv(CLEAN)
    return raw, clean

raw_df, clean_df = load_data()

#  Overview 
st.subheader("Dataset Overview")
col1, col2 = st.columns(2)
with col1:
    st.header("Before Cleaning")
    st.warning(f"{raw_df.shape[0]} Rows × {raw_df.shape[1]} Columns")
with col2:
    st.header("After Cleaning")
    st.success(f"{clean_df.shape[0]} Rows × {clean_df.shape[1]} Columns")

tab1, tab2 = st.tabs([ "Visualize Data", "Raw Data"])
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
        "Clean Missing": clean_df[common_cols].isna().sum().values
    })
    st.dataframe(missing_df, use_container_width=True, hide_index=True)


#Category Insights 
st.header("📊 Category-Based Insights")

@st.cache_data
def compute_category_summaries(df):
    grouped = df.groupby("category")[["likes", "comments", "watch_time_minutes"]].sum().reset_index()
    return grouped

raw_summary = compute_category_summaries(raw_df)
clean_summary = compute_category_summaries(clean_df)
missing_df = pd.DataFrame({
    "Category": raw_df["category"].unique()
})

def compute_missing_likes(df):
    return df.groupby("category")["likes"].apply(lambda x: x.isna().sum())

def compute_missing_comments(df):
    return df.groupby("category")["comments"].apply(lambda x: x.isna().sum())

def compute_missing_watch_time(df):
    return df.groupby("category")["watch_time_minutes"].apply(lambda x: x.isna().sum())

tab4, tab5 = st.tabs([ "Visualize Data", "Raw Data"])
with tab4:
    tab1, tab2, tab3 = st.tabs(["Likes", "Comments", "Watch Time"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.pie(raw_summary, values="likes", names="category",
                            title="Likes per Category before cleaning", color="category", hover_name="category")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.pie(clean_summary, values="likes", names="category",
                        title="Likes per Category after Cleaning", color="category", hover_name="category")
            st.plotly_chart(fig2, use_container_width=True)
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig3 = px.pie(raw_summary, values="comments", names="category",
                        title="Comments per Category before cleaning", color="category", hover_name="category")
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            fig4 = px.pie(clean_summary, values="comments", names="category",
                        title="Comments per Category after Cleaning", color="category", hover_name="category")
            st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig5 = px.pie(raw_summary, values="watch_time_minutes", names="category",
                        title="Watch Time per Category before cleaning", color="category", hover_name="category")
            st.plotly_chart(fig5, use_container_width=True)
        with col2:
            fig6 = px.pie(clean_summary, values="watch_time_minutes", names="category",
                        title="Watch Time per Category after Cleaning", color="category", hover_name="category")
            st.plotly_chart(fig6, use_container_width=True)

with tab5:
    missing_df = pd.DataFrame({
        "Category": compute_missing_likes(raw_df).index,
        "Raw Missing (likes)": compute_missing_likes(raw_df),
        "Clean Missing (likes)": compute_missing_likes(clean_df),
        "Raw Missing (comments)": compute_missing_comments(raw_df),
        "Clean Missing (comments)": compute_missing_comments(clean_df),
        "Raw Missing (watch time)": compute_missing_watch_time(raw_df),
        "Clean Missing (watch time)": compute_missing_watch_time(clean_df)
        }).reset_index(drop=True)
    total_row = pd.DataFrame({
        "Category": ["Total"],
        "Raw Missing (likes)": [missing_df["Raw Missing (likes)"].sum()],
        "Clean Missing (likes)": [missing_df["Clean Missing (likes)"].sum()],
        "Raw Missing (comments)": [missing_df["Raw Missing (comments)"].sum()],
        "Clean Missing (comments)": [missing_df["Clean Missing (comments)"].sum()],
        "Raw Missing (watch time)": [missing_df["Raw Missing (watch time)"].sum()],
        "Clean Missing (watch time)": [missing_df["Clean Missing (watch time)"].sum()]
    })
    missing_df = pd.concat([missing_df, total_row], ignore_index=True)
    st.dataframe(missing_df, use_container_width=True, hide_index=True)


# Target Distribution
st.header("🎯 Target Distribution (Ad Revenue USD)")
column = st.selectbox(
    "Select the column to visualize",
    options=clean_df.columns,
    index=clean_df.columns.get_loc("ad_revenue_usd")
)

col1, col2 = st.columns(2)
with col2:
    fig = px.histogram(clean_df, x=column, nbins=80, title=f"{column} Distribution after Cleaning")
    st.plotly_chart(fig, use_container_width=True)

with col1:
    if column in raw_df.columns:
        fig = px.histogram(raw_df, x=column, nbins=80, title=f"{column} Distribution before Cleaning")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"{column} not found in raw data.")

