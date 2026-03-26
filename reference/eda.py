import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import boto3
from scipy.stats import skew, kurtosis
from io import StringIO

st.set_page_config(page_title="📊 EDA | YouTube Ad Revenue Predictor", page_icon="📊", layout="wide")

aws_access_key = st.secrets["aws"]["aws_access_key_id"]
aws_secret_key = st.secrets["aws"]["aws_secret_access_key"]
aws_region = st.secrets["aws"]["region"]

S3_BUCKET = "youtube-ad-revenue-app-sagheer"
RAW_KEY = "Data/Raw/youtube_ad_revenue_dataset.csv"
CLEAN_KEY = "Data/Cleaned/youtube_ad_revenue_dataset_cleaned.csv"

# Load CSV from S3
@st.cache_data
def load_data_from_s3():
    # Create S3 client
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )

    def read_s3_csv(key):
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        return pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

    raw_df = read_s3_csv(RAW_KEY)
    clean_df = read_s3_csv(CLEAN_KEY)
    return raw_df, clean_df

#Load Data
raw_df, clean_df = load_data_from_s3()
st.success("Data loaded from S3 successfully!")
col1, col2 = st.columns(2)

with col1:
    st.dataframe(raw_df.columns)
    st.write(raw_df.describe())
    st.write(raw_df.dtypes)
with col2:
    st.dataframe(clean_df.columns)
    st.write(clean_df.describe())
    st.write(clean_df.dtypes)   
    
drop_columns = ['video_id','date']
st.success(f"Dropping columns: {drop_columns} due to irrelevance.")
raw_df = raw_df.drop(columns=drop_columns)
clean_df = clean_df.drop(columns=drop_columns)  
list_raw_cols = raw_df.columns.tolist()
list_clean_cols = clean_df.columns.tolist()
st.write(list_raw_cols)

#missing values
col1, col2 = st.columns(2)
with col1:
    st.write("Missing Values in Raw Data")
    missing_values = raw_df.isna().sum()
    st.dataframe(missing_values) 
    st.write(f"Total missing values: {missing_values.sum()}") 
with col2:
    st.write("Missing Values in Cleaned Data")
    missing_values_clean = clean_df.isna().sum()
    st.dataframe(missing_values_clean) 
    st.write(f"Total missing values: {missing_values_clean.sum()}")
    
# data distribution
st.write("## Data Distribution Comparison")
for i in list_raw_cols:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(raw_df[i], bins=30, ax=axes[0], color='blue')
    axes[0].set_title(f'Raw Data - {i} Distribution')
    sns.histplot(clean_df[i], bins=30, ax=axes[1], color='green')
    axes[1].set_title(f'Cleaned Data - {i} Distribution')
    st.pyplot(fig)
    
# Distribution for engagement_rate (only in clean_df)
fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(clean_df["engagement_rate"], bins=30, ax=ax, color='green')
ax.set_title("Cleaned Data - engagement_rate Distribution")
st.pyplot(fig)

# Distribution for avg_watch_time_per_view (only in clean_df)
fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(clean_df["avg_watch_time_per_view"], bins=30, ax=ax, color='purple')
ax.set_title("Cleaned Data - avg_watch_time_per_view Distribution")
st.pyplot(fig)
 
raw_numeric_cols = raw_df.select_dtypes(include=['float64', 'int64']).columns
clean_numeric_cols = clean_df.select_dtypes(include=['float64', 'int64']).columns
st.write(f"Numeric Columns: {clean_numeric_cols.tolist()}")

st.write("## Statistical Summary Comparison")
raw_stats = raw_df[raw_numeric_cols].describe().T
raw_stats['skewness'] = raw_df[raw_numeric_cols].skew() 
raw_stats['kurtosis'] = raw_df[raw_numeric_cols].kurtosis()

clean_stats = clean_df[clean_numeric_cols].describe().T
clean_stats['skewness'] = clean_df[clean_numeric_cols].skew()
clean_stats['kurtosis'] = clean_df[clean_numeric_cols].kurtosis()


st.write("Raw Data Statistical Summary")
st.write(raw_stats)
st.write("Cleaned Data Statistical Summary")
st.write(clean_stats)

cat_cols = raw_df.select_dtypes(include=['object', 'category']).columns

st.write(f"Categorical Columns: {cat_cols.tolist()}")

import plotly.express as px
st.write("## Categorical Summary Comparison")
def bar_chart(columnname):
    # Raw Data
    raw_counts = raw_df[columnname].value_counts().reset_index()
    raw_counts.columns = [columnname, 'Count']
    col1,col2 =st.columns(2)
    with col1:
        fig = px.bar(
            raw_counts,
            x=columnname,
            y='Count',
            title=f"Raw Data - {columnname} Value Counts",
            color=columnname,
            hover_data= columnname
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Unique values: {raw_df[columnname].nunique()}")

    
    # Cleaned Data
    clean_counts = clean_df[columnname].value_counts().reset_index()
    clean_counts.columns = [columnname, 'Count']
    with col2:
        fig2 = px.bar(
            clean_counts,
            x=columnname,
            y='Count',
            title=f"Cleaned Data - {columnname} Value Counts",
            color=columnname,
            hover_data= columnname,
            
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.write(f"Unique values: {clean_df[columnname].nunique()}")


for col in cat_cols:
    bar_chart(col)
    




