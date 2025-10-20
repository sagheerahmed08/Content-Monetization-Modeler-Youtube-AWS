import streamlit as st

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Home | YouTube Ad Revenue Predictor",
    page_icon="📊",
    layout="wide"
)

# ---------------------------
# Main Title
# ---------------------------
st.title("📊 YouTube Ad Revenue Predictor")

# ---------------------------
# Project Overview
# ---------------------------
st.markdown("""
### 📘 Project Overview
The **YouTube Ad Revenue Predictor** project forecasts estimated ad revenue for YouTube videos using performance metrics.

This app helps creators, analysts, and marketers understand revenue potential and optimize content strategy.
""")

# ---------------------------
# Key Features (Columns)
# ---------------------------
st.subheader("🔍 Key Features")
col1, col2, col3 = st.columns(3)

features = [
    "✅ Data preprocessing and cleaning",
    "✅ Feature engineering: Engagement rate, Avg watch time per view",
    "✅ Multiple regression models: Linear, Ridge, Lasso, RandomForest, XGBoost",
    "✅ Automated model training & best model selection",
    "✅ Interactive prediction dashboard (USD → INR)",
    "✅ Visual EDA of raw and cleaned datasets",
    "✅ Model performance visualization (Accuracy, RMSE, R²)"
]

for i, feat in enumerate(features):
    if i % 3 == 0:
        col1.markdown(feat)
    elif i % 3 == 1:
        col2.markdown(feat)
    else:
        col3.markdown(feat)

# ---------------------------
# Tech Stack (Columns)
# ---------------------------
st.subheader("🧱 Tech Stack")
col1, col2, col3 = st.columns(3)

tech_stack = [
    "**Python**: Data processing & modeling",
    "**Pandas & NumPy**: Data manipulation",
    "**Scikit-Learn & XGBoost**: Machine learning",
    "**Plotly, Matplotlib & Seaborn**: Visualization",
    "**Streamlit**: Interactive dashboard",
    "**AWS S3**: Dataset & model storage"
]

for i, tech in enumerate(tech_stack):
    if i % 3 == 0:
        col1.markdown(tech)
    elif i % 3 == 1:
        col2.markdown(tech)
    else:
        col3.markdown(tech)

st.subheader("⚡ How to Use This App")
col1, col2, col3 = st.columns(3)

with col1:
    st.success("1. Navigate to **EDA** to explore the dataset and understand features.")
    st.write("""
**EDA (Exploratory Data Analysis)**

**Purpose:** Understand the dataset before building models.

**What you do:**
- View raw and cleaned datasets.
- Check for missing values, outliers, or inconsistencies.
- Visualize distributions of features like `views`, `likes`, `comments`, `watch_time_minutes`.
- Explore relationships between features and the target variable `ad_revenue_usd`.
- Use plots like scatter plots, histograms, and box plots.

**Why it’s important:**
- Helps you identify which features are important for predicting ad revenue.
- Detects potential problems in the dataset that could affect model accuracy.
""")
        
with col2:   
    st.success("2. Use **Model Training** to train multiple models, evaluate and Visualize .")
    st.write("""
**Model Training**

**Purpose:** Build machine learning models that can predict ad revenue.

**What you do:**
- Select features (numerical and categorical).
- Apply preprocessing:
  - Handle missing values.
  - Scale numerical features.
  - One-hot encode categorical features.
- Train multiple regression models:
  - Linear Regression – baseline linear model.
  - Ridge & Lasso – linear models with regularization.
  - Random Forest – ensemble tree-based model capturing non-linear relationships.
  - XGBoost – advanced boosting model (if installed).
- Evaluate models using cross-validation (R², RMSE, MAE).
- Automatically select the **best model** based on CV R².
- Store trained models and results in AWS S3 for reuse.

**Why it’s important:**
- Ensures you have a reliable, accurate model.
- Allows comparison of different algorithms to pick the most suitable one.
""")

with col3:    
    st.success("3. Use **Prediction** to estimate ad revenue for new videos.")
    st.write("""
**Prediction**

**Purpose:** Predict YouTube ad revenue for new video data.

**What you do:**
- Input new video metrics: `views`, `likes`, `comments`, `watch_time_minutes`, `video_length_minutes`, `subscribers`, `category`, `device`, `country`.
- The app calculates derived features automatically:
  - **Engagement Rate** = `(likes + comments) / views`
  - **Average Watch Time per View** = `watch_time_minutes / views`
- The selected model predicts the **ad revenue in USD**.
- Converts revenue to INR using the latest USD → INR exchange rate.
- Displays predicted revenue, metrics, and a small feature chart for insight.
- Optionally download the input + prediction as a CSV.

**Why it’s important:**
- Helps YouTubers or analysts estimate potential revenue before uploading videos.
- Provides actionable insights for content strategy.
""")

# ---------------------------
# Footer / Contact
# ---------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center;">
    Project By <b>Sagheer Ahmed</b> | Data Science & ML Project<br><br>
    <a href="https://www.linkedin.com/in/sagheerahmedcse/" target="_blank">
        <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" height="30" style="margin-right:10px;">
    </a>
    <a href="https://github.com/sagheerahmed08" target="_blank">
        <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg" height="30">
    </a>
</div>
""", unsafe_allow_html=True)


