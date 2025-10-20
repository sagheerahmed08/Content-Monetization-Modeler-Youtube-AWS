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
# Hero Section
# ---------------------------
st.title("📊 YouTube Ad Revenue Predictor")
st.markdown("""
Welcome to the **YouTube Ad Revenue Predictor**!  
This interactive dashboard allows creators, analysts, and marketers to forecast estimated ad revenue for YouTube videos based on performance metrics.
""")

# ---------------------------
# Key Features
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
# Tech Stack
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

# ---------------------------
# How to Use This App
# ---------------------------
st.subheader("⚡ How to Use This App")
col1, col2, col3 = st.columns(3)

with col1:
    st.success("1. Navigate to **EDA** to explore the dataset")
    st.write("""
**EDA (Exploratory Data Analysis)**  
- Understand your dataset before building models.  
- View raw and cleaned datasets, check for missing values and outliers.  
- Visualize feature distributions and relationships with `ad_revenue_usd`.  
- Use scatter plots, histograms, and box plots.  
- **Goal:** Identify important features and dataset issues before modeling.
""")

with col2:
    st.success("2. Use **Model Training** to train & evaluate models")
    st.write("""
**Model Training**  
- Select numerical and categorical features.  
- Preprocess data: handle missing values, scale numerical features, one-hot encode categorical features.  
- Train multiple models: Linear, Ridge, Lasso, Random Forest, XGBoost (if installed).  
- Evaluate with cross-validation (R², RMSE, MAE) and select the best model.  
- Save models and results to AWS S3 for reuse.  
- **Goal:** Ensure a reliable, accurate prediction model.
""")

with col3:
    st.success("3. Use **Prediction** to estimate ad revenue")
    st.write("""
**Prediction**  
- Input new video metrics: views, likes, comments, watch time, video length, subscribers, category, device, country.  
- Automatic feature calculation:  
  - Engagement Rate = `(likes + comments) / views`  
  - Avg Watch Time per View = `watch_time_minutes / views`  
- Model predicts revenue in USD and converts to INR.  
- Displays prediction metrics and a mini chart.  
- Optionally download input + prediction as CSV.  
- **Goal:** Help YouTubers or analysts estimate potential revenue and plan content strategy.
""")

# ---------------------------
# Footer / Contact
# ---------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center;">
    <b>Project by Sagheer Ahmed</b> | Data Science & ML Project<br><br>
    <a href="https://www.linkedin.com/in/sagheerahmedcse/" target="_blank">
        <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" height="30" style="margin-right:10px;">
    </a>
    <a href="https://github.com/sagheerahmed08" target="_blank">
        <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg" height="30">
    </a>
</div>
""", unsafe_allow_html=True)
