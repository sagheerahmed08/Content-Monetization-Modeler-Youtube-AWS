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

# ---------------------------
# Fun Visualization / Banner
# ---------------------------
st.markdown("---")
st.image(
    "https://images.unsplash.com/photo-1557800636-894a64c1696f?auto=format&fit=crop&w=1400&q=80",
    caption="Visualize YouTube Performance & Revenue",
    use_column_width=True
)

# ---------------------------
# Footer / Contact
# ---------------------------
st.markdown("---")
st.markdown("""
Made with ❤️ by **Your Name** | Data Science & ML Project  
""")
