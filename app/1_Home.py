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
# Key Features
# ---------------------------
st.subheader("🔍 Key Features")
st.markdown("""
- ✅ **Data preprocessing and cleaning**
- ✅ **Feature engineering:** Engagement rate, Avg watch time per view
- ✅ **Multiple regression models:** Linear, Ridge, Lasso, RandomForest, XGBoost
- ✅ **Automated model training** & selection of the best model
- ✅ **Interactive prediction dashboard** with USD → INR conversion
- ✅ **Visual EDA** of raw and cleaned datasets
- ✅ **Model performance visualization** (Accuracy, RMSE, R²)
""")

# ---------------------------
# Tech Stack
# ---------------------------
st.subheader("🧱 Tech Stack")
st.markdown("""
- **Python**: Data processing & modeling  
- **Pandas & NumPy**: Data manipulation  
- **Scikit-Learn & XGBoost**: Machine learning  
- **Plotly, Matplotlib & Seaborn**: Visualization  
- **Streamlit**: Interactive dashboard  
- **AWS S3**: Dataset & model storage
""")

# ---------------------------
# Interactive Info Boxes
# ---------------------------
st.subheader("⚡ How to Use This App")
st.markdown("""
1. Navigate to **EDA** to explore the dataset and understand features.  
2. Go to **Model Training** to train multiple models and evaluate performance.  
3. Use **Prediction** to estimate ad revenue for new videos.  
""")

# ---------------------------
# Fun Visualization / Banner
# ---------------------------
st.markdown("---")
st.image(
    "https://images.unsplash.com/photo-1581091012184-7d6b3f539f96?auto=format&fit=crop&w=1400&q=80",
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
