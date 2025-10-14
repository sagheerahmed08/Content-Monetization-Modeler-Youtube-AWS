import streamlit as st

st.set_page_config(page_title="Home | YouTube Ad Revenue Predictor", page_icon="📊",layout="wide")

st.title("📊 YouTube Ad Revenue Prediction")
st.markdown("""
### 📘 Project Overview
This project predicts **YouTube Ad Revenue** using video performance metrics such as views, likes, comments, watch time, and engagement rate.

#### 🔍 Key Features:
- Data preprocessing and cleaning
- Feature engineering (engagement rate, avg watch time per view)
- Multiple regression models: Linear, Ridge, Lasso, RandomForest, XGBoost
- Automated model training, comparison, and best model selection
- Interactive prediction dashboard with currency conversion (USD → INR)
- Visual EDA of raw and cleaned datasets
- Model performance visualization (Accuracy, RMSE, R²)

#### 🧱 Tech Stack:
- Python, Pandas, NumPy  
- Scikit-Learn, XGBoost  
- Plotly, Matplotlib, Seaborn  
- Streamlit (for interactive dashboard)
""")
