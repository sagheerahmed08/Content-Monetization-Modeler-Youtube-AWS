# 🎬 Content Monetization Modeler (YouTube Revenue Prediction)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.47.1-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-EC4D37?logo=xgboost)](https://xgboost.ai/)
[![AWS S3](https://img.shields.io/badge/AWS-S3-FF9900?logo=amazon-aws)](https://aws.amazon.com/s3/)


---
The **Content Monetization Modeler** is a data-driven web application that predicts YouTube creators’ **ad revenue** using engagement metrics such as views, likes, comments, watch time, and subscribers.  
It enables creators and analysts to forecast monetization potential, visualize key patterns, and track performance over time — all through an interactive Streamlit interface.

---

## 🚀 Key Features
- **Interactive EDA**: Analyze engagement metrics and correlations.
- **ML Model Training**: Train multiple models (Linear, Ridge, Lasso, Random Forest, XGBoost) and auto-select the best one.
- **Prediction Module**: Upload data and predict revenue in USD & INR.
- **AWS Integration**:
  - Load and store datasets, models, and logs directly in **S3**.
  - Append and store predictions (`prediction.csv`) automatically.
- **Cloud Deployment**: Hosted on **AWS EC2 (Ubuntu 22.04)** for browser-based access.

---

## 🧠 Tech Stack

### **Languages & Libraries**
| Category | Tools / Libraries |
|-----------|------------------|
| Core Language | Python 3.13 |
| Web Framework | Streamlit |
| Machine Learning | scikit-learn, xgboost, joblib |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| AWS SDK | boto3, aiobotocore, awscli |
| Cloud Storage | AWS S3 |
| Statistical Tools | statsmodels |
| Utilities | requests, s3fs, fsspec |

---

## 🧩 Machine Learning Models Used
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**

The system compares model performance using R², MSE, and RMSE, and selects the **best model** automatically (`BestModel.joblib`).

---

## ☁️ AWS Setup

### 1. **S3 Bucket Structure**
```
youtube-ad-revenue-app-sagheer/
  ├── app/
     ├── Data/
        ├── Cleaned/
          └── youtube_ad_revenue_dataset_cleaned.csv
        ├── Raw/
          └── youtube_ad_revenue_dataset.csv
    ├── logs/
       └── prediction.csv
    └── models/
      ├── BestModel.joblib
      ├── LinearRegression.joblib
      ├── Ridge.joblib
      ├── Lasso.joblib
      ├── RandomForest.joblib
      ├── XGBoost.joblib
      └── results.csv
```

### 2. **IAM User Setup**
Create an IAM user with the following permissions:
- `AmazonS3FullAccess`
- `IAMUserChangePassword`

Generate **Access Key ID** and **Secret Access Key**, then configure locally:
```bash
aws configure
```

## 🖥️ Folder Structure (GitHub Repo)
```
Content-Monetization-Modeler-Youtube-AWS-main/
├── .gitignore
├── README.md
├── app/
  ├── 1_Home.py
  ├── preprocessing.py
  ├── requirements.txt
  └── pages/
        ├── 2_EDA.py
        ├── 3_Model.py
        └── 4_Prediction.py
└── .devcontainer/
    └── devcontainer.json
```

## ⚙️ Installation & Setup

###1️⃣ Clone the Repository
```
git clone https://github.com/yourusername/Content-Monetization-Modeler-Youtube-AWS.git
cd Content-Monetization-Modeler-Youtube-AWS/app
```

###2️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### requirements.txt
```
# Data handling
pandas==2.3.1
numpy==2.2.6

# Machine Learning
scikit-learn==1.7.2
joblib==1.5.2
xgboost

# Visualization
matplotlib==3.10.5
seaborn==0.13.2
plotly==6.3.1

# Web app
streamlit==1.47.1

# AWS
aiohttp==3.13.1
aioitertools==0.12.0
aiosignal==1.4.0
boto3
aiobotocore
botocore
awscli

# Utilities
requests==2.32.4
fsspec==2025.9.0
s3fs==2025.9.0
statsmodels
```
