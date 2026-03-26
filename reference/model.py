import pandas as pd
import boto3
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, SGDRegressor, HuberRegressor, PassiveAggressiveRegressor, TheilSenRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pickle
import streamlit as st
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,adjusted_rand_score
import io

S3_BUCKET = "youtube-ad-revenue-app-sagheer"
CLEAN_KEY = "Data/Cleaned/youtube_ad_revenue_dataset_cleaned.csv"
MODEL_PREFIX = "model"

def eval_metrics(y_true, y_pred):
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'adjusted_rand_score': adjusted_rand_score(y_true, y_pred)
    }
    
s3 = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
    region_name=st.secrets["aws"]["region"]
)

obj = s3.get_object(Bucket=S3_BUCKET, Key=CLEAN_KEY)
df = pd.read_csv(obj['Body'])

num_features = ['views', 'comments', 'video_length_minutes','subscribers','engagement_rate', 'avg_watch_time_per_view']
cat_features = ['category', 'device', 'country']

df['engagement_rate'] = (df['likes'].fillna(0) + df['comments'].fillna(0)) / df['views'].replace(0, np.nan)
df['avg_watch_time_per_view'] = df['watch_time_minutes'] / df['views'].replace(0, np.nan)

X = df[num_features + cat_features]
y = df['ad_revenue_usd']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, shuffle=True)

num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "BayesianRidge": BayesianRidge(),
    "SGDRegressor": SGDRegressor(max_iter=1000, tol=1e-3),
    "HuberRegressor": HuberRegressor(),
    "PassiveAggressiveRegressor": PassiveAggressiveRegressor(max_iter=1000, tol=1e-3),
    "TheilSenRegressor": TheilSenRegressor(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "AdaBoostRegressor": AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=100, random_state=42),
    "XGBRegressor": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}   
pipelines = {}
for name, model in models.items():
    print(f"Training model: {name}")
    pipelines[name] = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])  
    pipelines[name].fit(X_train, y_train)
    # Save model to S3  
    mkdir_cmd = MODEL_PREFIX
    import os
    os.system(mkdir_cmd)
    save_path = f"{MODEL_PREFIX}/{name}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(pipelines[name], f) 
        
    print(f"Evaluating model: {name}")
    y_pred = pipelines[name].predict(X_test)
    metrics = eval_metrics(y_test, y_pred)
    print(f"### {name} Performance:")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    perf_df = pd.DataFrame([{
        'Model': name,
        'R2': metrics['r2'],
        'RMSE': metrics['rmse'],
        'MAE': metrics['mae']
        
    }])
