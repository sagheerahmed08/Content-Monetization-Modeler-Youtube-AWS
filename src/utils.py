# util.py
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def load_pipeline(model_name="LinearRegression"):
    """Load pre-trained model pipeline"""
    model_file = os.path.join(BASE_DIR, "models", f"{model_name}_pipeline.joblib")
    return joblib.load(model_file)

def predict_revenue(pipeline, df):
    """Predict ad revenue using loaded pipeline"""
    return pipeline.predict(df)
