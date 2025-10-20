# preprocessing.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Helper Functions
def info(df):
    print(f"DataFrame info:{df.info()}")
    return df

def shape(df):
    print(f"DataFrame shape:{df.shape}")
    return df

def load_data(file_path):
    """Load dataset and parse dates"""
    df = pd.read_csv(file_path, parse_dates=['date'])
    print(f"Dataset loaded: {df.shape}")
    return df

def fill_na_with_group_median(df, col, group_cols):
    """Fill NaNs in a column using group median, fallback to overall median"""
    df[col] = df[col].fillna(df.groupby(group_cols)[col].transform('median'))
    
    print(f"Remaining NaNs in {col}: {df[col].isna().sum()}")
    return df

def feature_engineering(df):
    df['engagement_rate'] = (df['likes'].fillna(0) + df['comments'].fillna(0)) / df['views'].replace(0, np.nan)
    df['engagement_rate'] = df['engagement_rate'].fillna(0)

    df['avg_watch_time_per_view'] = df['watch_time_minutes'] / df['views'].replace(0, np.nan)
    df['avg_watch_time_per_view'] = df['avg_watch_time_per_view'].fillna(0)
    return df

def visualize_target(df, target='ad_revenue_usd'):
    """Histogram of target variable"""
    plt.figure(figsize=(10,5))
    plt.hist(df[target].clip(lower=0), bins=80, color='skyblue', edgecolor='black')
    plt.title(f"{target} Distribution")
    plt.xlabel(target)
    plt.ylabel("Frequency")
    plt.show()

def save_clean_data(df, output_path):
    """Save cleaned DataFrame to CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved at: {output_path}")
    
# Main Preprocessing Pipeline

def preprocess_pipeline(raw_path, clean_path):
    df = load_data(raw_path)

    # Impute missing values
    df = fill_na_with_group_median(df, 'likes', ['subscribers'])
    df = fill_na_with_group_median(df, 'likes', ['comments'])
    df['likes'] = df['likes'].fillna(df['likes'].median())
    
    df = fill_na_with_group_median(df, 'comments', ['subscribers'])
    df = fill_na_with_group_median(df, 'comments', ['views'])
    df['comments'] = df['comments'].fillna(df['comments'].median())
    
    df = fill_na_with_group_median(df, 'watch_time_minutes', ['video_length_minutes'])
    df = fill_na_with_group_median(df, 'watch_time_minutes', ['views'])
    df['watch_time_minutes'] = df['watch_time_minutes'].fillna(df['comments'].median())
    # Feature engineering
    df = feature_engineering(df)

    # Optional: visualize target distribution
    visualize_target(df, 'ad_revenue_usd')

    # # Save cleaned data
    # save_clean_data(df, clean_path)

    print(f"Preprocessing complete. Final dataset shape: {df.shape}")
    return df

# ----------------------------
# Run pipeline
# ----------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Project root
    raw_file = os.path.join(BASE_DIR, "Data", "Raw", "youtube_ad_revenue_dataset.csv")
    df = load_data(raw_file)
    info(df)
    shape(df)
    clean_file = os.path.join(BASE_DIR, "Data", "Cleaned", "youtube_ad_revenue_dataset_cleaned.csv")
    df_clean = preprocess_pipeline(raw_file, clean_file)
    cleaned_df = load_data(clean_file)
    info(cleaned_df)
    shape(cleaned_df)
    
    