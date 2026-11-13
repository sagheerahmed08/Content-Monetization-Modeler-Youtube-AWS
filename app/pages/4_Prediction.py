
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import importlib
import io
import boto3
import plotly.express as px
import requests
from datetime import datetime

from config import S3_BUCKET, CLEAN_KEY, MODEL_PREFIX, PREDICTION_KEY

try:
    _xgb = importlib.import_module("xgboost")
    XGBRegressor = getattr(_xgb, "XGBRegressor")
    has_xgb = True
except Exception:
    has_xgb = False

LINEAR_MODELS = {
    "LinearRegression",
    "Ridge",
    "Lasso",
    "PassiveAggressiveRegressor"
}

MAX_RPM = 20.0

COUNTRY_CURRENCY = {
    "IN": ("INR", "₹"),
    "US": ("USD", "$"),
    "UK": ("GBP", "£"),
    "CA": ("CAD", "CA$"),
}

st.set_page_config(
    page_title="Prediction | YouTube Ad Revenue Predictor",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Predict YouTube Ad Revenue")

s3 = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
    region_name=st.secrets["aws"]["region"],
)


@st.cache_data(ttl=3600)
def get_exchange_rates():
    """Fetch live USD exchange rates; fall back to hardcoded defaults."""
    try:
        res = requests.get(
            "https://v6.exchangerate-api.com/v6/01dd1651e64cb8df5a89b465/latest/USD",
            timeout=10
        ).json()

        rates = res.get("conversion_rates", {})

        return {
            "INR": rates.get("INR", 88.70),
            "GBP": rates.get("GBP", 0.79),
            "CAD": rates.get("CAD", 1.36),
            "USD": 1.0,
        }

    except Exception:
        return {
            "INR": 88.70,
            "GBP": 0.79,
            "CAD": 1.36,
            "USD": 1.0
        }


@st.cache_resource
def load_model_from_s3(bucket, key):
    """Load a joblib model from S3; cached across reruns."""
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return joblib.load(io.BytesIO(obj["Body"].read()))

    except Exception as e:
        st.error(f"Failed to load model from S3: {e}")
        return None


@st.cache_data
def load_results_from_s3():
    """Load model comparison results CSV from S3."""
    try:
        obj = s3.get_object(
            Bucket=S3_BUCKET,
            Key=f"{MODEL_PREFIX}/results.csv"
        )

        return pd.read_csv(io.BytesIO(obj["Body"].read()))

    except Exception:
        return None


def user_input_features():
    col1, col2, col3 = st.columns(3)

    views = col1.number_input(
        "Views",
        min_value=0,
        max_value=10_000_000,
        value=10_000
    )

    likes = col2.number_input(
        "Likes",
        min_value=0,
        max_value=1_000_000,
        value=500
    )

    comments = col3.number_input(
        "Comments",
        min_value=0,
        max_value=1_000_000,
        value=50
    )

    watch_time = col1.number_input(
        "Watch Time (min)",
        min_value=0.0,
        max_value=1_000_000.0,
        value=2000.0
    )

    length = col2.number_input(
        "Video Length (min)",
        min_value=0.1,
        max_value=120.0,
        value=10.0
    )

    subs = col3.number_input(
        "Subscribers",
        min_value=0,
        max_value=10_000_000,
        value=10000
    )

    category = col1.selectbox(
        "Category",
        ["Entertainment", "Gaming", "Education", "Music", "News"]
    )

    device = col2.selectbox(
        "Device",
        ["Mobile", "Tablet", "TV", "Desktop"]
    )

    country = col3.selectbox(
        "Country",
        list(COUNTRY_CURRENCY.keys())
    )

    df = pd.DataFrame({
        "views": [views],
        "likes": [likes],
        "comments": [comments],
        "watch_time_minutes": [watch_time],
        "video_length_minutes": [length],
        "subscribers": [subs],
        "category": [category],
        "device": [device],
        "country": [country],
    })

    df["engagement_rate"] = (
        df["likes"] + df["comments"]
    ) / df["views"].replace(0, 1)

    df["avg_watch_time_per_view"] = (
        df["watch_time_minutes"]
        / df["views"].replace(0, 1)
    )

    return df


def validate_inputs(df):
    """Return list of validation errors; empty list means valid."""
    errors = []
    row = df.iloc[0]

    if row["views"] < 0:
        errors.append("Views cannot be negative.")

    if row["likes"] < 0:
        errors.append("Likes cannot be negative.")

    if row["comments"] < 0:
        errors.append("Comments cannot be negative.")

    if row["watch_time_minutes"] < 0:
        errors.append("Watch time cannot be negative.")

    if row["video_length_minutes"] <= 0:
        errors.append("Video length must be greater than 0.")

    if row["subscribers"] < 0:
        errors.append("Subscribers cannot be negative.")

    if row["views"] > 0 and row["likes"] > row["views"]:
        errors.append("Likes cannot exceed views.")

    return errors


def apply_revenue_cap(raw_pred, views):
    """Cap prediction to maximum RPM per 1000 views."""
    cap = views * MAX_RPM / 1000.0
    return float(np.clip(raw_pred, 0.0, max(cap, 0.01)))


# ─── Main ───────────────────────────────────────────────────────────────

exchange_rates = get_exchange_rates()

st.success(
    f"💱 Current Exchange Rate: "
    f"1 USD = ₹{exchange_rates['INR']:.2f} INR"
)

select_model = st.selectbox(
    "Select Model",
    [
        "BestModel",
        "LinearRegression",
        "Ridge",
        "Lasso",
        "PassiveAggressiveRegressor",
        "RandomForest",
        "XGBoost"
    ],
    key="model_select",
)

model = load_model_from_s3(
    S3_BUCKET,
    f"{MODEL_PREFIX}/{select_model}.joblib"
)

df = user_input_features()

st.subheader("📋 Input Features")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)

errors = validate_inputs(df)

for err in errors:
    st.error(err)

results_df = load_results_from_s3()

predict_clicked = st.button(
    "📊 Predict Now",
    type="primary",
    use_container_width=True,
    disabled=bool(errors)
)


if predict_clicked and not errors and model:

    if select_model in LINEAR_MODELS:
        st.caption(
            "ℹ️ Linear models assume a straight-line relationship "
            "between features and revenue. Tree-based models "
            "(RandomForest, XGBoost) are typically more accurate."
        )

    raw_pred = model.predict(df)[0]

    views_val = int(df["views"].iloc[0])

    pred = apply_revenue_cap(raw_pred, views_val)

    country_val = df["country"].iloc[0]

    currency_code, currency_sym = COUNTRY_CURRENCY.get(
        country_val,
        ("USD", "$")
    )

    rate = exchange_rates.get(currency_code, 1.0)

    pred_local = pred * rate

    st.success(
        f"Predicted Ad Revenue: "
        f"{currency_sym}{pred_local:,.2f} ({currency_code}) "
        f"≈ ${pred:,.2f} USD"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👀 Views", f"{views_val:,}")

    col2.metric(
        "👍 Likes",
        f"{int(df['likes'].iloc[0]):,}"
    )

    col3.metric(
        "⏱ Watch Time",
        f"{df['watch_time_minutes'].iloc[0]:.0f} min"
    )

    col4.markdown(
        f"""
        <div style="
            background-color:#d4edda;
            color:#155724;
            padding:10px;
            border-radius:5px;
            text-align:center;
            font-size:1.2em;
            font-weight:bold;">
            💰 Predicted Revenue<br>
            (${pred:,.2f} USD)
            <span style="font-size:0.8em;">
            {currency_sym}{pred_local:,.2f}
    
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Confidence Range ──────────────────────────────────────────

    rmse = None

    if results_df is not None:

        lookup_name = select_model

        if (
            select_model == "BestModel"
            and "BestModel" not in results_df["Model"].values
        ):
            lookup_name = results_df.loc[
                results_df["CV_R2_Mean"].idxmax(),
                "Model"
            ]

        match = results_df[
            results_df["Model"] == lookup_name
        ]

        if not match.empty:
            rmse = float(match["RMSE"].iloc[0])

    if rmse is not None:

        lo_local = max(0.0, pred - rmse) * rate
        hi_local = (pred + rmse) * rate

        st.info(
            f"95% Confidence Range: "
            f"{currency_sym}{lo_local:,.2f} – "
            f"{currency_sym}{hi_local:,.2f} "
            f"(±1 RMSE = ${rmse:,.2f})"
        )

    # ─── Feature Insights ──────────────────────────────────────────

    st.subheader("📊 Feature Insights")

    st.bar_chart(
        df[
            [
                "views",
                "likes",
                "comments",
                "watch_time_minutes",
                "subscribers",
                "engagement_rate",
                "avg_watch_time_per_view"
            ]
        ].T
    )

    # ─── Build Export Row with Timestamp ───────────────────────────

    export_df = df.copy()

    # Timestamp as the first column
    prediction_timestamp = datetime.now().astimezone()

    export_df.insert(
        0,
        "Timestamp",
        prediction_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    )

    export_df["predicted_revenue_usd"] = pred

    export_df[
        f"predicted_revenue_{currency_code.lower()}"
    ] = pred_local

    export_df["model_used"] = select_model

    # ─── Append Prediction to AWS S3 Log ───────────────────────────

    try:

        try:
            existing_obj = s3.get_object(
                Bucket=S3_BUCKET,
                Key=PREDICTION_KEY
            )

            existing = pd.read_csv(
                io.BytesIO(existing_obj["Body"].read())
            )

            # Handle older CSV files with lowercase timestamp
            if (
                "timestamp" in existing.columns
                and "Timestamp" not in existing.columns
            ):
                existing.rename(
                    columns={"timestamp": "Timestamp"},
                    inplace=True
                )

            # Append new prediction without deleting old records
            log_df = pd.concat(
                [existing, export_df],
                ignore_index=True
            )

        except s3.exceptions.NoSuchKey:
            log_df = export_df.copy()

        # Convert complete history to CSV
        buf = io.StringIO()

        log_df.to_csv(
            buf,
            index=False
        )

        # Upload updated CSV to S3
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=PREDICTION_KEY,
            Body=buf.getvalue().encode("utf-8"),
            ContentType="text/csv"
        )

        st.success(
            "✅ Prediction successfully saved to AWS S3!"
        )

        st.caption(
            f"Prediction Timestamp: "
            f"{prediction_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    except Exception as e:

        st.error(
            f"Failed to save prediction to AWS S3: {e}"
        )

    # ─── Download Prediction CSV ───────────────────────────────────

    csv_data = export_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download Prediction Data",
        data=csv_data,
        file_name="prediction.csv",
        mime="text/csv",
    )


# ─── Model Performance Comparison ───────────────────────────────────────

st.subheader("📈 Model Performance Comparison")

if results_df is not None:

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        results_df,
        x="Model",
        y="CV_R2_Mean",
        error_y="CV_R2_STD",
        title="Model Cross-Validation R² Comparison",
        text="CV_R2_Mean",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.info(
        "ℹ️ No model performance data found. "
        "Train models on the Model page first."
    )


# ─── Prediction History from AWS S3 ────────────────────────────────────

st.divider()

st.subheader("📂 Prediction History")

PREDICTION_CSV_URL = (
    "https://youtube-ad-revenue-app-sagheer.s3."
    "eu-north-1.amazonaws.com/logs/prediction.csv"
)

st.markdown(
    f"""
    **AWS S3 Prediction Log**

    [🔗 View prediction.csv directly in AWS S3]({PREDICTION_CSV_URL})
    """
)

try:

    prediction_obj = s3.get_object(
        Bucket=S3_BUCKET,
        Key=PREDICTION_KEY
    )

    prediction_history = pd.read_csv(
        io.BytesIO(prediction_obj["Body"].read())
    )

    if not prediction_history.empty:

        st.success(
            f"Successfully loaded "
            f"{len(prediction_history):,} prediction records."
        )

        st.dataframe(
            prediction_history,
            use_container_width=True,
            hide_index=True
        )

        # Download complete prediction history
        history_csv = prediction_history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Complete Prediction History",
            data=history_csv,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            f"Source: s3://{S3_BUCKET}/{PREDICTION_KEY}"
        )

    else:

        st.info(
            "No prediction records available yet."
        )

except Exception as e:

    st.warning(
        f"Unable to load prediction history from S3: {e}"
    )
