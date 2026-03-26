import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
import io
import boto3

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, PassiveAggressiveRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    has_xgb = True
except ImportError:
    has_xgb = False

st.set_page_config(page_title="YouTube Ad Revenue Predictor", page_icon="📊", layout="wide")
st.title("📊 YouTube Ad Revenue Predictor")

from config import S3_BUCKET, CLEAN_KEY, MODEL_PREFIX

NUM_FEATURES = ['views', 'comments', 'video_length_minutes', 'subscribers', 'engagement_rate', 'avg_watch_time_per_view']
CAT_FEATURES = ['category', 'device', 'country']

s3 = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
    region_name=st.secrets["aws"]["region"]
)


@st.cache_data
def load_csv_from_s3(bucket, key):
    """Load a CSV from S3 and return a DataFrame."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))


@st.cache_resource
def load_model_from_s3(bucket, key):
    """Load a joblib model from S3; cached across reruns."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    return joblib.load(io.BytesIO(obj['Body'].read()))


def upload_model_to_s3(model, bucket, key):
    """Upload a trained model to S3."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    s3.upload_fileobj(buffer, bucket, key)


@st.cache_data
def load_results_from_s3():
    """Load model comparison results CSV from S3."""
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=f"{MODEL_PREFIX}/results.csv")
        return pd.read_csv(io.BytesIO(obj['Body'].read()))
    except Exception:
        return None


def eval_metrics(y_true, y_pred):
    """Return r2, rmse, and mae for a set of predictions."""
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }


def show_results_table_and_chart(df_results):
    """Render a two-tab Table/Chart view for model results."""
    t1, t2 = st.tabs(["Table", "Chart"])
    with t1:
        st.dataframe(df_results)
    with t2:
        fig = px.bar(df_results, x="Model", y="CV_R2_Mean", error_y="CV_R2_STD",
                     title="Model CV R² Comparison", color="Model", text="CV_R2_Mean")
        st.plotly_chart(fig, use_container_width=True)


tab1, tab2 = st.tabs(["Model Training", "Model Visualization"])

# ============================
# TAB 1: MODEL TRAINING
# ============================
with tab1:
    st.subheader("Load Dataset")
    try:
        df = load_csv_from_s3(S3_BUCKET, CLEAN_KEY)
        st.success(f"Dataset loaded from S3: {df.shape}")
    except Exception as e:
        st.error(f"Failed to load dataset from S3: {e}")
        df = None

    if df is not None:
        df['engagement_rate'] = (df['likes'] + df['comments']) / df['views'].replace(0, 1)
        df['avg_watch_time_per_view'] = df['watch_time_minutes'] / df['views'].replace(0, 1)

        X = df[NUM_FEATURES + CAT_FEATURES]
        y = df['ad_revenue_usd'].fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Training Samples", f"{X_train.shape[0]}")
        col2.metric("Testing Samples", f"{X_test.shape[0]}")
        col3.metric("Features", f"{X_train.shape[1]}")
        col4.metric("Target Variable", "ad_revenue_usd")

        st.subheader("Train Model")

        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=MODEL_PREFIX)
        existing_keys = [item['Key'] for item in response.get('Contents', [])]

        model_files = [
            f"{MODEL_PREFIX}/RandomForest.joblib",
            f"{MODEL_PREFIX}/Ridge.joblib",
            f"{MODEL_PREFIX}/Lasso.joblib",
            f"{MODEL_PREFIX}/PassiveAggressiveRegressor.joblib",
            f"{MODEL_PREFIX}/LinearRegression.joblib",
            f"{MODEL_PREFIX}/XGBoost.joblib",
            f"{MODEL_PREFIX}/BestModel.joblib"
        ]

        for model_path in model_files:
            if model_path in existing_keys:
                st.info(f"{model_path} already trained and uploaded.")
            else:
                st.warning(f"{model_path} not found in S3.")

        df_results = load_results_from_s3()
        if df_results is not None:
            st.success("Model results found in S3.")
            show_results_table_and_chart(df_results)
        else:
            st.info("No previous model results found.")

        if st.button("Train Model"):
            with st.spinner("Training models... Please wait..."):
                time.sleep(1)

                num_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
                cat_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])
                preprocessor = ColumnTransformer([
                    ('num', num_pipeline, NUM_FEATURES),
                    ('cat', cat_pipeline, CAT_FEATURES)
                ])

                alphas = np.logspace(-4, 3, 50)

                ridge_cv = Pipeline([
                    ('preprocessor', preprocessor),
                    ('ridge', RidgeCV(alphas=alphas, cv=5))
                ])
                ridge_cv.fit(X_train, y_train)
                best_alpha_ridge = ridge_cv.named_steps['ridge'].alpha_
                st.write(f"Best alpha for Ridge: {best_alpha_ridge:.6f}")

                lasso_cv = Pipeline([
                    ('preprocessor', preprocessor),
                    ('lasso', LassoCV(alphas=alphas, cv=5, max_iter=5000))
                ])
                lasso_cv.fit(X_train, y_train)
                best_alpha_lasso = lasso_cv.named_steps['lasso'].alpha_
                st.write(f"Best alpha for Lasso: {best_alpha_lasso:.6f}")

                models = {
                    'LinearRegression': LinearRegression(),
                    'Ridge': Ridge(alpha=best_alpha_ridge, random_state=42),
                    'Lasso': Lasso(alpha=best_alpha_lasso, random_state=42),
                    'PassiveAggressiveRegressor': PassiveAggressiveRegressor(max_iter=1000, random_state=42),
                    'RandomForest': RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1, max_depth=15)
                }

                if has_xgb:
                    models['XGBoost'] = XGBRegressor(
                        n_estimators=150, random_state=42, n_jobs=1,
                        objective='reg:squarederror', learning_rate=0.1, max_depth=10
                    )

                results = {}
                trained_pipelines = {}

                for name, model in models.items():
                    st.write(f"Training {name}...")
                    pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
                    pipe.fit(X_train, y_train)
                    trained_pipelines[name] = pipe

                    upload_model_to_s3(pipe, S3_BUCKET, f"{MODEL_PREFIX}/{name}.joblib")
                    st.success(f"{name}.joblib uploaded to S3")

                    y_pred = pipe.predict(X_test)
                    results[name] = {
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                        "MAE": mean_absolute_error(y_test, y_pred),
                        "R2": r2_score(y_test, y_pred),
                    }

                    scores = cross_val_score(pipe, X_train, y_train, scoring='r2', cv=5, n_jobs=-1)
                    results[name]["cv_r2_mean"] = np.mean(scores)
                    results[name]["cv_r2_std"] = np.std(scores)
                    st.write(f"{name}: CV R² = {results[name]['cv_r2_mean']:.4f} ± {results[name]['cv_r2_std']:.4f}")

                best_model_name = max(results, key=lambda k: results[k]['cv_r2_mean'])
                best_pipeline = trained_pipelines[best_model_name]
                metric = eval_metrics(y_test, best_pipeline.predict(X_test))
                st.success(
                    f"Best Model: {best_model_name} — "
                    f"CV R²={results[best_model_name]['cv_r2_mean']:.4f}, "
                    f"R²={metric['r2']:.3f}, RMSE={metric['rmse']:.3f}"
                )

                upload_model_to_s3(best_pipeline, S3_BUCKET, f"{MODEL_PREFIX}/BestModel.joblib")
                st.success("BestModel.joblib uploaded to S3")

                perf_df = pd.DataFrame([
                    {
                        "Model": k,
                        "RMSE": v["RMSE"],
                        "MAE": v["MAE"],
                        "R2": v["R2"],
                        "CV_R2_Mean": v["cv_r2_mean"],
                        "CV_R2_STD": v["cv_r2_std"],
                        "Percent_Of_Best": round(v["cv_r2_mean"] * 100, 2)
                    }
                    for k, v in results.items()
                ])
                csv_buffer = io.StringIO()
                perf_df.to_csv(csv_buffer, index=False)
                s3.put_object(Bucket=S3_BUCKET, Key=f"{MODEL_PREFIX}/results.csv", Body=csv_buffer.getvalue())
                st.success("Model training completed!")
                st.dataframe(perf_df)

# ============================
# TAB 2: MODEL VISUALIZATION
# ============================
with tab2:
    st.subheader("Model Visualization & Evaluation")

    try:
        df_viz = load_csv_from_s3(S3_BUCKET, CLEAN_KEY)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        df_viz = None

    if df_viz is not None:
        df_viz['engagement_rate'] = (df_viz['likes'] + df_viz['comments']) / df_viz['views'].replace(0, 1)
        df_viz['avg_watch_time_per_view'] = df_viz['watch_time_minutes'] / df_viz['views'].replace(0, 1)

        X_viz = df_viz[NUM_FEATURES + CAT_FEATURES]
        y_viz = df_viz['ad_revenue_usd'].fillna(0)
        _, X_test_viz, _, y_test_viz = train_test_split(X_viz, y_viz, test_size=0.2, random_state=42)

        df_results = load_results_from_s3()
        if df_results is not None:
            show_results_table_and_chart(df_results)

            try:
                fig = px.scatter(df_results, x="RMSE", y="CV_R2_Mean", color="Model",
                                 size="CV_R2_Mean", title="RMSE vs CV R² Trade-off",
                                 hover_data=["RMSE", "CV_R2_Mean"])
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Best Model Summary")
                best_row = df_results.loc[df_results["CV_R2_Mean"].idxmax()]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("RMSE", f"{best_row['RMSE']:.2f}", delta=f"Best: {best_row['Model']}")
                c2.metric("MAE", f"{best_row['MAE']:.2f}")
                c3.metric("R²", f"{best_row['R2']:.3f}")
                c4.metric("CV R²", f"{best_row['CV_R2_Mean']:.3f}")
            except Exception:
                pass
        else:
            st.warning("No model results found.")

        model_files_viz = [
            "BestModel.joblib", "LinearRegression.joblib", "Lasso.joblib",
            "Ridge.joblib", "PassiveAggressiveRegressor.joblib",
            "RandomForest.joblib", "XGBoost.joblib"
        ]

        # Feature Importance
        st.subheader("Feature Importance")
        fi_model_name = st.selectbox(
            "Select model to inspect",
            [f.replace(".joblib", "") for f in model_files_viz],
            key="fi_model_select"
        )
        try:
            fi_model = load_model_from_s3(S3_BUCKET, f"{MODEL_PREFIX}/{fi_model_name}.joblib")
            feature_names = [
                n.replace("num__", "").replace("cat__", "")
                for n in fi_model.named_steps['preprocessor'].get_feature_names_out()
            ]
            estimator = fi_model.named_steps['model']
            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
            elif hasattr(estimator, 'coef_'):
                raw = np.abs(estimator.coef_)
                importances = raw / raw.sum() if raw.sum() > 0 else raw
            else:
                importances = None

            if importances is not None:
                fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
                fi_df = fi_df.sort_values("Importance", ascending=True)
                fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation='h',
                                title=f"Feature Importance — {fi_model_name}",
                                text=fi_df["Importance"].apply(lambda v: f"{v:.4f}"))
                fig_fi.update_layout(yaxis_title="", xaxis_title="Importance Score")
                st.plotly_chart(fig_fi, use_container_width=True)
                top3 = fi_df.tail(3)["Feature"].tolist()[::-1]
                st.info(
                    f"**Top drivers of ad revenue for {fi_model_name}:** "
                    + " → ".join(f"`{f}`" for f in top3)
                    + ". Focus on growing **views** and **watch time** to maximise monetisation potential."
                )
            else:
                st.info("This model does not expose feature importances.")
        except Exception as e:
            st.warning(f"Could not load feature importance for {fi_model_name}: {e}")

        # Per-model evaluation
        st.subheader("Model Evaluation Summary")
        cols = st.columns(2)
        for idx, model_path in enumerate(model_files_viz):
            model_name = model_path.replace(".joblib", "")
            try:
                model = load_model_from_s3(S3_BUCKET, f"{MODEL_PREFIX}/{model_path}")
                if model is None:
                    continue
                y_pred = model.predict(X_test_viz)
                metrics = eval_metrics(y_test_viz, y_pred)
                with cols[idx % 2]:
                    st.markdown(f"### {model_name}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R²", f"{metrics['r2']:.3f}")
                    c2.metric("MAE", f"{metrics['mae']:.2f}")
                    c3.metric("RMSE", f"{metrics['rmse']:.2f}")
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=y_test_viz, y=y_pred, alpha=0.6, ax=ax)
                    lims = [min(y_test_viz.min(), y_pred.min()), max(y_test_viz.max(), y_pred.max())]
                    ax.plot(lims, lims, 'r--')
                    ax.set_xlabel("Actual Revenue")
                    ax.set_ylabel("Predicted Revenue")
                    ax.set_title(f"{model_name} — Actual vs Predicted")
                    st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not evaluate {model_name}: {e}")