import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pathlib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix
import plotly.express as px
import pathlib
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# Optional: XGBoost - import dynamically so the module isn't required at static-analysis time
import importlib
try:
    _xgb = importlib.import_module("xgboost")
    XGBRegressor = getattr(_xgb, "XGBRegressor")
    has_xgb = True
except Exception:
    # xgboost not installed or failed to import; downstream code will skip XGBoost model
    has_xgb = False
    
    
st.set_page_config(page_title="Model Visualization | YouTube Ad Revenue Predictor", page_icon="📊",layout="wide")
st.title("📊 Model")

tab1, tab2 = st.tabs(["Model Train", "Model Visualization"])
with tab1:
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent  # Project root
    DATA_FILE = BASE_DIR / "Data" / "Cleaned" / "youtube_ad_revenue_dataset_cleaned.csv"
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_FILE)
    print(f"Cleaned data loaded: {df.shape}")

    # ----------------------------
    # Features and target
    # ----------------------------
    num_features = [
        'views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes',
        'subscribers', 'engagement_rate', 'avg_watch_time_per_view'
    ]
    cat_features = ['category', 'device', 'country']

    # Derived features
    df['engagement_rate'] = (df['likes'] + df['comments']) / df['views'].replace(0, 1)
    df['avg_watch_time_per_view'] = df['watch_time_minutes'] / df['views'].replace(0, 1)

    X = df[num_features + cat_features]
    y = df['ad_revenue_usd'].fillna(0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.subheader("Train Your Models")

    
    if MODEL_DIR.exists() and any(MODEL_DIR.glob("*.joblib")):
        st.success(f"Model files already exist in {MODEL_DIR}. Training again will overwrite existing models.")
        for existing_model in MODEL_DIR.glob("*.joblib"):
            st.success(f"Existing model: {existing_model.name}")
        delete_button = st.button("🗑️ Delete Existing Models")
        if delete_button:
            for existing_model in MODEL_DIR.glob("*.joblib"):
                existing_model.unlink()
            st.success("Existing models deleted. You can now train new models.")
            train_button = st.button("🚀 Train Models")
    else:
        train_button = st.button("🚀 Train Models")
        if train_button:
            with st.spinner("Training models... ⏳ Please wait"):
                time.sleep(1)  # just to simulate loading
                
                # Preprocessing pipelines
                num_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
                cat_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])

                preprocessor = ColumnTransformer([
                    ('num', num_pipeline, num_features),
                    ('cat', cat_pipeline, cat_features)
                ])

                # Models for comparison
                models = {
                    'LinearRegression': LinearRegression(),
                    'Ridge': Ridge(random_state=42),
                    'Lasso': Lasso(random_state=42),
                    'RandomForest': RandomForestRegressor(
                        n_estimators=200, random_state=42, n_jobs=-1
                    )
                }

                if has_xgb:
                    models['XGBoost'] = XGBRegressor(
                        n_estimators=200, random_state=42, n_jobs=1, objective='reg:squarederror'
                    )

                # Train, cross-validate, save models
                results = {}

                for name, model in models.items():
                    pipe = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                    pipe.fit(X_train, y_train)
                    joblib.dump(pipe, MODEL_DIR / f"{name}.joblib")
                    st.success(f"Saved {name} model at {MODEL_DIR / f'{name}.joblib'}")
                    
                    scores = cross_val_score(pipe, X_train, y_train, scoring='r2', cv=5, n_jobs=-1)
                    results[name] = {'cv_r2_mean': np.mean(scores), 'cv_r2_std': np.std(scores)}
                    st.success(f"{name}: CV R² = {np.mean(scores):.4f} ± {np.std(scores):.4f}")

                # Save comparison results to CSV for Streamlit app
                perf_df = pd.DataFrame([
                    {"Model": k, "CV_R2_Mean": v['cv_r2_mean'], "CV_R2_STD": v['cv_r2_std']}
                    for k, v in results.items()
                ])
                perf_df.to_csv(MODEL_DIR / "results.csv", index=False)
                st.success(f"Model comparison results saved at {MODEL_DIR / 'results.csv'}")
                st.success("Model Training Completed!")
                st.write("### Model Performance Summary")
                st.dataframe(perf_df, use_container_width=True, hide_index=True)
                best_model_name = perf_df.iloc[0]["Model"]
                best_r2 = perf_df.iloc[0]["CV_R2_Mean"]
                st.metric(label="Best Model", value=best_model_name, delta=f"{best_r2:.3f} R²")
                # ----------------------------
                # Fit best model
                # ----------------------------
                best_name = max(results, key=lambda k: results[k]['cv_r2_mean'])
                best_model = models[best_name]
                pipe_best = Pipeline([('preprocessor', preprocessor), ('model', best_model)])
                pipe_best.fit(X_train, y_train)
                st.bar_chart(perf_df.set_index("Model")["CV_R2_Mean"])
                # ----------------------------
                # Evaluate on test set
                # ----------------------------
                y_pred = pipe_best.predict(X_test)

                def eval_metrics(y_true, y_pred):
                    return {
                        'r2': r2_score(y_true, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                        'mae': mean_absolute_error(y_true, y_pred)
                    }

                metrics = eval_metrics(y_test, y_pred)
                st.success(f"Best model: {best_name}")
                st.success(f"Test set performance: {metrics}")
                # Save best model
                joblib.dump(pipe_best, MODEL_DIR / "best_model.joblib")
                st.success(f"Best model saved at {MODEL_DIR / 'best_model.joblib'}")
            
                # Feature importance for tree models
                tree_models = ['RandomForest', 'XGBoost']
                if best_name in tree_models:
                    model_step = pipe_best.named_steps['model']
                    ohe = pipe_best.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
                    cat_names = ohe.get_feature_names_out(cat_features)
                    feature_names = num_features + list(cat_names)
                    
                    importances = model_step.feature_importances_
                    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
                    
                    st.write("🌟 Top 20 Feature Importances:")
                    st.bar_chart(fi.head(20))
                    plt.figure(figsize=(10, 6))
                    fi.head(20).plot.barh()
                    plt.title("Top 20 Feature Importances")
                    plt.tight_layout()
                    st.pyplot(plt, use_container_width=True)

with tab2:
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
    MODEL_DIR = BASE_DIR / "app" / "models"
    DATA_FILE = BASE_DIR / "app" / "Data" / "Cleaned" / "youtube_ad_revenue_dataset_cleaned.csv"


    # Load results
    results_file = MODEL_DIR / "results.csv"
    if results_file.exists():
        df_results = pd.read_csv(results_file)
        st.subheader("📈 Model Comparison (Cross-Validation R²)")
        st.dataframe(df_results)
        fig = px.bar(df_results, x="Model", y="CV_R2_Mean", error_y="CV_R2_STD",
                    title="Cross-Validation R² Comparison", text="CV_R2_Mean")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model results file not found.")

    # Load best model
    best_model_path = MODEL_DIR / "best_model.joblib"
    if best_model_path.exists():
        st.subheader("Best Model Evaluation")
        df = pd.read_csv(DATA_FILE)
        model = joblib.load(best_model_path)

        X = df.drop(columns=['ad_revenue_usd'])
        y = df['ad_revenue_usd']
        y_pred = model.predict(X)

        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        col1, col2, col3 = st.columns(3)

        col1.metric("R² Score", f"{r2:.3f}")
        col2.metric("MAE", f"{mae:.2f}")
        col3.metric("RMSE", f"{rmse:.2f}")

        # Scatter plot
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x=y, y=y_pred, alpha=0.6, ax=ax2)
        ax2.set_xlabel("Actual Revenue")
        ax2.set_ylabel("Predicted Revenue")
        #draw line visually representing y=x
        lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
        ax2.plot(lims, lims, 'r-')
        ax2.set_title("Actual vs Predicted Revenue")
        st.pyplot(fig2)
    else:
        st.warning("Best model not found.")

    best_model_path = MODEL_DIR / "best_model.joblib"
    linear_model_path = MODEL_DIR / "linearRegression.joblib"
    lasso_model_path = MODEL_DIR / "Lasso.joblib"
    ridge_model_path = MODEL_DIR / "Ridge.joblib"
    random_forest_model_path = MODEL_DIR / "RandomForest.joblib"
    xgboost_model_path = MODEL_DIR / "XGBoost.joblib"

    model_paths = {
        "Best Model": best_model_path,
        "Linear Regression": linear_model_path,
        "Lasso": lasso_model_path,
        "Ridge": ridge_model_path,
        "Random Forest": random_forest_model_path,
        "XGBoost": xgboost_model_path
    }

    # Load dataset
    df = pd.read_csv(DATA_FILE)
    X = df.drop(columns=["ad_revenue_usd"])
    y = df["ad_revenue_usd"]

    # Feature lists (same as used in training) - safe defaults in case train wasn't run here
    num_features = [
        'views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes',
        'subscribers', 'engagement_rate', 'avg_watch_time_per_view'
    ]
    cat_features = ['category', 'device', 'country']

    col1, col2 = st.columns(2)  # two columns for displaying models side-by-side
cols = [col1, col2]

for i, (model_name, model_path) in enumerate(model_paths.items()):
    with cols[i % 2]:  # alternate between columns for grid effect
        if model_path.exists():
            st.subheader(f"🔍 {model_name} Evaluation")

            # Load model
            model = joblib.load(model_path)

            # Predict
            y_pred = model.predict(X)

            # Metrics
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            m1, m2, m3 = st.columns(3)
            m1.metric("R² Score", f"{r2:.3f}")
            m2.metric("MAE", f"{mae:.2f}")
            m3.metric("RMSE", f"{rmse:.2f}")

            # Scatter Plot
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=y, y=y_pred, alpha=0.6, ax=ax, color="teal")
            ax.set_xlabel("Actual Revenue")
            ax.set_ylabel("Predicted Revenue")
            lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
            ax.plot(lims, lims, 'r--')
            ax.set_title(f"Actual vs Predicted - {model_name}")
            st.pyplot(fig, use_container_width=True)

            # Feature Importance (for tree-based models)
            if model_name in ["Random Forest", "XGBoost"]:
                st.write("**Top 20 Feature Importances**")

                # Handle pipeline and preprocessing
                if hasattr(model, "named_steps"):
                    model_step = model.named_steps.get("model", model)
                    preproc = model.named_steps.get("preprocessor", None)
                else:
                    model_step = model
                    preproc = None

                # Build feature names
                if preproc is not None:
                    try:
                        ohe = preproc.named_transformers_["cat"].named_steps["onehot"]
                        cat_names = ohe.get_feature_names_out(cat_features)
                        feature_names = num_features + list(cat_names)
                    except Exception:
                        feature_names = X.columns
                else:
                    feature_names = X.columns

                # Plot feature importances
                if hasattr(model_step, "feature_importances_"):
                    importances = model_step.feature_importances_
                    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)

                    st.bar_chart(fi.head(20))
                    plt.figure(figsize=(10, 6))
                    fi.head(20).plot.barh(color="coral")
                    plt.title(f"Top 20 Features - {model_name}")
                    plt.tight_layout()
                    st.pyplot(plt, use_container_width=True)
                else:
                    st.info(f"{model_name} does not expose feature importances.")
        else:
            st.warning(f"{model_name} model not found. Please train it first.")
