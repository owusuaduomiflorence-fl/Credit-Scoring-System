import streamlit as st
import pandas as pd
import numpy as np
import joblib
import boto3
from io import BytesIO
import shap

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="Credit Scoring System", layout="wide")
st.title("Credit Scoring & Loan Decision System")

# ---------------------------
# Load Data from R2
# ---------------------------
try:
    st.info("Connecting to Cloudflare R2...")

    R2_ENDPOINT = st.secrets["R2_ENDPOINT_URL"]
    R2_ACCESS_KEY = st.secrets["R2_ACCESS_KEY_ID"]
    R2_SECRET_KEY = st.secrets["R2_SECRET_ACCESS_KEY"]
    R2_BUCKET = st.secrets["R2_BUCKET_NAME"]

    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY
    )

    objects = s3.list_objects_v2(Bucket=R2_BUCKET)

    file_name = next(
        (obj['Key'] for obj in objects.get('Contents', []) if obj['Key'].lower().endswith('.csv')),
        None
    )

    if file_name is None:
        raise Exception("No CSV found in bucket")

    obj = s3.get_object(Bucket=R2_BUCKET, Key=file_name)
    data_df = pd.read_csv(BytesIO(obj['Body'].read()))

    st.success(f"Loaded dataset: {file_name}")

except Exception as e:
    st.error(f"R2 Error: {e}")
    st.stop()

# ---------------------------
# Load Models
# ---------------------------
try:
    logreg_model = joblib.load("models/logreg_v1.pkl")
    xgb_model = joblib.load("models/xgb_v1.pkl")
    scaler = joblib.load("models/scaler_v1.pkl")

    st.success("Models loaded successfully")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ---------------------------
# User Input
# ---------------------------
st.sidebar.header("Customer Input")

def get_user_input():
    data = {
        "RevolvingUtilizationOfUnsecuredLines": st.sidebar.number_input("Revolving Utilization", 0.0, 10.0, 0.5),
        "Age": st.sidebar.number_input("Age", 18, 100, 30),
        "NumberOfTime30-59DaysPastDueNotWorse": st.sidebar.number_input("30-59 Days Late", 0, 20, 0),
        "DebtRatio": st.sidebar.number_input("Debt Ratio", 0.0, 10.0, 0.2),
        "MonthlyIncome": st.sidebar.number_input("Monthly Income", 0, 100000, 3000),
        "NumberOfOpenCreditLinesAndLoans": st.sidebar.number_input("Open Credit Lines", 0, 50, 5),
        "NumberOfTimes90DaysLate": st.sidebar.number_input("90 Days Late", 0, 20, 0),
        "NumberRealEstateLoansOrLines": st.sidebar.number_input("Real Estate Loans", 0, 20, 1),
        "NumberOfTime60-89DaysPastDueNotWorse": st.sidebar.number_input("60-89 Days Late", 0, 20, 0),
        "NumberOfDependents": st.sidebar.number_input("Dependents", 0, 20, 0)
    }
    return pd.DataFrame([data])

input_df = get_user_input()

# ---------------------------
# Feature Engineering
# ---------------------------
engineered_df = input_df.copy()

engineered_df['TotalPastDue'] = (
    engineered_df['NumberOfTime30-59DaysPastDueNotWorse'] +
    engineered_df['NumberOfTime60-89DaysPastDueNotWorse'] +
    engineered_df['NumberOfTimes90DaysLate']
)

engineered_df['DebtPerIncome'] = (
    engineered_df['DebtRatio'] * engineered_df['MonthlyIncome']
)

# ---------------------------
# Predictions
# ---------------------------
st.subheader("Predictions")

# Logistic Regression (original features only)
try:
    logreg_features = scaler.feature_names_in_
    logreg_input = input_df[logreg_features]

    scaled_input = scaler.transform(logreg_input)

    logreg_prob = logreg_model.predict_proba(scaled_input)[0][1]
    logreg_pred = logreg_model.predict(scaled_input)[0]

except Exception as e:
    st.error(f"LogReg error: {e}")
    st.stop()

# XGBoost (engineered features)
try:
    xgb_features = xgb_model.feature_names_in_
    xgb_input = engineered_df[xgb_features]

    xgb_prob = xgb_model.predict_proba(xgb_input)[0][1]
    xgb_pred = xgb_model.predict(xgb_input)[0]

except Exception as e:
    st.error(f"XGBoost error: {e}")
    st.stop()

# ---------------------------
# Display Results
# ---------------------------
st.write("### Logistic Regression")
st.write(f"Prediction: {'Delinquent' if logreg_pred else 'Not Delinquent'}")
st.write(f"Probability: {logreg_prob:.2f}")

st.write("### XGBoost")
st.write(f"Prediction: {'Delinquent' if xgb_pred else 'Not Delinquent'}")
st.write(f"Probability: {xgb_prob:.2f}")

# ---------------------------
# SHAP Explainability
# ---------------------------
st.subheader("SHAP Explainability (XGBoost)")

try:
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(xgb_input)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.force_plot(explainer.expected_value, shap_values, xgb_input, matplotlib=True)
    st.pyplot(bbox_inches='tight')

except Exception as e:
    st.warning(f"SHAP failed: {e}")

# ---------------------------
# Batch Prediction
# ---------------------------
st.subheader("Batch Prediction")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    batch_df = pd.read_csv(file)

    # Feature engineering
    batch_df['TotalPastDue'] = (
        batch_df['NumberOfTime30-59DaysPastDueNotWorse'] +
        batch_df['NumberOfTime60-89DaysPastDueNotWorse'] +
        batch_df['NumberOfTimes90DaysLate']
    )

    batch_df['DebtPerIncome'] = batch_df['DebtRatio'] * batch_df['MonthlyIncome']

    # Predictions
    batch_logreg = scaler.transform(batch_df[logreg_features])
    batch_df['LogReg_Prob'] = logreg_model.predict_proba(batch_logreg)[:, 1]

    batch_xgb = batch_df[xgb_features]
    batch_df['XGB_Prob'] = xgb_model.predict_proba(batch_xgb)[:, 1]

    st.dataframe(batch_df)

    st.download_button(
        "Download Predictions",
        batch_df.to_csv(index=False),
        "predictions.csv"
    )