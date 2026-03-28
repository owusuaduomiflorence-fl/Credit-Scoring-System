import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import boto3
from io import BytesIO

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="Credit Scoring & Loan Decision System", layout="wide")
st.title("Credit Scoring & Loan Decision System")
st.markdown("""
This app predicts the probability of 90-day delinquency for customers using **Logistic Regression** and **XGBoost**.
""")

# ---------------------------
# Load Models & Scaler from repo
# ---------------------------
try:
    st.info("Loading pre-trained models...")

    logreg_model = joblib.load("models/logreg_v1.pkl")
    xgb_model = joblib.load("models/xgb_v1.pkl")
    scaler = joblib.load("models/scaler_v1.pkl")

except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# ---------------------------
# Cloudflare R2 Connection for Data
# ---------------------------
try:
    st.info("Connecting to Cloudflare R2 bucket for data...")

    endpoint_url = st.secrets["R2_ENDPOINT_URL"]
    access_key = st.secrets["R2_ACCESS_KEY_ID"]
    secret_key = st.secrets["R2_SECRET_ACCESS_KEY"]
    bucket_name = st.secrets["R2_BUCKET_NAME"]

    # Initialize S3 client
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

except Exception as e:
    st.error(f"Failed to connect to Cloudflare R2: {e}")
    st.stop()

# Function to load CSV from R2
def load_csv_from_r2(file_name):
    obj = s3.get_object(Bucket=bucket_name, Key=file_name)
    return pd.read_csv(BytesIO(obj['Body'].read()))

# ---------------------------
# User Input Section
# ---------------------------
st.sidebar.header("Customer Data Input")

def user_input_features():
    RevolvingUtilizationOfUnsecuredLines = st.sidebar.number_input(
        "Revolving Utilization (%)", min_value=0.0, max_value=10.0, value=0.5)
    Age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    NumberOfTime30_59DaysPastDueNotWorse = st.sidebar.number_input(
        "Times 30-59 Days Past Due", min_value=0, max_value=20, value=0)
    DebtRatio = st.sidebar.number_input("Debt Ratio", min_value=0.0, max_value=10.0, value=0.2)
    MonthlyIncome = st.sidebar.number_input("Monthly Income", min_value=0, value=3000)
    NumberOfOpenCreditLinesAndLoans = st.sidebar.number_input(
        "Open Credit Lines & Loans", min_value=0, max_value=50, value=5)
    NumberOfTimes90DaysLate = st.sidebar.number_input("Times 90 Days Late", min_value=0, max_value=20, value=0)
    NumberRealEstateLoansOrLines = st.sidebar.number_input(
        "Real Estate Loans/Lines", min_value=0, max_value=20, value=1)
    NumberOfTime60_89DaysPastDueNotWorse = st.sidebar.number_input(
        "Times 60-89 Days Past Due", min_value=0, max_value=20, value=0)
    NumberOfDependents = st.sidebar.number_input("Number of Dependents", min_value=0, max_value=20, value=0)

    data = {
        "RevolvingUtilizationOfUnsecuredLines": RevolvingUtilizationOfUnsecuredLines,
        "Age": Age,
        "NumberOfTime30-59DaysPastDueNotWorse": NumberOfTime30_59DaysPastDueNotWorse,
        "DebtRatio": DebtRatio,
        "MonthlyIncome": MonthlyIncome,
        "NumberOfOpenCreditLinesAndLoans": NumberOfOpenCreditLinesAndLoans,
        "NumberOfTimes90DaysLate": NumberOfTimes90DaysLate,
        "NumberRealEstateLoansOrLines": NumberRealEstateLoansOrLines,
        "NumberOfTime60-89DaysPastDueNotWorse": NumberOfTime60_89DaysPastDueNotWorse,
        "NumberOfDependents": NumberOfDependents
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ---------------------------
# Predictions
# ---------------------------
st.subheader("Predictions")

# Logistic Regression
scaled_input = scaler.transform(input_df)
logreg_prob = logreg_model.predict_proba(scaled_input)[:,1][0]
logreg_pred = logreg_model.predict(scaled_input)[0]

# XGBoost
xgb_prob = xgb_model.predict_proba(input_df)[:,1][0]
xgb_pred = xgb_model.predict(input_df)[0]

st.write("### Logistic Regression")
st.write(f"Prediction: {'Delinquent' if logreg_pred==1 else 'Not Delinquent'}")
st.write(f"Probability of 90-day delinquency: {logreg_prob:.2f}")

st.write("### XGBoost")
st.write(f"Prediction: {'Delinquent' if xgb_pred==1 else 'Not Delinquent'}")
st.write(f"Probability of 90-day delinquency: {xgb_prob:.2f}")

# ---------------------------
# SHAP Explainability
# ---------------------------
st.subheader("SHAP Explainability (XGBoost)")

try:
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(input_df)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.force_plot(explainer.expected_value, shap_values, input_df, matplotlib=True)
    st.pyplot(bbox_inches='tight')
except Exception as e:
    st.warning(f"SHAP explainability could not be rendered: {e}")

# ---------------------------
# Batch Predictions via CSV from R2
# ---------------------------
st.subheader("Batch Predictions via CSV (Cloudflare R2)")

uploaded_file = st.file_uploader("Upload CSV (or leave blank to fetch from R2)", type=["csv"])

try:
    if uploaded_file:
        batch_data = pd.read_csv(uploaded_file)
    else:
        st.info("Fetching sample CSV from R2 bucket...")
        # Example: replace 'batch_input.csv' with your actual filename in R2
        batch_data = load_csv_from_r2("batch_input.csv")

    batch_scaled = scaler.transform(batch_data)
    batch_logreg_probs = logreg_model.predict_proba(batch_scaled)[:,1]
    batch_xgb_probs = xgb_model.predict_proba(batch_data)[:,1]

    batch_data["LogReg_Prob"] = batch_logreg_probs
    batch_data["XGB_Prob"] = batch_xgb_probs

    st.dataframe(batch_data)
    st.download_button(
        "Download Predictions CSV",
        batch_data.to_csv(index=False),
        "predictions.csv"
    )

except Exception as e:
    st.error(f"Failed batch prediction: {e}")