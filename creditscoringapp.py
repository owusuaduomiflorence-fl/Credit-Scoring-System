import streamlit as st
import pandas as pd
import numpy as np
import joblib
import boto3
import shap

# ---------------------------
# Cloudflare R2 Connection
# ---------------------------
# Get keys from Streamlit secrets
R2_ENDPOINT = st.secrets["R2_ENDPOINT_URL"]
R2_ACCESS_KEY = st.secrets["R2_ACCESS_KEY_ID"]
R2_SECRET_KEY = st.secrets["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = st.secrets["R2_BUCKET_NAME"]

# Function to load files from R2
def load_from_r2(file_name):
    obj = s3.get_object(Bucket=R2_BUCKET, Key=file_name)
    return obj['Body'].read()

# ---------------------------
# Load models & scaler
# ---------------------------
st.sidebar.title("Load Models from Cloudflare R2")
logreg_file = "logreg_v1.pkl"
xgb_file = "xgb_v1.pkl"
scaler_file = "scaler_v1.pkl"

# Load Logistic Regression
logreg_model = joblib.load(load_from_r2(logreg_file))
# Load XGBoost
xgb_model = joblib.load(load_from_r2(xgb_file))
# Load Scaler
scaler = joblib.load(load_from_r2(scaler_file))

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.title("End-to-End Credit Scoring System")
st.markdown("""
This app predicts the probability of 90-day delinquency for customers using Logistic Regression and XGBoost.
""")

# ---------------------------
# User Input Section
# ---------------------------
st.sidebar.header("Customer Data Input")

def user_input_features():
    RevolvingUtilizationOfUnsecuredLines = st.sidebar.number_input("Revolving Utilization (%)", min_value=0.0, max_value=10.0, value=0.5)
    Age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    NumberOfTime30_59DaysPastDueNotWorse = st.sidebar.number_input("Times 30-59 Days Past Due", min_value=0, max_value=20, value=0)
    DebtRatio = st.sidebar.number_input("Debt Ratio", min_value=0.0, max_value=10.0, value=0.2)
    MonthlyIncome = st.sidebar.number_input("Monthly Income", min_value=0, value=3000)
    NumberOfOpenCreditLinesAndLoans = st.sidebar.number_input("Open Credit Lines & Loans", min_value=0, max_value=50, value=5)
    NumberOfTimes90DaysLate = st.sidebar.number_input("Times 90 Days Late", min_value=0, max_value=20, value=0)
    NumberRealEstateLoansOrLines = st.sidebar.number_input("Real Estate Loans/Lines", min_value=0, max_value=20, value=1)
    NumberOfTime60_89DaysPastDueNotWorse = st.sidebar.number_input("Times 60-89 Days Past Due", min_value=0, max_value=20, value=0)
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
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# ---------------------------
# Prediction
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
# SHAP Explainability (Optional)
# ---------------------------
st.subheader("SHAP Explainability (XGBoost)")

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(input_df)

st.set_option('deprecation.showPyplotGlobalUse', False)
shap.force_plot(explainer.expected_value, shap_values, input_df, matplotlib=True)
st.pyplot(bbox_inches='tight')

# ---------------------------
# CSV Upload Option
# ---------------------------
st.subheader("Batch Predictions via CSV")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    batch_scaled = scaler.transform(batch_data)
    batch_logreg_probs = logreg_model.predict_proba(batch_scaled)[:,1]
    batch_xgb_probs = xgb_model.predict_proba(batch_data)[:,1]
    batch_data["LogReg_Prob"] = batch_logreg_probs
    batch_data["XGB_Prob"] = batch_xgb_probs
    st.dataframe(batch_data)
    st.download_button("Download Predictions CSV", batch_data.to_csv(index=False), "predictions.csv")