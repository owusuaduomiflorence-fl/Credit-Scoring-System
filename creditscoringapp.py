import streamlit as st
import pandas as pd
import numpy as np
import joblib
import boto3
from io import BytesIO
import shap
import matplotlib.pyplot as plt
import re

# ---------------------------
# Streamlit Setup
# ---------------------------
st.set_page_config(page_title="Credit Scoring System", layout="wide")
st.title("Credit Scoring & Loan Decision System")

# ---------------------------
# Feature Columns (CRITICAL)
# ---------------------------
FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "Age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "TotalPastDue",
    "DebtPerIncome"
]

# ---------------------------
# Data Cleaning
# ---------------------------
def clean_numeric_columns(df):
    def to_float(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            # Remove brackets, quotes, spaces
            x = re.sub(r"[\[\]'\" ]", "", x)
        try:
            return float(x)
        except:
            return np.nan
    return df.applymap(to_float)

# ---------------------------
# Load Data from R2
# ---------------------------
st.sidebar.header("Data Source")
try:
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
        (obj['Key'] for obj in objects['Contents'] if obj['Key'].endswith('.csv')),
        None
    )
    obj = s3.get_object(Bucket=R2_BUCKET, Key=file_name)
    data_df = pd.read_csv(BytesIO(obj['Body'].read()))
    
    data_df = clean_numeric_columns(data_df)
    
    # Feature engineering
    data_df['TotalPastDue'] = (
        data_df['NumberOfTime30-59DaysPastDueNotWorse'] +
        data_df['NumberOfTime60-89DaysPastDueNotWorse'] +
        data_df['NumberOfTimes90DaysLate']
    )
    data_df['DebtPerIncome'] = data_df['DebtRatio'] * data_df['MonthlyIncome']
    
    data_df = data_df[FEATURE_COLUMNS]
    st.success(f"Loaded dataset: {file_name}")
except Exception as e:
    st.warning(f"Could not load R2 dataset: {e}")
    data_df = None

# ---------------------------
# Load Models
# ---------------------------
try:
    logreg_model = joblib.load("models/logreg_v2.pkl")
    xgb_model = joblib.load("models/xgb_v2.pkl")
    scaler = joblib.load("models/scaler_v2.pkl")
    st.success("Models loaded successfully")
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# ---------------------------
# User Input
# ---------------------------
st.sidebar.header("Customer Input")
def user_input_features():
    df = pd.DataFrame({
        "RevolvingUtilizationOfUnsecuredLines":[st.sidebar.number_input("Utilization",0.0,10.0,0.5)],
        "Age":[st.sidebar.number_input("Age",18,100,30)],
        "NumberOfTime30-59DaysPastDueNotWorse":[st.sidebar.number_input("30-59 Days Late",0,20,0)],
        "DebtRatio":[st.sidebar.number_input("Debt Ratio",0.0,10.0,0.2)],
        "MonthlyIncome":[st.sidebar.number_input("Income",0,1000000,3000)],
        "NumberOfOpenCreditLinesAndLoans":[st.sidebar.number_input("Credit Lines",0,50,5)],
        "NumberOfTimes90DaysLate":[st.sidebar.number_input("90 Days Late",0,20,0)],
        "NumberRealEstateLoansOrLines":[st.sidebar.number_input("Real Estate Loans",0,20,1)],
        "NumberOfTime60-89DaysPastDueNotWorse":[st.sidebar.number_input("60-89 Days Late",0,20,0)],
        "NumberOfDependents":[st.sidebar.number_input("Dependents",0,20,0)]
    })

    df = clean_numeric_columns(df)
    df['TotalPastDue'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] +
        df['NumberOfTime60-89DaysPastDueNotWorse'] +
        df['NumberOfTimes90DaysLate']
    )
    df['DebtPerIncome'] = df['DebtRatio'] * df['MonthlyIncome']
    
    return df[FEATURE_COLUMNS]

input_df = user_input_features()

# ---------------------------
# Predictions
# ---------------------------
st.subheader("Predictions")

scaled_input = scaler.transform(input_df)
logreg_prob = logreg_model.predict_proba(scaled_input)[0][1]
logreg_pred = logreg_model.predict(scaled_input)[0]

xgb_prob = xgb_model.predict_proba(input_df)[0][1]
xgb_pred = xgb_model.predict(input_df)[0]

st.write("### Logistic Regression")
st.write(f"{'Delinquent' if logreg_pred else 'Not Delinquent'}")
st.write(f"Probability: {logreg_prob:.2f}")

st.write("### XGBoost")
st.write(f"{'Delinquent' if xgb_pred else 'Not Delinquent'}")
st.write(f"Probability: {xgb_prob:.2f}")

# ---------------------------
# SHAP Explainability (FIXED)
# ---------------------------
st.subheader("SHAP Explainability")
try:
    # Use a small numeric background
    background = data_df.sample(min(50, len(data_df))) if data_df is not None else input_df.copy()
    
    # Ensure fully numeric
    background = clean_numeric_columns(background).fillna(0)
    input_array = clean_numeric_columns(input_df).fillna(0)

    # Use lambda-based explainer to avoid string conversion errors
    explainer = shap.Explainer(lambda x: xgb_model.predict_proba(x)[:,1], background)
    shap_values = explainer(input_array)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

except Exception as e:
    st.warning(f"SHAP failed: {e}")

# ---------------------------
# Batch Prediction
# ---------------------------
st.subheader("Batch Predictions")
file = st.file_uploader("Upload CSV")
if file:
    batch = pd.read_csv(file)
    batch = clean_numeric_columns(batch)
    batch['TotalPastDue'] = (
        batch['NumberOfTime30-59DaysPastDueNotWorse'] +
        batch['NumberOfTime60-89DaysPastDueNotWorse'] +
        batch['NumberOfTimes90DaysLate']
    )
    batch['DebtPerIncome'] = batch['DebtRatio'] * batch['MonthlyIncome']
    batch = batch[FEATURE_COLUMNS]

    batch_scaled = scaler.transform(batch)
    batch["LogReg_Prob"] = logreg_model.predict_proba(batch_scaled)[:,1]
    batch["XGB_Prob"] = xgb_model.predict_proba(batch)[:,1]

    st.dataframe(batch)
    st.download_button("Download Predictions", batch.to_csv(index=False), "predictions.csv")