import streamlit as st
import pandas as pd
import numpy as np
import joblib
import boto3
from io import BytesIO
import shap
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ---------------------------
# Streamlit Setup
# ---------------------------
st.set_page_config(page_title="Credit Scoring System", layout="wide")
st.title("Credit Scoring & Loan Decision System")

# ---------------------------
# Feature Columns
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
# Data Cleaning Function
# ---------------------------
def clean_numeric_columns(df):
    df_clean = df.applymap(
        lambda x: float(str(x).replace("[", "").replace("]", "").replace("'", "").replace('"',''))
        if isinstance(x, str) else x
    )
    return df_clean

# ---------------------------
# Logging / Monitoring Function
# ---------------------------
LOG_FILE = "logs/predictions_log.csv"

def log_prediction(df, log_file=LOG_FILE):
    """Logs predictions with timestamp to CSV"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    df_to_log = df.copy()
    df_to_log['Timestamp'] = datetime.now()
    
    if os.path.exists(log_file):
        df_to_log.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df_to_log.to_csv(log_file, index=False)

# ---------------------------
# Load Data from R2 (Optional)
# ---------------------------
st.sidebar.header("Data Source")
data_df = None

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
    file_name = next((obj['Key'] for obj in objects['Contents'] if obj['Key'].endswith('.csv')), None)
    obj = s3.get_object(Bucket=R2_BUCKET, Key=file_name)
    data_df = pd.read_csv(BytesIO(obj['Body'].read()))

    data_df = clean_numeric_columns(data_df)
    data_df.fillna(data_df.median(), inplace=True)

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

# ---------------------------
# Load Models
# ---------------------------
try:
    logreg_model = joblib.load("models/logreg_v2.pkl")
    xgb_model = joblib.load("models/xgb_best.pkl")
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
    df.fillna(df.median(), inplace=True)

    # Feature engineering
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
logreg_prob = logreg_model.predict_proba(scaled_input)[:,1][0]
logreg_pred = logreg_model.predict(scaled_input)[0]

xgb_prob = xgb_model.predict_proba(input_df)[:,1][0]
xgb_pred = xgb_model.predict(input_df)[0]

st.write("### Logistic Regression")
st.write(f"{'Delinquent' if logreg_pred else 'Not Delinquent'}")
st.write(f"Probability: {logreg_prob:.2f}")

st.write("### XGBoost")
st.write(f"{'Delinquent' if xgb_pred else 'Not Delinquent'}")
st.write(f"Probability: {xgb_prob:.2f}")

# ---------------------------
# Log single prediction
# ---------------------------
log_df = input_df.copy()
log_df['LogReg_Prob'] = logreg_prob
log_df['XGB_Prob'] = xgb_prob
log_prediction(log_df)

# ---------------------------
# SHAP Explainability & Business Interpretation
# ---------------------------
st.subheader("SHAP Explainability & Business Interpretation")

try:
    if data_df is not None:
        background = data_df.sample(min(50, len(data_df)))
    else:
        background = input_df.copy()

    background_array = clean_numeric_columns(background).fillna(0).to_numpy(dtype=float)
    input_array = clean_numeric_columns(input_df).fillna(0).to_numpy(dtype=float)

    explainer = shap.Explainer(lambda x: xgb_model.predict_proba(x)[:,1], background_array)
    shap_values = explainer(input_array)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    # Business Interpretation
    st.markdown("### Business Interpretation")
    feature_impact = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "SHAP_Value": shap_values.values[0]
    }).sort_values(by="SHAP_Value", key=abs, ascending=False)

    for i, row in feature_impact.iterrows():
        direction = "increases" if row['SHAP_Value'] > 0 else "decreases"
        st.write(f"- {row['Feature']} {direction} the likelihood of delinquency (impact: {row['SHAP_Value']:.2f})")

    top_features = feature_impact.head(3)
    st.write("**Top 3 factors influencing this prediction:**")
    for i, row in top_features.iterrows():
        direction = "increases" if row['SHAP_Value'] > 0 else "decreases"
        st.write(f"{row['Feature']} {direction} the risk of delinquency.")

except Exception as e:
    st.warning(f"SHAP failed: {e}")

# ---------------------------
# Batch Predictions
# ---------------------------
st.subheader("Batch Predictions")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    batch = pd.read_csv(file)
    batch = clean_numeric_columns(batch)
    batch.fillna(batch.median(), inplace=True)

    batch['TotalPastDue'] = (
        batch['NumberOfTime30-59DaysPastDueNotWorse'] +
        batch['NumberOfTime60-89DaysPastDueNotWorse'] +
        batch['NumberOfTimes90DaysLate']
    )
    batch['DebtPerIncome'] = batch['DebtRatio'] * batch['MonthlyIncome']

    batch_features = batch[FEATURE_COLUMNS]
    batch_scaled = scaler.transform(batch_features)

    batch["LogReg_Prob"] = logreg_model.predict_proba(batch_scaled)[:,1]
    batch["XGB_Prob"] = xgb_model.predict_proba(batch_features)[:,1]

    # Log batch predictions
    log_prediction(batch)

    st.dataframe(batch)
    st.download_button("Download Predictions", batch.to_csv(index=False), "predictions.csv")