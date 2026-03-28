import streamlit as st
import pandas as pd
import numpy as np
import joblib
import boto3
from io import BytesIO
import shap

# ---------------------------
# PAGE SETUP
# ---------------------------
st.set_page_config(page_title="Credit Scoring System", layout="wide")
st.title("Credit Scoring & Loan Decision System")

# ---------------------------
# CLEANING FUNCTION (BEST FIX)
# ---------------------------
def clean_numeric(df):
    def convert(x):
        try:
            x = str(x).replace("[", "").replace("]", "").strip()
            return float(x)
        except:
            return np.nan
    return df.applymap(convert)

# ---------------------------
# LOAD DATA FROM R2
# ---------------------------
try:
    st.info("Connecting to Cloudflare R2...")

    s3 = boto3.client(
        "s3",
        endpoint_url=st.secrets["R2_ENDPOINT_URL"],
        aws_access_key_id=st.secrets["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["R2_SECRET_ACCESS_KEY"]
    )

    bucket = st.secrets["R2_BUCKET_NAME"]

    objects = s3.list_objects_v2(Bucket=bucket)
    file_name = next(
        (obj["Key"] for obj in objects["Contents"] if obj["Key"].endswith(".csv")),
        None
    )

    obj = s3.get_object(Bucket=bucket, Key=file_name)
    data_df = pd.read_csv(BytesIO(obj["Body"].read()))

    st.success(f"Loaded dataset: {file_name}")

except Exception as e:
    st.error(f"R2 Error: {e}")
    st.stop()

# ---------------------------
# LOAD MODELS
# ---------------------------
try:
    logreg_model = joblib.load("models/logreg_v1.pkl")
    xgb_model = joblib.load("models/xgb_v1.pkl")
    scaler = joblib.load("models/scaler_v1.pkl")
    st.success("Models loaded successfully")

except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# ---------------------------
# USER INPUT
# ---------------------------
st.sidebar.header("Customer Input")

def get_input():
    data = {
        "RevolvingUtilizationOfUnsecuredLines": st.sidebar.number_input("Utilization", 0.0, 10.0, 0.5),
        "Age": st.sidebar.number_input("Age", 18, 100, 30),
        "NumberOfTime30-59DaysPastDueNotWorse": st.sidebar.number_input("30-59 Days Late", 0, 20, 0),
        "DebtRatio": st.sidebar.number_input("Debt Ratio", 0.0, 10.0, 0.2),
        "MonthlyIncome": st.sidebar.number_input("Income", 0, 100000, 3000),
        "NumberOfOpenCreditLinesAndLoans": st.sidebar.number_input("Credit Lines", 0, 50, 5),
        "NumberOfTimes90DaysLate": st.sidebar.number_input("90 Days Late", 0, 20, 0),
        "NumberRealEstateLoansOrLines": st.sidebar.number_input("Real Estate Loans", 0, 20, 1),
        "NumberOfTime60-89DaysPastDueNotWorse": st.sidebar.number_input("60-89 Days Late", 0, 20, 0),
        "NumberOfDependents": st.sidebar.number_input("Dependents", 0, 10, 0),
    }

    df = pd.DataFrame([data])

    # ---------------------------
    # FEATURE ENGINEERING
    # ---------------------------
    df["TotalPastDue"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] +
        df["NumberOfTime60-89DaysPastDueNotWorse"] +
        df["NumberOfTimes90DaysLate"]
    )

    df["DebtPerIncome"] = df["DebtRatio"] * df["MonthlyIncome"]

    return df

input_df = get_input()

# ---------------------------
# CLEAN + FORCE NUMERIC
# ---------------------------
input_df = clean_numeric(input_df)

# ---------------------------
# FEATURE ORDER (CRITICAL)
# ---------------------------
FEATURE_ORDER = [
    'RevolvingUtilizationOfUnsecuredLines',
    'Age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents',
    'TotalPastDue',
    'DebtPerIncome'
]

input_df = input_df[FEATURE_ORDER]

# ---------------------------
# PREDICTIONS
# ---------------------------
st.subheader("Predictions")

scaled_input = scaler.transform(input_df)

logreg_prob = logreg_model.predict_proba(scaled_input)[0][1]
logreg_pred = logreg_model.predict(scaled_input)[0]

xgb_prob = xgb_model.predict_proba(input_df)[0][1]
xgb_pred = xgb_model.predict(input_df)[0]

st.write("### Logistic Regression")
st.write(f"Prediction: {'Delinquent' if logreg_pred else 'Not Delinquent'}")
st.write(f"Probability: {logreg_prob:.2f}")

st.write("### XGBoost")
st.write(f"Prediction: {'Delinquent' if xgb_pred else 'Not Delinquent'}")
st.write(f"Probability: {xgb_prob:.2f}")

# ---------------------------
# SHAP EXPLAINABILITY (FIXED)
# ---------------------------
st.subheader("SHAP Explainability")

try:
    # Ensure clean numeric input
    shap_input = clean_numeric(input_df)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(shap_input)

    # Plot
    shap.plots.waterfall(shap_values[0])
    st.pyplot(bbox_inches="tight")

except Exception as e:
    st.warning(f"SHAP failed: {e}")

# ---------------------------
# BATCH PREDICTION
# ---------------------------
st.subheader("Batch Prediction")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    # Feature engineering
    df["TotalPastDue"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] +
        df["NumberOfTime60-89DaysPastDueNotWorse"] +
        df["NumberOfTimes90DaysLate"]
    )

    df["DebtPerIncome"] = df["DebtRatio"] * df["MonthlyIncome"]

    # CLEAN DATA (IMPORTANT)
    df = clean_numeric(df)

    df = df[FEATURE_ORDER]

    scaled = scaler.transform(df)

    df["LogReg_Prob"] = logreg_model.predict_proba(scaled)[:,1]
    df["XGB_Prob"] = xgb_model.predict_proba(df)[:,1]

    st.dataframe(df)

    st.download_button(
        "Download Predictions",
        df.to_csv(index=False),
        "predictions.csv"
    )