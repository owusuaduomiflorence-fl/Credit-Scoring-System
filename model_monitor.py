import pandas as pd
import joblib
from datetime import datetime
import os

LOG_FILE = "logs/predictions.log"

# Load models
xgb_model = joblib.load("models/xgb_best.pkl")
logreg_model = joblib.load("models/logreg_v2.pkl")
scaler = joblib.load("models/scaler_v2.pkl")

def log_prediction(input_df):
    """Logs predictions and timestamp"""
    input_scaled = scaler.transform(input_df)
    logreg_prob = logreg_model.predict_proba(input_scaled)[:,1]
    xgb_prob = xgb_model.predict_proba(input_df)[:,1]

    df_log = input_df.copy()
    df_log['LogReg_Prob'] = logreg_prob
    df_log['XGB_Prob'] = xgb_prob
    df_log['Timestamp'] = datetime.now()

    # Append to CSV log
    os.makedirs("logs", exist_ok=True)
    if os.path.exists(LOG_FILE):
        df_log.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df_log.to_csv(LOG_FILE, index=False)

    print(f"Logged {len(df_log)} predictions at {datetime.now()}")