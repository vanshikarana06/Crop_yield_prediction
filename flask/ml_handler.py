import pandas as pd
import numpy as np
import joblib
import json

MODEL_PATH = "models/best_model_rf.joblib"
ARTIFACTS_PATH = "artifacts/preprocessing.json"

# Load once on import
model = joblib.load(MODEL_PATH)
with open(ARTIFACTS_PATH, "r") as f:
    artifacts = json.load(f)

def preprocess_and_predict(user_input: dict):
    df = pd.DataFrame([user_input])

    # 1. Apply capping + log1p
    caps = artifacts["caps"] 
    for col in artifacts["num_caplog_cols"]:
        lower, upper = caps[col]["lower"], caps[col]["upper"]
        df[col] = df[col].clip(lower, upper)
        df[col] = np.log1p(df[col])

    # 2. Encode categoricals
    cat_levels = artifacts["cat_levels"]
    for col in artifacts["cat_cols"]:
        mapping = {v: i for i, v in enumerate(cat_levels[col])}
        df[col] = df[col].astype(str).map(mapping).fillna(-1)

    # 3. Drop & Reorder
    df = df.drop(columns=[c for c in artifacts["drop_leakage"] if c in df])
    df = df.reindex(columns=artifacts["feature_order"], fill_value=0)

    # 4. Predict and Invert Log
    y_log_pred = model.predict(df)
    y_pred = np.expm1(y_log_pred)[0]
    return float(y_pred)


