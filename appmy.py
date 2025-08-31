import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
# Load model + preprocessing artifacts
MODEL_PATH = "models/best_model_rf.joblib"
ARTIFACTS_PATH = "artifacts/preprocessing.json"

@st.cache_resource
def load_model_and_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(ARTIFACTS_PATH, "r") as f:
        artifacts = json.load(f)
    return model, artifacts

model, artifacts = load_model_and_artifacts()

# Extract preprocessing info
cat_cols = artifacts["cat_cols"]
cat_levels = artifacts["cat_levels"]
num_caplog_cols = artifacts["num_caplog_cols"]
maybe_num_cols = artifacts["maybe_num_cols"]
caps = artifacts["caps"]
feature_order = artifacts["feature_order"]
drop_leakage = artifacts["drop_leakage"]

# Preprocessing helper
def preprocess_input(user_input: dict):
    df = pd.DataFrame([user_input])

    # Apply capping + log1p
    for col in num_caplog_cols:
        lower = caps[col]["lower"]
        upper = caps[col]["upper"]
        df[col] = df[col].clip(lower, upper)
        df[col] = np.log1p(df[col])

    # Encode categoricals
    for col in cat_cols:
        levels = cat_levels[col]
        mapping = {v: i for i, v in enumerate(levels)}
        df[col] = df[col].astype(str).map(mapping)
        if df[col].isna().any():
            df[col] = -1

    # Drop leakage columns
    for col in drop_leakage:
        if col in df:
            df = df.drop(columns=[col])

    # Reorder columns
    df = df.reindex(columns=feature_order, fill_value=0)

    return df

# Streamlit UI
st.set_page_config(page_title="Crop Yield Predictor 🌾", layout="centered")

st.title("🌾 Crop Yield Prediction App")
st.write("Fill in the details below to predict the **expected crop yield** in simple terms.")

user_input = {}

#Dropdowns for categoricals
for col in cat_cols:
    user_input[col] = st.selectbox(f"Select {col}", options=cat_levels[col])

#Numeric inputs with real units
area = st.slider("🌱 Enter Area (hectares)", min_value=0.1, max_value=1000.0, value=1.0, step=0.1)
rainfall = st.slider("🌧️ Enter Annual Rainfall (mm)", min_value=100.0, max_value=3000.0, value=1000.0, step=10.0)
fertilizer = st.slider("💊 Enter Fertilizer (kg/hectare)", min_value=0.0, max_value=500.0, value=50.0, step=1.0)
pesticide = st.slider("🐛 Enter Pesticide (kg/hectare)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
crop_year = st.selectbox("📅 Select Crop Year", options=list(range(2000, 2026)))

# Add these to input dict
user_input["Area"] = area
user_input["Annual_Rainfall"] = rainfall
user_input["Fertilizer"] = fertilizer
user_input["Pesticide"] = pesticide
user_input["Crop_Year"] = crop_year

# Prediction
if st.button("🔮 Predict Yield"):
    X_input = preprocess_input(user_input)
    y_log_pred = model.predict(X_input)
    y_pred = np.expm1(y_log_pred)[0]  # invert log1p

    # Output: yield per hectare + total production
    st.success(f"✅ Expected Yield: **{y_pred:.2f} tons/hectare**")
    st.info(f"📦 For your {area:.1f} hectares, the total expected production is **{y_pred*area:.2f} tons**.")

