import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Attack Predictor", layout="centered")
st.title("💓 Heart Attack Risk Prediction App")

def user_input():
    age = st.slider("Age", 20, 90, 45)
    cholesterol = st.slider("Cholesterol Level", 100, 400, 200)
    bp_sys = st.slider("BP Systolic", 80, 200, 120)
    bp_dia = st.slider("BP Diastolic", 60, 140, 80)
    sugar = st.slider("Blood Sugar", 50, 300, 100)
    hr = st.slider("Heart Rate", 60, 200, 120)
    smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes?", ["No", "Yes"])
    obesity = st.selectbox("Obesity?", ["No", "Yes"])
    family = st.selectbox("Family History of Heart Disease?", ["No", "Yes"])
    exercise = st.selectbox("Exercise Level", ["Low", "Medium", "High"])
    prev_issue = st.selectbox("Previous Heart Issues?", ["No", "Yes"])
    gender = st.radio("Gender", ["Male", "Female"])
    region = st.selectbox("Region", ["North", "South", "East", "West", "Central", "Northeast"])
    ecg = st.selectbox("ECG Result", ["Normal", "Abnormal", "Borderline"])

    # Binary conversions
    def yes_no(val): return 1 if val == "Yes" else 0

    gender_female = 1 if gender == "Female" else 0
    gender_male = 1 if gender == "Male" else 0

    region_cols = ["Region_Central", "Region_East", "Region_North", "Region_Northeast", "Region_South", "Region_West"]
    region_vals = [1 if region in col else 0 for col in region_cols]

    ecg_cols = ["ECG_Result_Abnormal", "ECG_Result_Borderline", "ECG_Result_Normal"]
    ecg_vals = [1 if ecg in col else 0 for col in ecg_cols]

    exercise_map = {"Low": 0, "Medium": 1, "High": 2}

    input_data = pd.DataFrame([[
        age, cholesterol, bp_sys, bp_dia, sugar, hr,
        yes_no(smoking), yes_no(diabetes), yes_no(obesity), yes_no(family),
        exercise_map[exercise], yes_no(prev_issue),
        gender_female, gender_male,
        *region_vals,
        *ecg_vals
    ]], columns=[
        'Age', 'Cholesterol_Level', 'BP_Systolic', 'BP_Diastolic',
        'Blood_Sugar', 'Heart_Rate', 'Smoking', 'Diabetes', 'Obesity',
        'Family_History', 'Exercise_Level', 'Previous_Heart_Issue',
        'Gender_Female', 'Gender_Male', 'Region_Central', 'Region_East',
        'Region_North', 'Region_Northeast', 'Region_South', 'Region_West',
        'ECG_Result_Abnormal', 'ECG_Result_Borderline', 'ECG_Result_Normal'
    ])
    
    return input_data

# Get user input
input_df = user_input()

# Scale input
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    pred = model.predict(scaled_input)
    prob = model.predict_proba(scaled_input)[0][1]

    if pred[0] == 1:
        st.error(f"⚠️ High risk of heart attack! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low risk of heart attack. (Probability: {prob:.2f})")
