import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Load Model & Encoders
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "xgboost_credit_model.pkl")
)

encoders = joblib.load(
    os.path.join(BASE_DIR, "encoders.pkl")
)

target_encoder = joblib.load(
    os.path.join(BASE_DIR, "target_encoder.pkl")
)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction")

st.write(
    "Enter the applicant's details below to determine whether the loan application "
    "is likely to be approved or rejected."
)
st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=30
    )

    sex = st.selectbox(
        "Sex",
        ["female", "male"]
    )

    job = st.selectbox(
        "Job Level",
        [0, 1, 2, 3],
        help="""
0 = Unskilled & Non-resident

1 = Unskilled & Resident

2 = Skilled

3 = Highly Skilled / Management
"""
    )

    housing = st.selectbox(
        "Housing",
        ["free", "own", "rent"]
    )

with col2:

    saving_accounts = st.selectbox(
        "Saving Accounts",
        ["little", "moderate", "quite rich", "rich"]
    )

    checking_account = st.selectbox(
        "Checking Account",
        ["little", "moderate", "rich"]
    )

    credit_amount = st.number_input(
        "Credit Amount",
        min_value=0,
        value=5000,
        step=500
    )

    duration = st.number_input(
        "Duration (Months)",
        min_value=1,
        value=24
    )

# -----------------------------
# Encode Inputs
# -----------------------------
sex_encoded = encoders["Sex"].transform([sex])[0]

housing_encoded = encoders["Housing"].transform([housing])[0]

saving_encoded = encoders["Saving accounts"].transform([saving_accounts])[0]

checking_encoded = encoders["Checking account"].transform([checking_account])[0]

# -----------------------------
# Create Input DataFrame
# -----------------------------
input_df = pd.DataFrame({

    "Age": [age],

    "Sex": [sex_encoded],

    "Job": [job],

    "Housing": [housing_encoded],

    "Saving accounts": [saving_encoded],

    "Checking account": [checking_encoded],

    "Credit amount": [credit_amount],

    "Duration": [duration]

})

# -----------------------------
# Show Input Data
# -----------------------------
with st.expander("View Input Data"):
    st.dataframe(input_df)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Loan Eligibility", use_container_width=True):

    prediction = model.predict(input_df)[0]

    prediction_label = target_encoder.inverse_transform([prediction])[0]

    probability = model.predict_proba(input_df)[0]

    confidence = probability[prediction]

    st.markdown("---")

    if prediction_label.lower() == "good":
        st.success("✅ **Prediction:** GOOD RISK")
        st.success("Loan is likely to be approved.")
    else:
        st.error("❌ **Prediction:** BAD RISK")
        st.error("Loan is likely to be rejected.")

    st.subheader("Prediction Confidence")

    st.progress(float(confidence))

    st.write(f"Confidence: {confidence*100:.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Good Risk",
            f"{probability[1]*100:.2f}%"
        )

    with col2:
        st.metric(
            "Bad Risk",
            f"{probability[0]*100:.2f}%"
        )

st.markdown("---")
st.caption("Machine Learning Model: XGBoost Classifier")