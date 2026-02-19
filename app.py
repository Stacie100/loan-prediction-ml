import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="💰",
    layout="wide"
)
import pandas as pd
import joblib
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("model/model.pkl")

model = load_model()



# Title
st.title("💰 Loan Approval Prediction System")
st.markdown("Predict loan approval status using Machine Learning")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    st.subheader("Financial Information")
    applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=150)
    loan_term = st.number_input("Loan Term (months)", min_value=0, value=360)
    credit_history = st.selectbox("Credit History", ["Good (1.0)", "Bad (0.0)"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Predict button
if st.button("🔍 Predict Loan Status", type="primary"):
    # Encode inputs
    female_encoded = 0 if gender == "Male" else 1
    married_encoded = 1 if married == "Yes" else 0
    dependents_encoded = 3 if dependents == "3+" else int(dependents)
    education_encoded = 1 if education == "Graduate" else 0
    self_employed_encoded = 1 if self_employed == "Yes" else 0
    credit_history_encoded = 1.0 if credit_history == "Good (1.0)" else 0.0
    property_area_encoded = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]
    
    # Create feature array in correct order
    features = pd.DataFrame({
        'female': [female_encoded],
        'married': [married_encoded],
        'dependents': [dependents_encoded],
        'education': [education_encoded],
        'self_employed': [self_employed_encoded],
        'applicantincome': [applicant_income],
        'coapplicantincome': [coapplicant_income],
        'loanamount': [loan_amount],
        'loan_amount_term': [loan_term],
        'credit_history': [credit_history_encoded],
        'property_area': [property_area_encoded]
    })
    
    # Predict
    prediction_proba = model.predict_proba(features)[0]
    prediction = (prediction_proba[1] > 0.40).astype(int)
    
    # Display results
    st.markdown("---")
    st.subheader("Prediction Results")
    
    if prediction == 1:
        st.success("✅ **Loan Status: APPROVED**")
        confidence = prediction_proba[1] * 100
    else:
        st.error("❌ **Loan Status: REJECTED**")
        confidence = prediction_proba[0] * 100
    
    st.metric("Confidence Level", f"{confidence:.2f}%")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rejection Probability", f"{prediction_proba[0]:.2%}")
    with col2:
        st.metric("Approval Probability", f"{prediction_proba[1]:.2%}")

st.markdown("---")
st.markdown("**Note:** This is a ML prediction for informational purposes only.")
