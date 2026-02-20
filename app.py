import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model/loan_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.title("🏦 Loan Prediction App")
st.write("Fill in the details below to predict loan approval.")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", [0, 1, 2, 3])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=0)
credit_history = st.selectbox("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

input_data = pd.DataFrame({
    'married':                  [1 if married == "Yes" else 0],
    'dependents':               [dependents],
    'self_employed':            [1 if self_employed == "Yes" else 0],
    'applicantincome':          [applicant_income],
    'coapplicantincome':        [coapplicant_income],
    'loanamount':               [loan_amount],
    'loan_amount_term':         [loan_amount_term],
    'credit_history':           [credit_history],
    'female_Male':              [1 if gender == "Male" else 0],
    'education_Not Graduate':   [1 if education == "Not Graduate" else 0],
    'property_area_Semiurban':  [1 if property_area == "Semiurban" else 0],
    'property_area_Urban':      [1 if property_area == "Urban" else 0],
})

numeric_cols = ['applicantincome', 'coapplicantincome', 'loanamount', 'loan_amount_term']
input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

if st.button("Predict"):
    prob = model.predict_proba(input_data)[:, 1][0]
    prediction = int(prob > 0.43)
    if prediction == 1:
        st.success(f"✅ Loan Approved! (Confidence: {prob:.2%})")
    else:
        st.error(f"❌ Loan Denied. (Confidence: {1-prob:.2%})")
