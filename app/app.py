import streamlit as st
import requests as rs
import json

st.title("Bank Customer Churn Prediction")

CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.number_input("Age", min_value=18, max_value=100, value=35)
Tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
Balance = st.number_input("Balance", min_value=0.0, value=75000.0)
NumOfProducts = st.number_input("Num Of Products", min_value=1, max_value=4, value=2)
HasCrCard = st.radio("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
IsActiveMember = st.radio("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)


def get_prediction(data):
    url = "http://api:8086/predict"
    headers = {"Content-Type": "application/json"}
    response = rs.post(url, data=json.dumps(data), headers=headers)
    return response.json()


if st.button("Predict Churn"):
    data = {
        "CreditScore": int(CreditScore),
        "Geography": str(Geography),
        "Gender": str(Gender),
        "Age": int(Age),
        "Tenure": int(Tenure),
        "Balance": float(Balance),
        "NumOfProducts": int(NumOfProducts),
        "HasCrCard": int(HasCrCard),
        "IsActiveMember": int(IsActiveMember),
        "EstimatedSalary": float(EstimatedSalary)
    }

    with st.spinner('Predicting...'):
        result = get_prediction(data)
    
    st.write(result)
