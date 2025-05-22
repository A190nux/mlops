import streamlit as st
import requests as rs

CreditScore = st.text_input("Credit Score")
Geography = st.text_input("Geography")
Gender = st.text_input("Gender")
Age = st.text_input("Age")
Tenure = st.text_input("Tenure")
Balance = st.text_input("Balance")
NumOfProducts = st.text_input("Num Of Products")
HasCrCard = st.text_input("HasCrCard")
IsActiveMember = st.text_input("IsActiveMember")
EstimatedSalary = st.text_input("Estimated Salary")


def get_api(params):
    url = f"http://api:8086/predict/"
    response = rs.get(url, params=params)
    return response.content


if st.button("Get response"):
    params = {
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

    data = get_api(params)
    st.write(data)

