
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Prediction", page_icon="📊")

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict whether they may leave the bank.")

model = joblib.load("models/churn_model.pkl")

age = st.number_input("Age", 18, 100, 40)
tenure = st.number_input("Tenure", 0, 10, 3)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
products = st.selectbox("Number of products", [1, 2, 3, 4])
credit_card = st.selectbox("Has credit card?", ["Yes", "No"])
active = st.selectbox("Is active member?", ["Yes", "No"])
salary = st.number_input("Estimated salary", min_value=0.0, value=100000.0)
credit_bins = st.selectbox("Credit score category", [0, 1, 2, 3, 4])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])

if st.button("Predict churn"):
    customer = pd.DataFrame([{
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": int(credit_card == "Yes"),
        "IsActiveMember": int(active == "Yes"),
        "EstimatedSalary": salary,
        "CreditScoreBins": credit_bins,
        "Geography_France": geography == "France",
        "Geography_Germany": geography == "Germany",
        "Geography_Spain": geography == "Spain",
        "Gender_Female": gender == "Female",
        "Gender_Male": gender == "Male"
    }])

    customer = customer[model.feature_names_in_]
    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.metric("Estimated churn probability", f"{probability:.1%}")

    if prediction == 1:
        st.error("Prediction: Customer is likely to leave.")
    else:
        st.success("Prediction: Customer is likely to stay.")

