import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# تحميل النموذج المحفوظ
model = load_model("bank_churn_model.h5")

st.title("Bank Churn Prediction Dashboard")
st.write("Enter the customer details below to predict if they will leave the bank:")

# خانات الإدخال الأساسية
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
age = st.number_input("Age", min_value=18, max_value=100, value=35)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=1)

if st.button("Predict Churn"):
    # الموديل متدرب على 12 عمود، هندخل الـ 5 اللي معانا ونكمل الباقي بأصفار أو قيم افتراضية
    input_data = np.array([[credit_score, age, balance, estimated_salary, num_of_products, 0, 0, 0, 0, 1, 0, 0]])
    
    prediction = model.predict(input_data)
    is_churning = prediction[0][0] > 0.5
    
    if is_churning:
        st.error(f"⚠️ Warning: This customer is likely to churn! (Probability: {prediction[0][0]:.2f})")
    else:
        st.success(f"✅ Stable Customer. (Probability: {prediction[0][0]:.2f})")


