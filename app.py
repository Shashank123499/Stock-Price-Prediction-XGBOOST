import streamlit as st
import joblib
import pandas as pd

# Load model and encoder
model = joblib.load("stock_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("📈 Stock Price Prediction App")

# Company names
company = st.selectbox(
    "Select Company",
    encoder.classes_
)

open_price = st.number_input("Open Price", min_value=0.0)
high = st.number_input("High Price", min_value=0.0)
low = st.number_input("Low Price", min_value=0.0)
volume = st.number_input("Volume", min_value=0.0)
dividends = st.number_input("Dividends", min_value=0.0)
stock_splits = st.number_input("Stock Splits", min_value=0.0)

if st.button("Predict Close Price"):

    company_encoded = encoder.transform([company])[0]

    data = pd.DataFrame({
        "Open": [open_price],
        "High": [high],
        "Low": [low],
        "Volume": [volume],
        "Dividends": [dividends],
        "Stock Splits": [stock_splits],
        "Company": [company_encoded]
    })

    prediction = model.predict(data)

    st.success(f"Predicted Closing Price: {prediction[0]:.2f}")
