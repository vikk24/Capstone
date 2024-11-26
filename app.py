import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model and scaler
model = joblib.load('path_to_your_best_rf_model.pkl')
scaler = joblib.load('path_to_your_scaler.pkl')

# Create the app's UI
st.title('Car Selling Price Prediction')

# Input features from the user
year = st.number_input("Year", min_value=2000, max_value=2024, step=1)
km_driven = st.number_input("Kilometers Driven")
transmission = st.selectbox("Transmission", options=["Automatic", "Manual"])
owner = st.selectbox("Owner", options=["First Owner", "Second Owner", "Third Owner", "Fourth & Above"])
fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "Electric", "LPG"])
seller_type = st.selectbox("Seller Type", options=["Individual", "Trustmark Dealer"])

# Convert input values to a DataFrame (match your model's input format)
input_data = pd.DataFrame({
    'year': [year],
    'km_driven': [km_driven],
    'transmission': [transmission],
    'owner': [owner],
    'fuel_type': [fuel_type],
    'seller_type': [seller_type],
})

# Preprocess input data (one-hot encoding, scaling, etc.)
input_data = pd.get_dummies(input_data)  # Assuming you need one-hot encoding for categorical features
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)

# Show the prediction
st.write(f"Predicted Selling Price: ₹{prediction[0]:,.2f}")
