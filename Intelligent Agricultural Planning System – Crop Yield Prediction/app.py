import streamlit as st
import numpy as np
import pickle
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Crop Recommendation System", layout="centered")

# -------------------------------
# SAFE MODEL LOADER (IMPORTANT FIX)
# -------------------------------
def load_pickle(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, filename)

    with open(file_path, "rb") as f:
        return pickle.load(f)

# -------------------------------
# Load models safely
# -------------------------------
model = load_pickle("crop_model.pkl")
label_encoder = load_pickle("label_encoder.pkl")

# Optional scaler
scaler = None
use_scaler = False

try:
    scaler = load_pickle("scaler.pkl")
    use_scaler = True
except:
    use_scaler = False

# -------------------------------
# UI
# -------------------------------
st.title("🌾 Intelligent Crop Recommendation System")
st.markdown("Enter soil nutrients and climate values")

with st.form("form"):
    col1, col2 = st.columns(2)

    with col1:
        N = st.slider("Nitrogen (N)", 0, 140, 70)
        K = st.slider("Potassium (K)", 5, 205, 50)
        temperature = st.slider("Temperature (°C)", 10, 45, 25)
        ph = st.slider("Soil pH", 3.5, 9.5, 6.5)

    with col2:
        P = st.slider("Phosphorous (P)", 5, 145, 50)
        humidity = st.slider("Humidity (%)", 10, 100, 65)
        rainfall = st.slider("Rainfall (mm)", 20, 300, 100)

    submit = st.form_submit_button("🌱 Predict Crop")

# -------------------------------
# Prediction
# -------------------------------
if submit:
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    if use_scaler:
        input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    crop = label_encoder.inverse_transform(prediction)[0]

    st.success(f"🌾 Recommended Crop: **{crop.capitalize()}**")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("Made by Lokeshwaran | ML + Streamlit Project")
