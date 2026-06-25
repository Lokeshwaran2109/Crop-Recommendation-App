import streamlit as st
import numpy as np
import os
import joblib

# -------------------------------
# Page config (must be first)
# -------------------------------
st.set_page_config(page_title="Crop Recommendation", layout="centered")

# -------------------------------
# Safe loader (CACHE ENABLED)
# -------------------------------
@st.cache_resource
def load_model(file_path):
    return joblib.load(file_path)

# -------------------------------
# File paths (STREAMLIT SAFE)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "crop_model.pkl")
encoder_path = os.path.join(BASE_DIR, "label_encoder.pkl")

# Load model + encoder
model = load_model(model_path)
label_encoder = load_model(encoder_path)

# Optional scaler
scaler = None
use_scaler = False

scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
if os.path.exists(scaler_path):
    scaler = load_model(scaler_path)
    use_scaler = True

# -------------------------------
# UI
# -------------------------------
st.title("🌾 Intelligent Crop Recommendation System")
st.markdown("Enter soil nutrients and climate conditions")

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

    submit = st.form_submit_button("🌱 Recommend Crop")

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
