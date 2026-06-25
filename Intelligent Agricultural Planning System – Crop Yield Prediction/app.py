import streamlit as st
import numpy as np
import pickle

# -------------------------------
# Load saved model and encoders
# -------------------------------

st.cache_resource()
def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

model = load_pickle("crop_model.pkl")               # Trained ML model
label_encoder = load_pickle("label_encoder.pkl")    # Encoder for crop labels

# Load scaler (optional)
try:
    scaler = load_pickle("scaler.pkl")
    use_scaler = True
except:
    use_scaler = False

# -------------------------------
# App UI
# -------------------------------

st.set_page_config(page_title="Crop Recommendation", layout="centered")
st.title(" Intelligent Crop Recommendation System")
st.markdown(" Enter the soil nutrients and climate details to get the most suitable crop ")

# -------------------------------
# User Input Section
# -------------------------------

with st.form("crop_form"):
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

    submitted = st.form_submit_button(" Recommend Crop")

# -------------------------------
# Prediction Logic
# -------------------------------

if submitted:
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    if use_scaler:
        input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    predicted_crop = label_encoder.inverse_transform(prediction)[0]

    st.success(f"🌱 **Recommended Crop:** `{predicted_crop.capitalize()}`")

# -------------------------------
# Footer Section
# -------------------------------

st.markdown("---")
st.markdown("Made by Lokeshwaran  |  Powered by **Machine Learning + Streamlit")
