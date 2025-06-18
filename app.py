import sklearn
import streamlit as st
import numpy as np
import joblib


st.sidebar.markdown(f"**scikit-learn**: {sklearn.__version__}")
st.sidebar.markdown(f"**joblib**: {joblib.__version__}")
st.sidebar.markdown(f"**numpy**: {np.__version__}")

# Load model and encoder
model = joblib.load("models/smart_crop_recommender.pkl")
encoder = joblib.load("models/label_encoder.pkl")

st.title("🌾 Smart Crop Recommender")
st.markdown("Enter soil and climate data to predict the best crop to grow.")

# Input fields
N = st.number_input("Nitrogen (N)", 0, 140, 50)
P = st.number_input("Phosphorus (P)", 0, 140, 40)
K = st.number_input("Potassium (K)", 0, 200, 40)
temperature = st.slider("Temperature (°C)", 10.0, 40.0, 25.0)
humidity = st.slider("Humidity (%)", 10.0, 100.0, 60.0)
ph = st.slider("pH Level", 3.5, 9.5, 6.5)
rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

# Predict button
if st.button("Predict Crop"):
    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(sample)
    crop = encoder.inverse_transform(prediction)[0]
    st.success(f"🌱 Recommended Crop: **{crop.capitalize()}**")

# Footer
st.markdown("---")
st.markdown("Developed by [Mohamed Hassan]()")
st.markdown("Source code available on [GitHub](https://github.com/mohamed7456/smart-crop-recommender)")