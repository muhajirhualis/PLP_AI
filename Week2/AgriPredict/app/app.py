import streamlit as st
import joblib
import numpy as np

# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/rf_agripredict.pkl")

model = load_model()

st.set_page_config(
    page_title="🌾 AgriPredict",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 AgriPredict")
st.subheader("Maize Yield Forecast for Smallholder Farmers")
st.markdown("Enter agronomic conditions to forecast yield (tonnes/hectare).")

# Input sliders
ndvi = st.slider("Peak NDVI (0.2–0.9)", 0.2, 0.9, 0.65, 0.01)
rain = st.number_input("Cumulative Rainfall (60d, mm)", 100, 800, 350)
temp = st.slider("Mean Temp (°C)", 15.0, 32.0, 23.5, 0.5)
soc = st.slider("Soil Organic Carbon (%)", 0.3, 3.0, 1.8, 0.1)
elev = st.number_input("Elevation (m)", 200, 3000, 1500)
slope = st.slider("Slope (°)", 0, 45, 10)
ph = st.slider("Soil pH", 4.5, 8.0, 5.8, 0.1)
doy = st.slider("Planting Day of Year", 110, 150, 130)

# Predict
if st.button("🔮 Forecast Yield"):
    features = np.array([[ndvi, rain, temp, ph, soc, elev, slope, doy]])
    pred = model.predict(features)[0]
    st.success(f"### Predicted Maize Yield: **{pred:.2f} tonnes/ha**")
    
    st.info("""
    ℹ️ **Interpretation**:  
    - < 1.5 t/ha: Low yield (drought/stress likely)  
    - 1.5–2.5 t/ha: Average (typical smallholder)  
    - > 2.5 t/ha: High yield (favorable conditions)
    """)
    
    st.markdown("""
    > 🌍 Part of **SDG 2: Zero Hunger**  
    > ✅ Open-source | 📱 Low-bandwidth friendly | 🔍 Explainable AI
    """)

st.markdown("---")
st.caption("Data: FAO, CHIRPS, SoilGrids | Model: Random Forest | © 2025 AgriPredict for PLP Academy")