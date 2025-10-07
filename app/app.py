import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "random_forest_model.pkl")

model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")
st.title("🏠 House Price Prediction App")
st.write("Use the sidebar to input house details and predict the price.")

st.sidebar.header("📋 Input House Details")

bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=8, value=2)
living_area = st.sidebar.number_input("Living Area (sq ft)", min_value=500, max_value=10000, value=2000)
lot_area = st.sidebar.number_input("Lot Area (sq ft)", min_value=500, max_value=20000, value=5000)
floors = st.sidebar.number_input("Number of Floors", min_value=1, max_value=4, value=1)
waterfront = st.sidebar.selectbox("Waterfront Present?", options=["No", "Yes"])
views = st.sidebar.slider("Number of Views", 0, 4, 0)
condition = st.sidebar.slider("Condition of House (1–5)", 1, 5, 3)
grade = st.sidebar.slider("Grade of the House (1–13)", 1, 13, 7)
area_no_basement = st.sidebar.number_input("Area of the House (excl. Basement)", min_value=500, max_value=10000, value=2000)
area_basement = st.sidebar.number_input("Area of the Basement", min_value=0, max_value=5000, value=0)
built_year = st.sidebar.number_input("Built Year", min_value=1900, max_value=2025, value=2000)
renovation_year = st.sidebar.number_input("Renovation Year", min_value=1900, max_value=2025, value=2000)
latitude = st.sidebar.number_input("Latitude", value=37.77)
longitude = st.sidebar.number_input("Longitude", value=-122.42)
living_area_renov = st.sidebar.number_input("Living Area after Renovation", min_value=500, max_value=10000, value=2000)
lot_area_renov = st.sidebar.number_input("Lot Area after Renovation", min_value=500, max_value=20000, value=5000)
schools_nearby = st.sidebar.number_input("Number of Schools Nearby", min_value=0, max_value=50, value=5)
distance_airport = st.sidebar.number_input("Distance from Airport (miles)", min_value=0.0, max_value=100.0, value=10.0)

waterfront = 1 if waterfront == "Yes" else 0

if st.sidebar.button("🔮 Predict Price"):
    features = np.array([[
        bedrooms,
        bathrooms,
        living_area,
        lot_area,
        floors,
        waterfront,
        views,
        condition,
        grade,
        area_no_basement,
        area_basement,
        built_year,
        renovation_year,
        latitude,
        longitude,
        living_area_renov,
        lot_area_renov,
        schools_nearby,
        distance_airport
    ]])
    
    prediction = model.predict(features)
    st.success(f"💰 Estimated House Price: ₹{prediction[0]:,.0f}")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Machine Learning")
