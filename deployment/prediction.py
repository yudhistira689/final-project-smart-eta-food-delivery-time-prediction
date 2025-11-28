import streamlit as st
import pandas as pd
import pickle
import os

@st.cache_resource
def load_model():
    # gunakan path absolut agar selalu benar walau di cloud
    model_path = os.path.join(os.path.dirname(__file__), "best_xgboost_delivery_time_tuned.pkl")

    if not os.path.exists(model_path):
        st.error(f"Model file tidak ditemukan di path: {model_path}")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def run():
    st.title("🚀 Predict Delivery Time")
    st.image(
        "https://miro.medium.com/v2/resize:fit:1358/format:webp/1*SDQ9ly2pzx3UnGSS0L0WHQ.png",
        caption="Source: Medium — Predicting Delivery Time Using Machine Learning"
    )

    st.markdown("""
    Masukkan parameter pengiriman untuk memprediksi **perkiraan waktu pengantaran (menit)**  
    menggunakan model **XGBoost terbaik yang telah dituning**.
    """)

    # Load trained model
    try:
        model = load_model()
    except FileNotFoundError:
        st.error("❌ File model tidak ditemukan! Pastikan file `best_xgboost_delivery_time_tuned.pkl` ada di folder `deployment/`.")
        return

    # --- Input user ---
    st.subheader("📦 Input Kondisi Pengiriman")
    col1, col2 = st.columns(2)
    with col1:
        distance_km = st.number_input("Distance (km)", min_value=0.1, max_value=100.0, value=5.2)
        weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rainy"], index=0)
        traffic_level = st.selectbox("Traffic Level", ["Low", "Medium", "High"], index=1)
    with col2:
        time_of_day = st.selectbox("Time of Day", ["Morning", "Lunch", "Evening", "Night"], index=1)
        vehicle_type = st.selectbox("Vehicle Type", ["Motorbike", "Car", "Bicycle"], index=0)
        preparation_time_min = st.number_input("Preparation Time (min)", min_value=1.0, max_value=120.0, value=15.0)

    courier_experience_yrs = st.number_input("Courier Experience (years)", min_value=0.0, max_value=20.0, value=2.0)

    if st.button("🔮 Predict Delivery Time"):
        new_data = pd.DataFrame({
            "distance_km": [distance_km],
            "weather": [weather],
            "traffic_level": [traffic_level],
            "time_of_day": [time_of_day],
            "vehicle_type": [vehicle_type],
            "preparation_time_min": [preparation_time_min],
            "courier_experience_yrs": [courier_experience_yrs]
        })

        # Feature engineering
        new_data["prep_to_deliv_ratio"] = new_data["preparation_time_min"] / (new_data["distance_km"] + 1)
        new_data["speed_km_per_min"] = new_data["distance_km"] / (new_data["preparation_time_min"] + 1)
        new_data["experience_level"] = pd.cut(
            new_data["courier_experience_yrs"],
            bins=[0, 2, 5, 10, 20],
            labels=["Newbie", "Intermediate", "Experienced", "Veteran"],
            right=False
        )

        try:
            predictions = model.predict(new_data)
            new_data["predicted_delivery_time_min"] = predictions
            st.markdown("### Hasil Prediksi")
            st.dataframe(new_data)
            st.success(f"Estimasi Waktu Pengantaran: **{predictions[0]:.2f} menit**")
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")

if __name__ == "__main__":
    run()
