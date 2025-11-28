import sys, os
sys.path.append(os.path.dirname(__file__))

import prediction
import streamlit as st
import eda
import prediction

# ===========================
# Sidebar Navigation
# ===========================
st.sidebar.title("🍔 Food Delivery Time App")
st.sidebar.markdown("---")

page = st.sidebar.radio("📂 Pilih Halaman:", ("Exploratory Data Analysis", "Predict Delivery Time"))

st.sidebar.markdown("---")
st.sidebar.subheader("Tentang Aplikasi")
st.sidebar.info("""
Aplikasi ini menampilkan **analisis data eksploratif (EDA)**  
dan **prediksi waktu pengantaran** menggunakan **model XGBoost**  
berdasarkan data pengantaran makanan.
""")

st.sidebar.markdown("👨‍💻 Developed by: Group 1 FTDS 032")

# ===========================
# Page Routing
# ===========================
if page == "Exploratory Data Analysis":
    eda.run()
else:
    prediction.run()
