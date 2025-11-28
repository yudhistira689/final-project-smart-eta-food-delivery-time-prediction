import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ============================================
# Cache resource agar data tidak dibaca berulang
# ============================================
@st.cache_resource
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "Food_Delivery_Times_Clean.csv")
    return pd.read_csv(file_path)

def run():
    st.title("Delivery Time Prediction — Exploratory Data Analysis (EDA)")

    st.image(
        "https://www.apto.digital/au/wp-content/uploads/2023/09/Swiggy-Banner-1.webp",
        caption="Source: apto.digital (Swiggy Delivery Banner)"
    )

    st.markdown("""
    ## 🧭 Project Background  
    Efisiensi pengiriman merupakan faktor kunci dalam **operasional logistik dan kepuasan pelanggan**.
    """)

    # --- Load dataset ---
    df = load_data()
    st.dataframe(df.head())
    st.markdown(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    # ===============================
    # 1️⃣ Distribusi waktu pengantaran
    # ===============================
    st.subheader("1️⃣ Distribusi Waktu Pengantaran (`delivery_time_min`)")
    fig = px.histogram(df, x="delivery_time_min", nbins=30, color_discrete_sequence=["#FF7F50"])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insight:**
    - Rata-rata waktu pengantaran selesai sekitar **56.7 menit**  
    - Terdapat beberapa *outlier* hingga **153 menit**  
    - Distribusi condong ke kanan (*right-skewed*), menunjukkan sebagian kecil pengiriman memakan waktu jauh lebih lama
    """)

    # ===============================
    # 2️⃣ Distribusi jarak & waktu persiapan
    # ===============================
    st.subheader("2️⃣ Distribusi `distance_km` dan `preparation_time_min`")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df, x="distance_km", nbins=25, color_discrete_sequence=["#00BFFF"])
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.histogram(df, x="preparation_time_min", nbins=25, color_discrete_sequence=["#8A2BE2"])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    **Insight:**
    - Jarak pengiriman relatif merata antara **0–20 km**  
    - Waktu persiapan bervariasi antara **5–30 menit** dan distribusinya cukup seragam  
    """)

    # ===============================
    # 3️⃣ Pengaruh cuaca
    # ===============================
    st.subheader("3️⃣ Pengaruh Cuaca terhadap Waktu Pengantaran")
    fig = px.box(df, x="weather", y="delivery_time_min", color="weather")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insight:**
    - Cuaca **Snowy** dan **Rainy** meningkatkan rata-rata waktu antar  
    - Snowy: ~**67 menit**, Rainy: ~**59.8 menit**  
    - Cuaca **Clear** memiliki pengantaran tercepat: ~**53 menit**  
    """)

    # ===============================
    # 4️⃣ Pengaruh tingkat lalu lintas
    # ===============================
    st.subheader("4️⃣ Pengaruh Tingkat Lalu Lintas (`traffic_level`)")
    fig = px.box(df, x="traffic_level", y="delivery_time_min", color="traffic_level")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insight:**
    - Low traffic: ~**52.9 menit**  
    - Medium traffic: ~**56.4 menit**  
    - High traffic: ~**64.9 menit**  
    - Traffic tinggi menyebabkan waktu pengantaran lebih lama dan lebih bervariasi  
    """)

    # ===============================
    # 5️⃣ Perbedaan waktu dalam sehari
    # ===============================
    st.subheader("5️⃣ Waktu Pengantaran berdasarkan `time_of_day`")
    avg_tod = df.groupby("time_of_day")["delivery_time_min"].mean().reset_index()
    fig = px.bar(avg_tod, x="time_of_day", y="delivery_time_min", color="time_of_day")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insight:**
    - Seluruh slot waktu (*morning, afternoon, evening, night*) memiliki perbedaan kecil  
    - Tidak ada tren mencolok antar waktu dalam sehari  
    """)

    # ===============================
    # 6️⃣ Hubungan pengalaman kurir
    # ===============================
    st.subheader("6️⃣ Pengalaman Kurir terhadap Efisiensi Pengiriman")
    fig = px.scatter(df, x="courier_experience_yrs", y="delivery_time_min", trendline="ols",
                     color_discrete_sequence=["#32CD32"])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insight:**
    - Hampir tidak ada korelasi antara pengalaman dan waktu antar  
    - Regresi linear mendatar (slope ≈ 0)  
    - Durasi pengantaran relatif sama di semua tingkat pengalaman  
    """)

    # ===============================
    # 7️⃣ Korelasi antar fitur numerik
    # ===============================
    st.subheader("7️⃣ Korelasi Antar Fitur Numerik")
    num_cols = ["distance_km", "preparation_time_min", "courier_experience_yrs", "delivery_time_min"]
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insight Utama:**
    - `delivery_time_min` sangat berkorelasi dengan:  
      - `distance_km` (+0.78)  
      - `preparation_time_min` (+0.31)  
    - Korelasi dengan `courier_experience_yrs` lemah  
    """)

    # ===============================
    # 8️⃣ Pengaruh Jenis Kendaraan
    # ===============================
    st.subheader("8️⃣ Pengaruh Jenis Kendaraan (`vehicle_type`) terhadap Waktu Pengantaran")
    fig = px.box(df, x="vehicle_type", y="delivery_time_min", color="vehicle_type", 
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insight:**
    - **Scooter** memiliki waktu pengantaran paling cepat (median ~54–55 menit) dibanding **Bike** dan **Car**.  
    - **Car** cenderung sedikit lebih lambat (median ~56 menit), kemungkinan karena faktor lalu lintas.  
    - **Bike** dan **Scooter** memiliki sebaran yang mirip, dengan Scooter sedikit lebih konsisten.  
    - **Perbedaan rata-rata antar kendaraan tidak terlalu signifikan**, sehingga faktor lain seperti **jarak, cuaca, dan lalu lintas** mungkin lebih berpengaruh.  
    """)

    # ===============================
    # Ringkasan
    # ===============================
    st.header("📋 Kesimpulan")
    st.markdown("""
    - Variabel target (`delivery_time_min`) **skewed dan mengandung outlier**  
    - Fitur paling berpengaruh: **distance_km**, **preparation_time_min**, **traffic_level**, dan **weather**  
    - Fitur kurang relevan: **courier_experience_yrs**, **time_of_day** (dampaknya kecil)  
    - **Jenis kendaraan** menunjukkan sedikit perbedaan waktu, namun tidak signifikan  
    - EDA menunjukkan pentingnya **kondisi eksternal (cuaca & lalu lintas)** dalam durasi pengiriman  
    """)

if __name__ == "__main__":
    run()



