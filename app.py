import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi Sampah Rumah Tangga", layout="centered")

st.title("🗑️ Prediksi Jumlah Sampah Rumah Tangga")
st.write("Model prediksi berdasarkan jumlah penghuni dan konsumsi makanan per orang per hari.")

# --- Upload Dataset ---
st.subheader("📄 Upload Dataset (Opsional)")
uploaded_file = st.file_uploader("Upload file CSV dengan kolom: 'Jumlah Penghuni', 'Konsumsi Makanan (kg)', 'Sampah (kg)'", type="csv")

if uploaded_file:
    try:
        df_data = pd.read_csv(uploaded_file)
        st.success("✅ Dataset berhasil dimuat.")
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()
else:
    # Simulasi dataset jika tidak upload
    np.random.seed(42)
    X_data = []
    y_data = []
    for _ in range(100):
        penghuni = np.random.randint(1, 10)
        konsumsi = np.random.uniform(0.5, 3.0)
        sampah = penghuni * konsumsi * np.random.uniform(0.4, 0.6)
        X_data.append([penghuni, konsumsi])
        y_data.append(sampah)

    df_data = pd.DataFrame(X_data, columns=["Jumlah Penghuni", "Konsumsi Makanan (kg)"])
    df_data["Sampah (kg)"] = y_data
    st.info("🔁 Menggunakan dataset bawaan (simulasi)")

# --- Tampilkan Data ---
st.subheader("📊 Data Latih")
st.dataframe(df_data.head(20))

# --- Input Prediksi ---
st.subheader("🧮 Input Prediksi")
jumlah_penghuni = st.number_input("Jumlah Penghuni Rumah", min_value=1, value=3)
konsumsi_makanan = st.number_input("Konsumsi Makanan per Orang (kg/hari)", min_value=0.1, value=1.5)

# --- Latih Model ---
X = df_data[["Jumlah Penghuni", "Konsumsi Makanan (kg)"]].values
y = df_data["Sampah (kg)"].values
model = LinearRegression()
model.fit(X, y)

# --- Tombol Prediksi ---
if st.button("🔍 Prediksi Sekarang"):
    input_data = np.array([[jumlah_penghuni, konsumsi_makanan]])
    prediksi = model.predict(input_data)[0]
    st.success(f"🧾 Perkiraan jumlah sampah harian: **{prediksi:.2f} kg**")

    # --- Visualisasi ---
    st.subheader("📈 Visualisasi (Jumlah Penghuni vs Sampah)")

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], y, alpha=0.6, label="Data Latih")
    x_line = np.linspace(1, 10, 100)
    x_full = np.column_stack((x_line, np.full_like(x_line, konsumsi_makanan)))
    y_line = model.predict(x_full)
    ax.plot(x_line, y_line, color="red", label=f"Regresi (konsumsi={konsumsi_makanan} kg)")
    ax.set_xlabel("Jumlah Penghuni")
    ax.set_ylabel("Sampah (kg/hari)")
    ax.legend()
    st.pyplot(fig)

    # --- Penjelasan Model ---
    st.subheader("ℹ️ Penjelasan Model Prediksi")

    with st.expander("🧠 Apa itu Model Prediksi yang Digunakan?"):
        st.markdown("""
        **Model prediksi** yang digunakan di aplikasi ini adalah **Regresi Linear**.

        Regresi linear memodelkan hubungan linier antara:
        - **Input (fitur)**: Jumlah penghuni & konsumsi makanan
        - **Output (target)**: Sampah harian dalam kg

        Formula umumnya:
        \n
        \[
        \text{Sampah} = b_0 + b_1 \times \text{Jumlah Penghuni} + b_2 \times \text{Konsumsi}
        \]

        Model ini dilatih dari dataset (bawaan atau hasil upload), kemudian digunakan untuk menghitung prediksi berdasarkan input kamu.

        ### Kelebihan:
        - Cepat, sederhana, dan bisa dijelaskan secara logis.
        - Cocok untuk data dengan hubungan linier.

        ### Keterbatasan:
        - Kurang cocok untuk hubungan kompleks.
        - Tidak memperhitungkan variabel lain (misalnya: musim, jenis makanan, daur ulang).

        Jika ingin model lebih akurat, kamu bisa menyediakan dataset nyata atau menambahkan lebih banyak variabel input.
        """)

