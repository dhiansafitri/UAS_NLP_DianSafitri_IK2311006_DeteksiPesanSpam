import streamlit as st
import joblib

# ===============================
# Konfigurasi Halaman
# ===============================
st.set_page_config(
    page_title="Deteksi Spam SMS",
    page_icon="📩",
    layout="centered"
)

# ===============================
# Load Model
# ===============================
try:
    model = joblib.load("spam_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    st.error("Model tidak ditemukan.")
    st.error(e)
    st.stop()

# ===============================
# Judul
# ===============================
st.title("📩 Deteksi Spam SMS Bahasa Indonesia")

st.markdown("""
Aplikasi ini menggunakan:

- **TF-IDF** sebagai metode ekstraksi fitur.
- **Multinomial Naive Bayes** sebagai algoritma klasifikasi.

Masukkan sebuah pesan SMS untuk mengetahui apakah pesan tersebut termasuk **Spam** atau **Bukan Spam**.
""")

st.divider()

# ===============================
# Input
# ===============================
pesan = st.text_area(
    "Masukkan Pesan",
    height=180,
    placeholder="Contoh:\nSelamat! Anda memenangkan hadiah 100 juta rupiah. Klik link berikut..."
)

# ===============================
# Tombol Prediksi
# ===============================
if st.button("🔍 Prediksi", use_container_width=True):

    if pesan.strip() == "":
        st.warning("Silakan masukkan pesan terlebih dahulu.")
    else:

        data = tfidf.transform([pesan])

        hasil = model.predict(data)[0]

        st.subheader("Hasil Prediksi")

        if hasil.lower() == "spam":
            st.error("🚨 Pesan termasuk **SPAM**")
        else:
            st.success("✅ Pesan termasuk **BUKAN SPAM**")

st.divider()

# ===============================
# Contoh Pesan
# ===============================
st.subheader("📝 Contoh Pesan")

col1, col2 = st.columns(2)

with col1:
    st.info("""
**Spam**

Selamat!

Anda memenangkan hadiah Rp100.000.000.

Klik link berikut untuk mengklaim hadiah sekarang.
""")

with col2:
    st.success("""
**Bukan Spam**

Halo Dian,

Besok kuliah dimulai pukul 08.00 pagi.

Jangan lupa membawa laptop.
""")

st.divider()

# ===============================
# Informasi Model
# ===============================
st.subheader("📊 Informasi Model")

st.write("**Algoritma :** Multinomial Naive Bayes")

st.write("**Ekstraksi Fitur :** TF-IDF")

st.write("**Dataset :** SMS Spam Bahasa Indonesia")

st.write("**Akurasi Model :** 98.25%")

st.divider()

st.caption("UAS Natural Language Processing | Universitas Mega Buana Palopo")