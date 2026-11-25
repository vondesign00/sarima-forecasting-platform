import streamlit as st

st.title("📈 PACAK: Platform Forecasting Praktis")
st.markdown("##### Platform interaktif untuk proyeksi data deret waktu")

st.markdown(
    """
Platform ini membantu Anda membuat **proyeksi (forecast)** dari data deret waktu, misalnya
arus kas, penjualan, produksi, atau harga, dengan model **ARIMA/SARIMA**, tanpa perlu menulis kode.
Bila ada variabel penjelas (mis. dummy hari raya, harga, indeks), gunakan **SARIMAX** dan bandingkan langsung
hasilnya dengan model tanpa variabel tersebut.
"""
)

with st.expander("Apa itu ARIMA?", expanded=False):
    st.markdown(
        """
**ARIMA** (*AutoRegressive Integrated Moving Average*) adalah model statistik yang meramal nilai
ke depan berdasarkan pola data itu sendiri:

- **AR (p)**: nilai sekarang dipengaruhi nilai periode sebelumnya.
- **I (d)**: berapa kali data dibedakan (*differencing*) agar stabil (stasioner).
- **MA (q)**: nilai sekarang dipengaruhi kesalahan ramalan periode sebelumnya.

Bila ada pola yang berulang tiap tahun (musiman), dipakai **SARIMA** dengan komponen musiman
(P, D, Q) dan panjang musim *s* (mis. 12 untuk data bulanan). Bila ditambah variabel independen (X),
modelnya disebut **SARIMAX**: X ikut menjelaskan naik-turunnya data, dan nilai X ke depan bisa diisi manual
(*ex-post*) atau diproyeksikan dengan SARIMA (*ex-ante*).
"""
    )

st.markdown("### Alur kerja")
langkah = [
    ("1", "Unggah data", "File Excel/CSV: tanggal, nilai, dan (opsional) variabel X."),
    ("2", "Eksplorasi", "Lihat grafik, statistik, dan nilai kosong."),
    ("3", "Konfigurasi", "Transformasi, periode, X, orde (bisa otomatis)."),
    ("4", "Hasil", "Koefisien model, grafik, dan tabel ramalan."),
    ("5", "Evaluasi", "Akurasi periode uji; dengan vs tanpa X."),
    ("6", "Unduh", "Excel, CSV, PNG, dan ringkasan teks."),
]
kolom = st.columns(len(langkah))
for k, (no, judul, isi) in zip(kolom, langkah):
    k.markdown(f"<div class='langkah'><b>{no}. {judul}</b><br><small>{isi}</small></div>", unsafe_allow_html=True)

st.markdown("### Cara memakai")
st.markdown(
    """
1. Buka **Unggah Data**, unggah file Anda (atau pakai data contoh), lalu pilih kolom tanggal, kolom yang ingin diramal, dan frekuensinya.
2. Periksa data di **Eksplorasi Data**.
3. Di **Konfigurasi & Jalankan**, atur model lalu klik **Jalankan Proyeksi**.
4. Lihat hasilnya di **Hasil Proyeksi** dan **Evaluasi Model**, lalu unduh di **Unduh Hasil**.

Data yang Anda unggah hanya diproses selama sesi berjalan dan tidak disimpan oleh aplikasi.
"""
)

if st.button("Mulai proyek baru", type="primary", icon=":material/arrow_forward:"):
    st.switch_page("halaman/unggah.py")
