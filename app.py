"""
SARIMA / SARIMAX Forecasting Platform — titik masuk aplikasi.
Jalankan dari folder proyek:  python -m streamlit run app.py
"""
import streamlit as st

from halaman import komponen

st.set_page_config(page_title="SARIMA / SARIMAX Forecasting Platform", page_icon="📈", layout="wide")
komponen.gaya()

navigasi = st.navigation({
    "Mulai": [
        st.Page("halaman/beranda.py", title="Beranda", icon=":material/home:", default=True),
    ],
    "Alur kerja": [
        st.Page("halaman/unggah.py", title="1. Unggah Data", icon=":material/upload_file:"),
        st.Page("halaman/eksplorasi.py", title="2. Eksplorasi Data", icon=":material/query_stats:"),
        st.Page("halaman/konfigurasi.py", title="3. Konfigurasi & Jalankan", icon=":material/tune:"),
        st.Page("halaman/hasil.py", title="4. Hasil Proyeksi", icon=":material/show_chart:"),
        st.Page("halaman/evaluasi.py", title="5. Evaluasi Model", icon=":material/fact_check:"),
        st.Page("halaman/unduh.py", title="6. Unduh Hasil", icon=":material/download:"),
    ],
})
komponen.status_sidebar()
navigasi.run()
