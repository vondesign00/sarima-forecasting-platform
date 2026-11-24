"""Komponen tampilan bersama dan pengelolaan session state."""
from __future__ import annotations

import streamlit as st

from modules.data_preprocessing import label_periode

# Kunci session state
DF = "df_mentah"
NAMA_FILE = "nama_file"
SERI = "seri"
FREKUENSI = "frekuensi"
TARGET = "kolom_target"
HASIL = "hasil"
X_DASAR = "x_dasar"
KOLOM_X = "kolom_x"
TABEL_ORDE = "tabel_orde"


NAVY, BIRU, ABU = "#0B1F3A", "#1D4ED8", "#F1F5F9"


def gaya():
    """Tema navy - biru - putih - abu."""
    st.markdown(
        f"""
        <style>
        .block-container {{padding-top: 2rem; max-width: 1300px;}}
        h1, h2, h3 {{color: {NAVY};}}
        h1 {{border-bottom: 3px solid {BIRU}; padding-bottom: .35rem;}}
        section[data-testid="stSidebar"] {{background: {NAVY};}}
        section[data-testid="stSidebar"] * {{color: #E2E8F0 !important;}}
        section[data-testid="stSidebar"] a[aria-current="page"],
        section[data-testid="stSidebar"] a:hover {{background: rgba(29,78,216,.45) !important; border-radius: 8px;}}
        div[data-testid="stMetric"] {{background: {ABU}; border-left: 4px solid {BIRU}; border-radius: 8px;
                                       padding: 10px 14px;}}
        div[data-testid="stMetricValue"] {{font-size: 1.4rem; color: {NAVY};}}
        div[data-testid="stMetricLabel"] p {{color: #475569;}}
        .stTabs [aria-selected="true"] {{color: {BIRU} !important;}}
        .langkah {{background: {ABU}; border-top: 4px solid {BIRU}; border-radius: 10px; padding: 14px 16px;
                   height: 100%;}}
        .langkah b {{color: {NAVY};}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def reset_hasil():
    st.session_state.pop(HASIL, None)


def reset_data():
    for k in (SERI, HASIL, X_DASAR, KOLOM_X, TABEL_ORDE):
        st.session_state.pop(k, None)


def status_sidebar():
    with st.sidebar:
        st.markdown("### Status proyek")
        nama = st.session_state.get(NAMA_FILE)
        seri = st.session_state.get(SERI)
        hasil = st.session_state.get(HASIL)
        st.markdown(f"{'✅' if nama else '⬜'} Data: {nama or 'belum diunggah'}")
        if seri is not None:
            terisi = seri.dropna().index
            st.caption(f"{st.session_state.get(TARGET)} · {st.session_state.get(FREKUENSI)} · "
                       f"{label_periode(terisi.min())} – {label_periode(terisi.max())}")
            kolom_x = st.session_state.get(KOLOM_X) or []
            st.caption(f"Variabel X: {', '.join(kolom_x) if kolom_x else '—'}")
        st.markdown(f"{'✅' if hasil else '⬜'} Model: {hasil.spek.label() if hasil else 'belum dijalankan'}")


def perlu_seri():
    """Hentikan halaman bila data belum siap."""
    if st.session_state.get(SERI) is None:
        st.info("Belum ada data yang siap dipakai. Mulai dari halaman **Unggah Data**.")
        st.page_link("halaman/unggah.py", label="Ke halaman Unggah Data", icon=":material/upload_file:")
        st.stop()
    return st.session_state[SERI]


def perlu_hasil():
    if st.session_state.get(HASIL) is None:
        st.info("Model belum dijalankan. Atur model lalu klik **Jalankan Proyeksi** di halaman Konfigurasi.")
        st.page_link("halaman/konfigurasi.py", label="Ke halaman Konfigurasi", icon=":material/tune:")
        st.stop()
    return st.session_state[HASIL]


def fmt_angka(v, desimal: int = 2) -> str:
    """Format angka gaya Indonesia: 1.234.567,89"""
    try:
        teks = f"{float(v):,.{desimal}f}"
    except (TypeError, ValueError):
        return str(v)
    return teks.replace(",", "_").replace(".", ",").replace("_", ".")
