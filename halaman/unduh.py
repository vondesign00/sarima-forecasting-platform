import re

import streamlit as st

from halaman import komponen as kp
from modules import export_results as ex
from modules.visualization import png_hasil

h = kp.perlu_hasil()
target = st.session_state.get(kp.TARGET, "Nilai")
dasar = re.sub(r"[^A-Za-z0-9_-]+", "_", f"proyeksi_{target}").strip("_")

st.title("6. Unduh Hasil")
st.caption(f"Semua file dibuat dari model yang terakhir dijalankan: **{h.spek.label()}**.")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Laporan lengkap (Excel)**  \nPengaturan, hasil, evaluasi, koefisien, statistik model.")
    st.download_button("Unduh laporan .xlsx", ex.excel_laporan(h), f"{dasar}_laporan.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       type="primary", icon=":material/table_view:")
    st.markdown("**Hasil ramalan (Excel)**")
    st.download_button("Unduh hasil .xlsx", ex.excel_hasil_saja(h), f"{dasar}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", icon=":material/grid_on:")
    st.markdown("**Hasil ramalan (CSV)**")
    st.download_button("Unduh hasil .csv", ex.csv_hasil(h), f"{dasar}.csv", "text/csv", icon=":material/description:")
with c2:
    st.markdown("**Grafik (PNG)**")
    png = png_hasil(h, target)
    st.image(png)
    st.download_button("Unduh grafik .png", png, f"{dasar}.png", "image/png", icon=":material/image:")
    st.markdown("**Ringkasan model (TXT)**")
    st.download_button("Unduh ringkasan .txt", ex.teks_ringkasan(h), f"{dasar}_ringkasan.txt", "text/plain",
                       icon=":material/article:")
