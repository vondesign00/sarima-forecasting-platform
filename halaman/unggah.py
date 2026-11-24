from pathlib import Path

import pandas as pd
import streamlit as st

from halaman import komponen as kp
from modules.data_loader import GalatData, baca_file, ringkasan_dataset, tebak_kolom
from modules.data_preprocessing import (FREKUENSI, adalah_dummy, kolom_x_otomatis, label_periode, siapkan_seri,
                                        siapkan_x, tebak_frekuensi)

CONTOH = Path(__file__).resolve().parent.parent / "sample_data" / "sample_dataset.xlsx"

st.title("1. Unggah Data")
st.caption("Unggah file Excel (.xlsx) atau CSV (.csv). Satu baris = satu periode. Kolom angka lain selain target "
           "bisa dipakai sebagai variabel independen (X) untuk SARIMAX; baris tanggal ke depan dengan target kosong "
           "dibaca sebagai nilai X masa depan.")

c1, c2 = st.columns([3, 1])
unggahan = c1.file_uploader("Pilih file", type=["xlsx", "xls", "csv"])
c2.write("")
c2.write("")
pakai_contoh = c2.button("Pakai data contoh", help="Data sintetis untuk mencoba aplikasi.")

id_unggahan = getattr(unggahan, "file_id", None) or (unggahan.name if unggahan else None)
if unggahan is not None and id_unggahan != st.session_state.get("id_unggahan"):
    st.session_state["id_unggahan"] = id_unggahan  # hanya dibaca sekali per file baru
    try:
        st.session_state[kp.DF] = baca_file(unggahan)
        st.session_state[kp.NAMA_FILE] = unggahan.name
        kp.reset_data()
    except GalatData as e:
        st.error(str(e))
        st.stop()
elif pakai_contoh:
    st.session_state[kp.DF] = baca_file(CONTOH)
    st.session_state[kp.NAMA_FILE] = "sample_dataset.xlsx (contoh sintetis)"
    kp.reset_data()

df = st.session_state.get(kp.DF)
if df is None:
    st.info("Belum ada file. Unggah file Anda atau klik **Pakai data contoh**.")
    st.stop()

st.success(f"File: **{st.session_state[kp.NAMA_FILE]}**")
ringkas = ringkasan_dataset(df)
m1, m2, m3 = st.columns(3)
m1.metric("Jumlah baris (observasi)", ringkas["jumlah_baris"])
m2.metric("Jumlah kolom (variabel)", ringkas["jumlah_kolom"])
m3.metric("Total sel kosong", int(df.isna().sum().sum()))

t1, t2 = st.tabs(["Pratinjau data", "Informasi kolom"])
t1.dataframe(df.head(50))
t2.dataframe(ringkas["info_kolom"])

st.subheader("Pilih kolom dan frekuensi")
tebak_tgl, tebak_target = tebak_kolom(df)
kolom = list(df.columns)
c1, c2, c3 = st.columns(3)
kolom_tgl = c1.selectbox("Kolom tanggal/periode", kolom, index=kolom.index(tebak_tgl),
                         help="Format yang dikenali: 2021-01-31, 31/01/2021, tanggal Excel, 2021M01, 2021Q1, 2021.")
kolom_target = c2.selectbox("Variabel yang diramal (target)", kolom, index=kolom.index(tebak_target),
                            help="Kolom angka yang ingin diproyeksikan, dalam satuan aslinya.")
frek_tebakan = tebak_frekuensi(df[kolom_tgl])
frekuensi = c3.selectbox("Frekuensi data", list(FREKUENSI), index=list(FREKUENSI).index(frek_tebakan),
                         help="Ditebak otomatis dari kolom tanggal; ubah bila perlu. Menentukan panjang musim bawaan.")

validasi = siapkan_seri(df, kolom_tgl, kolom_target, frekuensi)
for p in validasi.peringatan:
    st.warning(p)
if not validasi.ok:
    for g in validasi.galat:
        st.error(g)
    st.session_state.pop(kp.SERI, None)
    st.stop()

seri = validasi.seri
akhir_y = seri.last_valid_index()

# ---------------------------------------------------------------- variabel X
st.subheader("Variabel independen (X) — opsional")
kandidat_x = [k for k in kolom if k not in (kolom_tgl, kolom_target)]
bawaan_x = [k for k in kolom_x_otomatis(df, kolom_tgl, kolom_target)]
kolom_x = st.multiselect(
    "Kolom X untuk SARIMAX", kandidat_x, default=bawaan_x,
    help="Bawaan: semua kolom angka selain tanggal dan target. Kosongkan untuk SARIMA biasa. "
         "Keluarkan kolom yang bukan variabel penjelas (mis. kolom hasil ramalan software lain).")
X_dasar = None
if kolom_x:
    vx = siapkan_x(df, kolom_tgl, kolom_x, frekuensi, seri.index)
    for p in vx.peringatan:
        st.warning(p)
    if not vx.ok:
        for g in vx.galat:
            st.error(g)
        st.stop()
    X_dasar = vx.X
    ringkas_x = pd.DataFrame({
        "Jenis": ["dummy (0/1)" if adalah_dummy(X_dasar[k]) else "kontinu" for k in kolom_x],
        "Terisi s.d. akhir target": [int(X_dasar[k].loc[:akhir_y].notna().sum()) for k in kolom_x],
        "Kosong s.d. akhir target": [int(X_dasar[k].loc[:akhir_y].isna().sum()) for k in kolom_x],
        "Nilai manual ke depan": [int(X_dasar[k].loc[akhir_y + 1:].notna().sum()) for k in kolom_x],
    }, index=pd.Index(kolom_x, name="Variabel X"))
    st.dataframe(ringkas_x)
    if ringkas_x["Kosong s.d. akhir target"].any():
        st.warning("Ada nilai X yang kosong di periode historis. Model membutuhkan X lengkap di periode estimasi; "
                   "lengkapi data atau keluarkan kolom tersebut.")
    if not ringkas_x["Nilai manual ke depan"].any():
        st.info("Belum ada nilai X untuk periode ke depan. Proyeksi akan memakai X hasil proyeksi SARIMA (ex-ante); "
                "untuk ex-post, tambahkan baris tanggal ke depan berisi nilai X (target dikosongkan).")
else:
    st.caption("Tanpa X: model SARIMA biasa.")

lama = st.session_state.get(kp.SERI)
lama_x = st.session_state.get(kp.X_DASAR)
if (lama is None or not lama.equals(seri) or st.session_state.get(kp.TARGET) != kolom_target
        or st.session_state.get(kp.KOLOM_X) != kolom_x
        or (lama_x is None) != (X_dasar is None) or (X_dasar is not None and not X_dasar.equals(lama_x))):
    kp.reset_hasil()
    st.session_state.pop(kp.TABEL_ORDE, None)
st.session_state[kp.SERI] = seri
st.session_state[kp.FREKUENSI] = frekuensi
st.session_state[kp.TARGET] = kolom_target
st.session_state[kp.KOLOM_X] = kolom_x
st.session_state[kp.X_DASAR] = X_dasar

terisi = seri.loc[:akhir_y]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rentang data target", f"{label_periode(terisi.index.min())} – {label_periode(akhir_y)}")
m2.metric("Jumlah periode", len(terisi))
m3.metric("Periode kosong", int(terisi.isna().sum()))
m4.metric("Variabel X", len(kolom_x))
st.success("Data siap. Lanjutkan ke eksplorasi atau langsung ke konfigurasi model.")
c1, c2 = st.columns(2)
if c1.button("Eksplorasi data", icon=":material/query_stats:"):
    st.switch_page("halaman/eksplorasi.py")
if c2.button("Konfigurasi model", type="primary", icon=":material/tune:"):
    st.switch_page("halaman/konfigurasi.py")
