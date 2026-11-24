import streamlit as st

from halaman import komponen as kp
from modules.data_preprocessing import label_periode
from modules.evaluation import PENJELASAN
from modules.visualization import grafik_akurasi

h = kp.perlu_hasil()
peng = h.pengaturan

st.title("5. Evaluasi Model")
if h.akurasi is None:
    st.info("Tidak ada periode uji, jadi akurasi ramalan tidak bisa diukur. Aktifkan "
            "'Sisihkan periode uji' di halaman Konfigurasi.")
    st.stop()

st.caption(f"Akurasi **di luar sampel**: model diestimasi dengan data {label_periode(peng.awal_latih)}–"
           f"{label_periode(peng.akhir_latih)}, lalu meramal {label_periode(peng.akhir_latih + 1)}–"
           f"{label_periode(peng.akhir_uji)} tanpa melihat nilai aktualnya.")

utama = h.akurasi.iloc[0]
if h.pakai_x:
    st.markdown(f"Angka di bawah: **{h.akurasi.index[0]}**. Perbandingan lengkap dengan ex-ante dan tanpa X ada "
                "di tabel dan grafik berikutnya.")
m = st.columns(5)
m[0].metric("MAE", kp.fmt_angka(utama["MAE"]))
m[1].metric("RMSE", kp.fmt_angka(utama["RMSE"]))
m[2].metric("MAPE", f"{kp.fmt_angka(utama['MAPE (%)'])}%")
m[3].metric("sMAPE", f"{kp.fmt_angka(utama['sMAPE (%)'])}%")
m[4].metric("Theil U2", kp.fmt_angka(utama.get("Theil U2"), 3))

nol = int(utama.get("Aktual = 0 (dikecualikan dari MAPE)", 0))
if nol:
    st.warning(f"{nol} periode uji bernilai aktual 0 dan dikecualikan dari MAPE (pembagian dengan nol). "
               "MAE, RMSE, dan sMAPE tetap menghitungnya.")

st.subheader("Perbandingan model" if h.pakai_x else "Dibandingkan dengan ramalan sederhana")
if h.pakai_x:
    st.caption("**ex-post (X aktual)**: X periode uji dianggap diketahui. **ex-ante (X diproyeksi SARIMA)**: X periode "
               "uji diproyeksikan dari data latih (kondisi nyata). **tanpa X**: model yang sama tanpa variabel X. "
               "Bila model dengan X tidak lebih akurat dari tanpa X, variabel X belum menambah daya ramal.")
ukuran = st.radio("Ukuran untuk grafik", ["MAPE (%)", "sMAPE (%)", "MAE", "RMSE", "Theil U2"], horizontal=True)
st.plotly_chart(grafik_akurasi(h.akurasi, ukuran))
st.caption("**Naif musiman**: nilai periode yang sama satu musim sebelumnya. **Rata-rata musiman**: rata-rata "
           "tiap posisi musim (mis. tiap bulan) di data latih. **Naif (nilai terakhir)**: nilai terakhir data latih "
           "diulang. Pembanding musiman hanya muncul bila s ≥ 2. Model yang baik seharusnya lebih akurat dari pembanding.")
st.dataframe(h.akurasi.style.format(kp.fmt_angka, subset=[c for c in h.akurasi.columns
                                                           if c not in ("n", "Aktual = 0 (dikecualikan dari MAPE)")]))

st.subheader("Arti setiap ukuran")
for nama, isi in PENJELASAN.items():
    st.markdown(f"- **{nama}**: {isi}")
st.caption("Periode uji yang pendek membuat ukuran-ukuran ini sensitif terhadap satu-dua periode yang meleset jauh. "
           "Lihat juga kolom galat per periode di halaman Hasil.")
