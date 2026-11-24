import pandas as pd
import streamlit as st

from halaman import komponen as kp
from modules.arima_model import statistik_ringkas, tabel_koefisien
from modules.data_preprocessing import label_periode
from modules.pipeline import tabel_hasil, tabel_perbandingan_model, tabel_perbandingan_ramalan
from modules.visualization import grafik_acf, grafik_hasil

h = kp.perlu_hasil()
peng = h.pengaturan
target = st.session_state.get(kp.TARGET, "Nilai")

st.title("4. Hasil Proyeksi")
for p in h.diagnostik_akhir["peringatan"]:
    st.warning(p)
for p in h.peringatan:
    st.info(p)

m = st.columns(4)
m[0].metric("Model", h.spek.label())
m[1].metric("Data latih", f"{label_periode(peng.awal_latih)} – {label_periode(peng.akhir_latih)}")
m[2].metric("Periode uji", f"{label_periode(peng.akhir_latih + 1)} – {label_periode(peng.akhir_uji)}"
            if peng.akhir_uji else "Tidak ada")
m[3].metric("Horizon proyeksi", f"{peng.horizon} periode")
if h.pakai_x:
    st.info(f"**Variabel X:** {', '.join(h.spek.exog)}. Garis merah utama = **{h.nama_ramalan_utama}**; garis titik-titik "
            "berwarna lain = varian lain (ex-ante / tanpa X). Klik nama di legenda untuk menyembunyikan/menampilkan garis.")

st.plotly_chart(grafik_hasil(h, target))

if h.pakai_x:
    st.subheader("Perbandingan ramalan: dengan X (ex-post, ex-ante) dan tanpa X")
    st.caption("Periode uji: model dilatih hanya dengan data latih. Ex-post memakai X aktual; ex-ante memakai X hasil "
               "proyeksi SARIMA. Proyeksi ke depan: ex-post memakai X yang Anda isi manual di file (bila ada).")
    tp = tabel_perbandingan_ramalan(h)
    st.dataframe(tp.style.format({c: kp.fmt_angka for c in tp.columns if c != "Jenis"}, na_rep="—"))
    if h.info_x:
        with st.expander("Keterangan variabel X dan proyeksinya"):
            st.dataframe(pd.Series(h.info_x, name="Keterangan").to_frame())

st.subheader("Tabel aktual dan ramalan")
st.caption("Baris 'Uji (backtest)': ramalan dari model yang hanya memakai data latih. "
           "Baris 'Proyeksi ke depan': aktual belum ada. Semua nilai dalam skala asli data.")
tabel = tabel_hasil(h)
st.dataframe(tabel.style.format({c: kp.fmt_angka for c in tabel.columns if c != "Jenis"}, na_rep="—"))

st.subheader("Estimasi model")
nama_tab = ["Model untuk proyeksi", "Model backtest", "Diagnostik residual", "Ringkasan statsmodels"]
if h.pakai_x:
    nama_tab.insert(2, "Dengan vs tanpa X")
tabs = st.tabs(nama_tab)
tab_akhir, tab_bt = tabs[0], tabs[1]
tab_diag, tab_ringkas = tabs[-2], tabs[-1]

with tab_akhir:
    st.caption(f"Diestimasi dengan data {label_periode(peng.awal_latih)} s.d. {label_periode(h.akhir_data_proyeksi)}.")
    st.dataframe(h.koefisien_akhir.style.format("{:.6f}"))
    st.dataframe(h.statistik_akhir.to_frame("Nilai").style.format("{:.6f}"))
    st.caption("Exact maximum likelihood; kriteria informasi per observasi. Bila ada differencing, "
               "statistik dihitung pada data setelah differencing.")

with tab_bt:
    if h.backtest is None:
        st.info("Tidak ada periode uji.")
    else:
        st.caption(f"Diestimasi hanya dengan data latih {label_periode(peng.awal_latih)} s.d. "
                   f"{label_periode(peng.akhir_latih)}; dipakai untuk ramalan periode uji.")
        st.dataframe(tabel_koefisien(h.backtest["hasil"]).style.format("{:.6f}"))
        st.dataframe(statistik_ringkas(h.backtest["hasil"]).to_frame("Nilai").style.format("{:.6f}"))

if h.pakai_x:
    with tabs[2]:
        st.caption("Model tanpa X memakai orde ARMA, konstanta, dan sampel yang sama. AIC/SC lebih kecil = lebih baik; "
                   "Prob. koefisien X < 0,05 berarti X berpengaruh signifikan.")
        if h.backtest is not None:
            st.markdown("**Model backtest (data latih)**")
            st.dataframe(tabel_perbandingan_model(h, pakai_backtest=True).style.format("{:.6f}"))
        st.markdown("**Model untuk proyeksi (seluruh data)**")
        st.dataframe(tabel_perbandingan_model(h, pakai_backtest=False).style.format("{:.6f}"))
        st.markdown(f"**Koefisien model tanpa X** — {h.hasil_tanpa_x.spek.label()}")
        st.dataframe(tabel_koefisien(h.hasil_tanpa_x).style.format("{:.6f}"))

with tab_diag:
    diag = h.diagnostik_akhir
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ljung-Box**: p > 0,05 berarti tidak ada autokorelasi residual yang tersisa.")
        st.dataframe(diag["ljung_box"].rename(columns={"lb_stat": "Statistik Q", "lb_pvalue": "p-value"}))
        jb = diag["jarque_bera"]
        st.markdown(f"**Jarque-Bera**: {jb['statistik']:.3f}, p = {jb['p_value']:.4f} "
                    f"→ residual {'normal' if jb['p_value'] > 0.05 else 'tidak normal'}.")
        ar, ma = diag["akar_ar_invers_maks"], diag["akar_ma_invers_maks"]
        st.markdown(f"**Akar invers maksimum** (harus < 1): AR = {ar:.4f}" if ar is not None else "AR: –")
        if ma is not None:
            st.markdown(f"MA = {ma:.4f}")
    with c2:
        st.plotly_chart(grafik_acf(diag))

with tab_ringkas:
    st.code(str(h.hasil_akhir.summary()), language="text")
