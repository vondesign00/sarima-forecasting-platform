import traceback

import streamlit as st

from halaman import komponen as kp
from modules import arima_model as am
from modules.data_preprocessing import (FREKUENSI, TRANSFORMASI, cek_transformasi, ke_skala_model, label_periode,
                                        petunjuk_log_ganda)
from modules.pipeline import GalatPengaturan, Pengaturan, jalankan, panjang_uji_bawaan, siapkan_desain_x

seri = kp.perlu_seri()
X_dasar = st.session_state.get(kp.X_DASAR)
kolom_x = st.session_state.get(kp.KOLOM_X) or []
s_bawaan = FREKUENSI[st.session_state[kp.FREKUENSI]][1]
musiman = s_bawaan > 1
periode = list(seri.dropna().index)
n = len(periode)

# nilai awal widget orde (bisa diisi otomatis oleh tombol "Cari orde terbaik")
for kunci, nilai in {"orde_p": 0 if musiman else 1, "orde_q": 0, "orde_P": 1 if musiman else 0,
                     "orde_Q": 0}.items():
    st.session_state.setdefault(kunci, nilai)

st.title("3. Konfigurasi & Jalankan")

# ---------------------------------------------------------------- transformasi
st.subheader("Transformasi data")
transformasi = st.radio(
    "Bagaimana data diolah sebelum dimodelkan?", list(TRANSFORMASI), format_func=TRANSFORMASI.get,
    help="Log membuat pola musiman yang membesar-mengecil menjadi lebih stabil. Hasil ramalan selalu "
         "dikembalikan ke skala asli (dengan exp) secara otomatis.")
if (petunjuk := petunjuk_log_ganda(seri, transformasi)):
    st.warning(petunjuk)
for g in cek_transformasi(seri, transformasi):
    st.error(g)
if transformasi == "sudah_log":
    st.info("Data dianggap sudah berupa log. Model memakai data apa adanya; hasil ditampilkan dalam skala asli (exp).")

# ---------------------------------------------------------------- periode
st.subheader("Periode latih dan uji")
st.caption("Data dibagi menurut urutan waktu (tidak diacak). Model hanya belajar dari data latih; "
           "data uji dipakai untuk mengukur akurasi ramalan.")
pakai_uji = st.checkbox("Sisihkan periode uji (disarankan)", value=n > max(2 * s_bawaan, 10) + 10)
panjang_uji = panjang_uji_bawaan(n, s_bawaan)
c1, c2, c3 = st.columns(3)
awal_latih = c1.selectbox("Awal data latih", periode, index=0, format_func=label_periode)
opsi_akhir = [p for p in periode if p > awal_latih]
idx_akhir = max(0, len(opsi_akhir) - 1 - (panjang_uji if pakai_uji else 0))
akhir_latih = c2.selectbox("Akhir data latih", opsi_akhir, index=idx_akhir, format_func=label_periode)
akhir_uji = None
if pakai_uji:
    opsi_uji = [p for p in periode if p > akhir_latih]
    if opsi_uji:
        akhir_uji = c3.selectbox("Akhir data uji", opsi_uji, index=len(opsi_uji) - 1, format_func=label_periode)
    else:
        c3.warning("Tidak ada data setelah akhir data latih.")

# ---------------------------------------------------------------- variabel X
transformasi_x, lag_x, proyeksi_x = "tanpa", (0,), True
if kolom_x:
    st.subheader("Variabel independen (X)")
    st.caption(f"Dipilih di halaman Unggah Data: **{', '.join(kolom_x)}**. Model dengan X selalu dibandingkan "
               "dengan model yang sama tanpa X, pada sampel yang sama.")
    c1, c2, c3 = st.columns(3)
    transformasi_x = c1.radio("Transformasi X", ["tanpa", "log"], horizontal=True,
                              help="Dengan 'log', kolom dummy 0/1 tetap tidak di-log.")
    lag_x = tuple(sorted(c2.multiselect("Lag X", list(range(0, max(2, s_bawaan) + 1)), default=[0],
                                        help="0 = X periode yang sama; 1 = X periode sebelumnya, dst.") or [0]))
    proyeksi_x = c3.checkbox("Proyeksikan X dengan SARIMA (ex-ante)", value=True,
                             help="Orde SARIMA tiap X dipilih otomatis (AIC terkecil). Nilai X yang diisi manual "
                                  "di file tetap dipakai untuk ramalan ex-post.")
    with st.expander("Arti ex-post dan ex-ante"):
        st.markdown("- **Ex-post**: ramalan memakai X yang sudah diketahui (X aktual di periode uji, atau X yang "
                    "Anda isi manual untuk periode ke depan).\n"
                    "- **Ex-ante**: ramalan memakai X hasil proyeksi SARIMA, yaitu kondisi nyata saat X belum diketahui.\n"
                    "- **Tanpa X**: model yang sama tanpa variabel X, sebagai pembanding.")

# ---------------------------------------------------------------- model
st.subheader("Spesifikasi model")
with st.expander("Penjelasan parameter"):
    st.markdown(
        """
| Parameter | Arti |
|---|---|
| **p** | Jumlah lag AR: pengaruh nilai periode-periode sebelumnya |
| **d** | Berapa kali data dibedakan (differencing). 0 bila data sudah stasioner |
| **q** | Jumlah lag MA: pengaruh kesalahan ramalan sebelumnya |
| **P, D, Q** | Sama seperti p, d, q tetapi pada lag musiman (mis. 12 bulan lalu) |
| **s** | Panjang satu musim: 12 bulanan, 4 triwulanan, 52 mingguan, 7 harian, 1 = tanpa musim |
| **Konstanta C** | Rata-rata deret; bila ada differencing (d/D > 0), C menjadi drift |
"""
    )
c = st.columns(7)
p = c[0].number_input("p", 0, 5, key="orde_p")
d = c[1].number_input("d", 0, 2, 0)
q = c[2].number_input("q", 0, 5, key="orde_q")
P = c[3].number_input("P", 0, 3, key="orde_P")
D = c[4].number_input("D", 0, 1, 0)
Q = c[5].number_input("Q", 0, 3, key="orde_Q")
s = c[6].number_input("s", 1, 366, s_bawaan)
konstanta = st.checkbox("Sertakan konstanta C (rata-rata)", value=True)
if konstanta and (d or D):
    st.warning("Dengan differencing (d atau D > 0), konstanta berarti tren/drift; X juga ikut di-differencing.")
if s < 2 and (P or D or Q):
    st.error("Untuk data tanpa musim (s = 1), isi P = D = Q = 0.")

with st.expander("Opsi lanjutan"):
    paksa_stasioner = st.checkbox("Paksa model stasioner", value=False,
                                  help="Bawaan: tidak dipaksa (exact maximum likelihood tanpa batasan).")
    paksa_invertible = st.checkbox("Paksa model invertible", value=False)
    koreksi_bias = st.checkbox("Koreksi bias saat mengembalikan dari log", value=False,
                               help="exp(ramalan log) adalah median, bukan rata-rata. Koreksi mengalikan "
                                    "dengan exp(σ²/2). Bawaan: tidak dipakai.")

# ---------------------------------------------------------------- proyeksi
st.subheader("Proyeksi")
c1, c2 = st.columns(2)
horizon = c1.number_input("Horizon (jumlah periode ke depan)", 1, 5 * s_bawaan if musiman else 60,
                          s_bawaan if musiman else 5)
tingkat = c2.select_slider("Tingkat kepercayaan selang ramalan", [80, 90, 95, 99], value=95)

spek = am.SpesifikasiModel(order=(int(p), int(d), int(q)), seasonal_order=(int(P), int(D), int(Q), int(s)),
                           konstanta=konstanta, paksa_stasioner=paksa_stasioner, paksa_invertible=paksa_invertible)
peng = Pengaturan(spek=spek, transformasi=transformasi, awal_latih=awal_latih, akhir_latih=akhir_latih,
                  akhir_uji=akhir_uji, horizon=int(horizon), tingkat_kepercayaan=int(tingkat),
                  koreksi_bias=koreksi_bias, kolom_x=tuple(kolom_x), transformasi_x=transformasi_x, lag_x=lag_x,
                  proyeksi_x_sarima=proyeksi_x)


# ---------------------------------------------------------------- pemilihan orde otomatis
def cari_orde(maks, kriteria):
    """Callback: estimasi semua kombinasi ARMA pada data latih, isi p, q, P, Q dengan yang terbaik."""
    try:
        _, X_desain, galat = siapkan_desain_x(X_dasar, peng)
        if galat:
            st.session_state[kp.TABEL_ORDE] = ("galat", galat)
            return
        y = ke_skala_model(seri, transformasi).loc[awal_latih:akhir_latih]
        tabel, daftar = am.pilih_orde(y, int(s), X=X_desain,
                                      exog=tuple(X_desain.columns) if X_desain is not None else (),
                                      konstanta=konstanta, d=int(d), D=int(D), maks_p=maks[0], maks_q=maks[1],
                                      maks_P=maks[2], maks_Q=maks[3], kriteria=kriteria)
        terbaik = daftar[0]
        st.session_state["orde_p"], st.session_state["orde_q"] = terbaik.order[0], terbaik.order[2]
        st.session_state["orde_P"], st.session_state["orde_Q"] = terbaik.seasonal_order[0], terbaik.seasonal_order[2]
        st.session_state[kp.TABEL_ORDE] = ("ok", tabel, kriteria)
        kp.reset_hasil()
    except Exception as e:  # tampilkan, jangan hentikan halaman
        st.session_state[kp.TABEL_ORDE] = ("galat", [f"Pencarian orde gagal: {e}"])


with st.expander("Pilih orde otomatis (p, q, P, Q)", expanded=kp.TABEL_ORDE in st.session_state):
    st.caption("Mengestimasi semua kombinasi ARMA hingga batas maksimum pada **data latih** (d, D, s, konstanta, dan X "
               "mengikuti pengaturan di atas), lalu mengisi p, q, P, Q dengan model yang kriterianya terkecil.")
    c = st.columns(5)
    maks = (c[0].number_input("Maks p", 0, 4, 2), c[1].number_input("Maks q", 0, 4, 2),
            c[2].number_input("Maks P", 0, 2, 1 if s >= 2 else 0, disabled=s < 2),
            c[3].number_input("Maks Q", 0, 2, 1 if s >= 2 else 0, disabled=s < 2))
    kriteria = c[4].selectbox("Kriteria", ["AIC", "SC", "HQ"])
    jumlah = (maks[0] + 1) * (maks[1] + 1) * ((maks[2] + 1) * (maks[3] + 1) if s >= 2 else 1)
    st.button(f"Cari orde terbaik ({jumlah} model)", icon=":material/search:", on_click=cari_orde,
              args=(tuple(int(v) for v in maks), kriteria))
    hasil_orde = st.session_state.get(kp.TABEL_ORDE)
    if hasil_orde and hasil_orde[0] == "galat":
        for g in hasil_orde[1]:
            st.error(g)
    elif hasil_orde:
        _, tabel, krit = hasil_orde
        st.success(f"Terpilih ARMA {tabel.iloc[0]['ARMA (p,q)(P,Q)']} · {krit} = {tabel.iloc[0][krit]:.6f} · "
                   f"{len(tabel)} model diestimasi, {int((~tabel['Konvergen']).sum())} tidak konvergen, "
                   f"{int(tabel['Batasan stasioner'].sum())} dengan batasan stasioner. Isian p, q, P, Q sudah diperbarui.")
        st.dataframe(tabel.head(10).style.format({"AIC": "{:.6f}", "SC": "{:.6f}", "HQ": "{:.6f}",
                                                  "Log likelihood": "{:.4f}"}))

st.divider()
label_x = f" + X ({', '.join(kolom_x)}; lag {', '.join(map(str, lag_x))})" if kolom_x else ""
st.markdown(f"**Model:** {spek.label()}{label_x} · **Latih:** {label_periode(awal_latih)}–{label_periode(akhir_latih)}"
            + (f" · **Uji:** {label_periode(akhir_latih + 1)}–{label_periode(akhir_uji)}" if akhir_uji else " · tanpa uji")
            + f" · **Horizon:** {horizon}")

if st.button("Jalankan Proyeksi", type="primary", icon=":material/play_arrow:"):
    with st.status("Menjalankan model...", expanded=True) as status:
        try:
            st.write("Memeriksa data dan pengaturan, mengestimasi model"
                     + (", memproyeksikan X, dan membandingkan dengan model tanpa X" if kolom_x else "")
                     + ", membuat ramalan...")
            hasil = jalankan(seri, peng, X_dasar)
            st.session_state[kp.HASIL] = hasil
            status.update(label="Selesai", state="complete")
        except GalatPengaturan as e:
            status.update(label="Pengaturan perlu diperbaiki", state="error")
            for baris in str(e).splitlines():
                st.error(baris)
        except Exception as e:
            status.update(label="Estimasi gagal", state="error")
            st.error("Model tidak berhasil diestimasi dengan pengaturan ini. Coba model yang lebih sederhana, "
                     "periksa nilai kosong, atau aktifkan opsi 'Paksa model stasioner'.")
            with st.expander("Detail teknis (untuk pengembang)"):
                st.code(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")

if st.session_state.get(kp.HASIL) is not None:
    st.success(f"Hasil tersedia untuk {st.session_state[kp.HASIL].spek.label()}.")
    if st.button("Lihat hasil", icon=":material/show_chart:"):
        st.switch_page("halaman/hasil.py")
