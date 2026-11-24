# SARIMA / SARIMAX Forecasting Platform

Aplikasi web (Streamlit) dan notebook Google Colab untuk proyeksi deret waktu dengan **SARIMA** dan
**SARIMAX**, untuk data harian, mingguan, bulanan, triwulanan, atau tahunan. Semua tampilan berbahasa Indonesia.

## Fitur utama

- Unggah Excel/CSV; kolom tanggal, target, frekuensi, dan variabel X ditebak otomatis (bisa diubah).
- **SARIMA** dan **SARIMAX**: model dengan X selalu dibandingkan dengan model yang sama **tanpa X** pada sampel yang sama.
- Nilai X ke depan: **diisi manual** di file (ramalan *ex-post*) dan/atau **diproyeksikan dengan SARIMA** (*ex-ante*).
- Transformasi log (Y dan X; dummy 0/1 tidak di-log), lag X, differencing (d, D), konstanta/drift.
- **Pemilihan orde otomatis** (p, q)(P, Q) berdasarkan AIC / SC / HQ.
- Backtest tanpa kebocoran data uji, pembanding naif, MAE / RMSE / MAPE / sMAPE / Theil U2.
- Diagnostik residual (Ljung-Box, Jarque-Bera, ACF, akar AR/MA) dan statistik model lengkap
  (R², adj. R², S.E. of regression, F-statistic, Durbin-Watson, AIC/SC/HQ per observasi).
- Unduh laporan Excel, CSV, PNG, dan ringkasan teks.

## Metodologi estimasi

Exact maximum likelihood (state ARMA diinisialisasi stasioner, tanpa burn-in), kovarians OPG, uji-t dengan
derajat bebas n − k, differencing eksplisit pada Y dan X sebelum estimasi, konstanta C sebagai rata-rata
(atau drift bila ada differencing). Detail ada di docstring `modules/arima_model.py`.

## Struktur

```
app.py                  titik masuk Streamlit
halaman/                halaman aplikasi (unggah, eksplorasi, konfigurasi, hasil, evaluasi, unduh)
modules/                mesin perhitungan (tanpa UI) — dipakai bersama oleh aplikasi dan notebook
tests/                  uji otomatis (pytest)
sample_data/            data contoh sintetis (termasuk dummy, variabel X, dan baris X masa depan)
notebooks/              notebook Google Colab (berisi salinan mesin yang sama)
.streamlit/config.toml  tema (navy - biru - putih - abu)
```

## Menjalankan di komputer sendiri

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
python -m pytest -q              # uji otomatis
```

## Deploy ke Streamlit Community Cloud

1. Push repo ini ke GitHub.
2. Di [share.streamlit.io](https://share.streamlit.io): **Create app** → pilih repo, branch `main`, main file `app.py`.
3. Setiap push ke `main` otomatis memperbarui aplikasi. Bila paket di `requirements.txt` berubah dan aplikasi
   error, buka menu aplikasi → **Reboot app**.

> ⚠️ Aplikasi di Community Cloud bisa diakses publik bila repo/aplikasinya publik. Untuk data rahasia, jadikan
> aplikasi privat (Settings → Sharing) atau jalankan di server sendiri.

## Format data

| Kolom | Isi |
|---|---|
| Tanggal | `2021-01-31`, `31/01/2021`, tanggal Excel, `2021M01`, `2021Q1`, atau `2021` |
| Target | Nilai yang diramal (angka). Kosongkan pada baris periode ke depan. |
| Variabel X (opsional) | Angka atau dummy 0/1. Isi juga pada baris periode ke depan bila nilainya sudah diketahui. |
