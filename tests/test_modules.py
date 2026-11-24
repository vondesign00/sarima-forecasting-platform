"""
Uji modul. Jalankan dari folder proyek:  python -m pytest -q
Tes yang membutuhkan statsmodels otomatis dilewati bila statsmodels belum terpasang.
"""
import io
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR))

from modules import data_loader as dl  # noqa: E402
from modules import data_preprocessing as dp  # noqa: E402
from modules import evaluation as ev  # noqa: E402
from modules import exploration as ex  # noqa: E402
from modules import forecasting as fc  # noqa: E402
from modules.arima_model import SpesifikasiModel, nama_parameter, peringatan_model  # noqa: E402

CONTOH_XLSX = AKAR / "sample_data" / "sample_dataset.xlsx"
CONTOH_CSV = AKAR / "sample_data" / "sample_dataset.csv"


# ------------------------------------------------------------------ 1-2. file valid
@pytest.mark.parametrize("path", [CONTOH_XLSX, CONTOH_CSV])
def test_file_contoh_terbaca(path):
    df = dl.baca_file(str(path))
    assert list(df.columns) == ["tanggal", "nilai", "dummy_libur", "indeks_harga"]
    tgl, target = dl.tebak_kolom(df)
    assert (tgl, target) == ("tanggal", "nilai")
    frek = dp.tebak_frekuensi(df[tgl])
    assert frek == "Bulanan"
    v = dp.siapkan_seri(df, tgl, target, frek)
    assert v.ok and len(v.peringatan) == 1 and "masa depan" in v.peringatan[0]
    assert len(v.seri) == len(df) and str(v.seri.index.freq) in ("<MonthEnd>", "M")
    assert dp.kolom_x_otomatis(df, tgl, target) == ["dummy_libur", "indeks_harga"]


def test_csv_titik_koma_dan_format_indonesia():
    teks = "periode;nilai\n2021M01;1.234.567,5\n2021M02;200.000\n2021M03;300000\n"
    f = io.StringIO(teks)
    f.name = "x.csv"
    df = dl.baca_file(f)
    v = dp.siapkan_seri(df, "periode", "nilai", "Bulanan")
    assert v.seri.tolist() == [1234567.5, 200000.0, 300000.0]


def test_format_file_salah():
    with pytest.raises(dl.GalatData):
        dl.baca_file("data.txt")


# ------------------------------------------------------------------ 3. nilai kosong
def _df(tanggal, nilai):
    return pd.DataFrame({"t": tanggal, "v": nilai})


def test_periode_hilang_dilaporkan_tidak_dihapus():
    per = [f"2021M{m:02d}" for m in range(1, 13) if m != 5]
    v = dp.siapkan_seri(_df(per, np.arange(1.0, 12.0)), "t", "v", "Bulanan")
    assert v.ok
    assert len(v.seri) == 12 and np.isnan(v.seri.loc["2021-05"])
    assert any("tidak ada di data" in p for p in v.peringatan)


def test_nilai_kosong_dilaporkan():
    per = [f"2021M{m:02d}" for m in range(1, 13)]
    nilai = [1.0] * 12
    nilai[3] = None
    v = dp.siapkan_seri(_df(per, nilai), "t", "v", "Bulanan")
    assert v.ok and v.seri.isna().sum() == 1
    assert any("kosong" in p for p in v.peringatan)


# ------------------------------------------------------------------ 4. tanggal tidak valid
def test_tanggal_tidak_valid():
    v = dp.siapkan_seri(_df(["abc", "def"] * 6, range(12)), "t", "v", "Bulanan")
    assert not v.ok and "tidak bisa dibaca" in v.galat[0]


def test_periode_ganda_salah_frekuensi():
    harian = pd.date_range("2021-01-01", periods=40, freq="D")
    v = dp.siapkan_seri(_df(harian, range(40)), "t", "v", "Bulanan")
    assert not v.ok and "periode ganda" in v.galat[0]


def test_tanggal_format_indonesia_hari_dulu():
    v = dp.siapkan_seri(_df([f"01/{m:02d}/2021" for m in range(2, 13)], range(1, 12)), "t", "v", "Bulanan")
    assert v.ok, v.galat
    assert dp.label_periode(v.seri.index[0]) == "2021-02" and len(v.seri) == 11


# ------------------------------------------------------------------ 5. target bukan angka
def test_target_bukan_angka():
    per = [f"2021M{m:02d}" for m in range(1, 13)]
    v = dp.siapkan_seri(_df(per, ["tinggi", "rendah"] * 6), "t", "v", "Bulanan")
    assert not v.ok and "bukan angka" in v.galat[0]


def test_kolom_sama():
    v = dp.siapkan_seri(_df(["2021M01"], [1]), "t", "t", "Bulanan")
    assert not v.ok


# ------------------------------------------------------------------ 6. observasi kurang
def test_observasi_terlalu_sedikit():
    per = [f"2021M{m:02d}" for m in range(1, 6)]
    v = dp.siapkan_seri(_df(per, range(1, 6)), "t", "v", "Bulanan")
    assert not v.ok and "terlalu sedikit" in v.galat[0]


def test_minimal_observasi_model_musiman():
    assert SpesifikasiModel(seasonal_order=(1, 0, 0, 12)).minimal_observasi() >= 18
    assert SpesifikasiModel(order=(1, 0, 0), seasonal_order=(0, 0, 0, 12)).minimal_observasi() == 10


# ------------------------------------------------------------------ 7. nol/negatif dengan log
def test_log_menolak_nol_negatif():
    s = pd.Series([1.0, 0.0, -2.0, 4.0], index=pd.period_range("2021-01", periods=4, freq="M"))
    assert dp.cek_transformasi(s, "log")
    assert not dp.cek_transformasi(s, "tanpa")


def test_log_dan_kebalikannya():
    s = pd.Series([100.0, 200.0, 50.0], index=pd.period_range("2021-01", periods=3, freq="M"))
    for mode in dp.TRANSFORMASI:
        balik = dp.ke_skala_asli(dp.ke_skala_model(s, mode), mode)
        harapan = np.exp(s) if mode == "sudah_log" else s
        np.testing.assert_allclose(balik, harapan)


def test_petunjuk_log_ganda():
    s = pd.Series(np.linspace(11.5, 13.5, 24), index=pd.period_range("2021-01", periods=24, freq="M"))
    assert dp.petunjuk_log_ganda(s, "log")
    assert dp.petunjuk_log_ganda(s * 1e5, "log") is None


# ------------------------------------------------------------------ frekuensi & periode otomatis
@pytest.mark.parametrize("tanggal,harapan", [
    (pd.date_range("2020-01-31", periods=24, freq="ME"), "Bulanan"),
    (pd.date_range("2020-03-31", periods=12, freq="QE"), "Triwulanan"),
    (pd.date_range("2024-01-01", periods=30, freq="W-MON"), "Mingguan"),
    (pd.date_range("2024-01-01", periods=30, freq="D"), "Harian"),
    ([f"2021M{m:02d}" for m in range(1, 13)], "Bulanan"),
    (["2020Q1", "2020Q2", "2020Q3"], "Triwulanan"),
    ([2001, 2002, 2003], "Tahunan"),
])
def test_tebak_frekuensi(tanggal, harapan):
    assert dp.tebak_frekuensi(pd.Series(tanggal)) == harapan


def test_periode_otomatis():
    from modules.pipeline import periode_otomatis
    seri = pd.Series(np.arange(60.0), index=pd.period_range("2015-01", periods=60, freq="M"))
    awal, akhir_latih, akhir_uji = periode_otomatis(seri, 12)
    assert (awal, akhir_latih, akhir_uji) == (seri.index[0], seri.index[-13], seri.index[-1])
    assert periode_otomatis(seri, 12, pakai_uji=False)[2] is None
    pendek = seri.iloc[:30]
    assert periode_otomatis(pendek, 12)[1] == pendek.index[-7]  # 20% data


# ------------------------------------------------------------------ frekuensi lain
@pytest.mark.parametrize("frek,tanggal,n", [
    ("Triwulanan", ["2020Q1", "2020Q2", "2020Q3", "2020Q4"] * 1 + ["2021Q1", "2021Q2", "2021Q3", "2021Q4",
                                                                     "2022Q1", "2022Q2", "2022Q3"], 11),
    ("Mingguan", pd.date_range("2024-01-01", periods=30, freq="W-MON"), 30),
    ("Harian", pd.date_range("2024-01-01", periods=30, freq="D"), 30),
])
def test_frekuensi_lain(frek, tanggal, n):
    v = dp.siapkan_seri(_df(tanggal, np.arange(1.0, n + 1)), "t", "v", frek)
    assert v.ok, v.galat
    assert len(v.seri) == n
    assert dp.label_periode(v.seri.index[0])


# ------------------------------------------------------------------ evaluasi
def test_metrik_definisi():
    a, f = [100, 200, 150], [110, 180, 150]
    m = ev.ukuran_akurasi(a, f)
    assert m["MAE"] == pytest.approx(10.0)
    assert m["RMSE"] == pytest.approx(np.sqrt((100 + 400 + 0) / 3))
    assert m["MAPE (%)"] == pytest.approx((10 / 100 + 20 / 200 + 0) / 3 * 100)
    assert m["Theil U2"] == pytest.approx(0.1940285, rel=1e-6)


def test_mape_mengecualikan_nol():
    m = ev.ukuran_akurasi([0, 100], [10, 110])
    assert m["MAPE (%)"] == pytest.approx(10.0)
    assert m["Aktual = 0 (dikecualikan dari MAPE)"] == 1
    assert m["MAE"] == pytest.approx(10.0)


# ------------------------------------------------------------------ baseline
def test_baseline_musiman():
    idx = pd.period_range("2010-01", periods=36, freq="M")
    y = pd.Series(np.arange(36.0), index=idx)
    latih, uji = y.iloc[:24], idx[24:]
    np.testing.assert_allclose(fc.baseline_naif_musiman(latih, uji, 12), y.iloc[12:24].to_numpy())
    np.testing.assert_allclose(fc.baseline_rata_musiman(latih, uji, 12), (y.iloc[:12].to_numpy() + y.iloc[12:24].to_numpy()) / 2)


# ------------------------------------------------------------------ peringatan model
def test_peringatan_model_degeneratif():
    params = pd.Series({"C": 10.0, "ar.S.L12": 1.0, "ma.S.L12": -0.999895, "sigma2": 0.4})
    bse = pd.Series({"C": 0.22, "ar.S.L12": 7.39e-5, "ma.S.L12": 9.3e-5, "sigma2": 0.09})
    pesan = peringatan_model(params, bse, np.ones(12), np.ones(12) * 0.99999, 12)
    assert len(pesan) == 4
    params2 = pd.Series({"C": 10.0, "ar.S.L12": 0.45, "sigma2": 0.4})
    assert peringatan_model(params2, params2 * 0 + 0.1, np.full(12, 0.45 ** (1 / 12)), np.array([]), 12) == []


def test_nama_parameter():
    assert [nama_parameter(n) for n in ["ar.S.L12", "ma.L1", "sigma2", "C"]] == ["SAR(12)", "MA(1)", "SIGMASQ", "C"]


# ------------------------------------------------------------------ eksplorasi & PNG
def test_statistik_deskriptif():
    s = pd.Series([1.0, 2.0, None, 4.0], index=pd.period_range("2021-01", periods=4, freq="M"))
    st = ex.statistik_deskriptif(s)
    assert st["Observasi terisi"] == 3 and st["Median"] == 2.0
    assert ex.daftar_kosong(s) == ["2021-03"]


def test_png_dari_hasil_tiruan():
    from modules.visualization import png_hasil
    idx = pd.period_range("2021-01", periods=24, freq="M")
    y = pd.Series(np.arange(24.0) + 10, index=idx)
    uji = idx[18:]
    depan = pd.period_range("2023-01", periods=6, freq="M")
    h = types.SimpleNamespace(
        pengaturan=types.SimpleNamespace(awal_latih=idx[0], akhir_latih=idx[17], akhir_uji=idx[-1],
                                         tingkat_kepercayaan=95, spek=SpesifikasiModel()),
        y_asli=y, akhir_data_proyeksi=idx[-1],
        backtest={"tabel": pd.DataFrame({"aktual": y[uji], "ramalan": y[uji] + 1, "batas_bawah": y[uji] - 2,
                                         "batas_atas": y[uji] + 3}, index=uji)},
        ramalan=pd.DataFrame({"ramalan": np.ones(6) * 30, "batas_bawah": np.ones(6) * 25,
                              "batas_atas": np.ones(6) * 35}, index=depan),
    )
    png = png_hasil(h, "nilai")
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10_000


# ------------------------------------------------------------------ 8-10. butuh statsmodels
@pytest.fixture
def seri_sar12():
    """Proses SAR(12) sintetis: log y = 12,6 + u, u_t = 0,5 u_{t-12} + e."""
    rng = np.random.default_rng(42)
    n = 240
    u = np.zeros(n + 120)
    e = rng.normal(0, 0.3, n + 120)
    for t in range(12, n + 120):
        u[t] = 0.5 * u[t - 12] + e[t]
    idx = pd.period_range("2000-01", periods=n, freq="M")
    return pd.Series(np.exp(12.6 + u[120:]), index=idx, name="nilai")


def _pengaturan(seri, horizon=12, spek=None, uji=True):
    from modules.pipeline import Pengaturan
    return Pengaturan(spek=spek or SpesifikasiModel(), transformasi="log", awal_latih=seri.index[0],
                      akhir_latih=seri.index[-25], akhir_uji=seri.index[-1] if uji else None, horizon=horizon)


def test_estimasi_memulihkan_parameter(seri_sar12):
    pytest.importorskip("statsmodels")
    from modules.pipeline import jalankan
    h = jalankan(seri_sar12, _pengaturan(seri_sar12))
    k = h.koefisien_akhir["Koefisien"]
    assert k["C"] == pytest.approx(12.6, abs=0.1)
    assert k["SAR(12)"] == pytest.approx(0.5, abs=0.15)
    assert h.akurasi is not None and len(h.backtest["tabel"]) == 24


@pytest.mark.parametrize("horizon", [1, 12, 36])
def test_horizon(seri_sar12, horizon):
    pytest.importorskip("statsmodels")
    from modules.pipeline import jalankan
    h = jalankan(seri_sar12, _pengaturan(seri_sar12, horizon, uji=False))
    assert len(h.ramalan) == horizon and h.backtest is None
    assert h.ramalan.index[0] == seri_sar12.index[-1] + 1
    assert (h.ramalan["batas_bawah"] < h.ramalan["ramalan"]).all()
    assert (h.ramalan["ramalan"] < h.ramalan["batas_atas"]).all()


def test_backtest_tanpa_kebocoran(seri_sar12):
    """Mengubah nilai periode uji tidak boleh mengubah ramalan backtest."""
    pytest.importorskip("statsmodels")
    from modules.pipeline import jalankan
    peng = _pengaturan(seri_sar12)
    a = jalankan(seri_sar12, peng).backtest["tabel"]["ramalan"]
    rusak = seri_sar12.copy()
    rusak.iloc[-24:] *= 5
    b = jalankan(rusak, peng).backtest["tabel"]["ramalan"]
    np.testing.assert_allclose(a, b)


def test_likelihood_eksak(seri_sar12):
    """LL = log-likelihood Gaussian eksak (matriks kovarians penuh), tanpa burn-in."""
    pytest.importorskip("statsmodels")
    from scipy.linalg import toeplitz
    from scipy.stats import multivariate_normal
    from statsmodels.tsa.arima_process import arma_acovf
    from modules import arima_model as am
    y = np.log(seri_sar12.iloc[:120])
    m = am.estimasi(y, SpesifikasiModel())
    phi, c, s2 = m.params["ar.S.L12"], m.params["C"], m.params["sigma2"]
    gamma = arma_acovf(np.r_[1, np.zeros(11), -phi], [1], nobs=len(y), sigma2=s2)
    ll = multivariate_normal(mean=np.full(len(y), c), cov=toeplitz(gamma)).logpdf(y.to_numpy())
    assert m.llf == pytest.approx(ll, rel=1e-7)
    assert m.nobs == len(y)
    st = am.statistik_ringkas(m)
    assert st["Akaike info criterion (AIC)"] == pytest.approx((-2 * ll + 2 * 3) / len(y), rel=1e-6)


def _integrasi_manual(m, langkah):
    """Ramalan level dari ramalan model ARMA pada data hasil differencing, diintegrasikan manual."""
    spek = m.spek
    d = spek.order[1]
    _, D, _, s = spek.seasonal_order
    exog = np.ones((langkah, 1)) if spek.konstanta else None
    f = np.asarray(m.res.get_forecast(langkah, exog=exog).predicted_mean)
    y = list(m.y_model.to_numpy())
    # hanya untuk satu jenis differencing per tes
    for h in range(langkah):
        if d == 1:
            y.append(y[-1] + f[h])
        else:
            y.append(y[-s] + f[h])
    return np.array(y[-langkah:])


@pytest.mark.parametrize("order,seasonal", [((1, 1, 0), (0, 0, 0, 12)), ((0, 0, 1), (0, 1, 0, 12))])
def test_differencing_dan_drift(order, seasonal):
    """Dengan differencing, C = drift dan ramalan level = integrasi ramalan data hasil differencing."""
    pytest.importorskip("statsmodels")
    from modules import arima_model as am
    rng = np.random.default_rng(3)
    idx = pd.period_range("2005-01", periods=150, freq="M")
    musim = np.tile(np.sin(np.arange(12) / 12 * 2 * np.pi), 13)[:150]
    y = pd.Series(10 + np.cumsum(0.02 + rng.normal(0, 0.05, 150)) + musim, index=idx)
    spek = SpesifikasiModel(order=order, seasonal_order=seasonal)
    m = am.estimasi(y, spek)
    assert m.nobs == len(y) - spek.observasi_hilang()
    ramal = fc.proyeksi(m, 24, "tanpa")
    np.testing.assert_allclose(ramal["ramalan"].to_numpy(), _integrasi_manual(m, 24), rtol=1e-6)
    assert (ramal["batas_atas"] > ramal["batas_bawah"]).all()
    lebar = (ramal["batas_atas"] - ramal["batas_bawah"]).to_numpy()
    assert lebar[-1] > lebar[0]  # ketidakpastian bertambah dengan horizon


def test_tanpa_musim_tahunan():
    pytest.importorskip("statsmodels")
    from modules.pipeline import Pengaturan, jalankan
    rng = np.random.default_rng(5)
    e = rng.normal(0, 1, 60)
    u = np.zeros(60)
    for t in range(1, 60):
        u[t] = 0.6 * u[t - 1] + e[t]
    df = _df([str(t) for t in range(1965, 2025)], 50 + u)
    assert dp.tebak_frekuensi(df["t"]) == "Tahunan"
    seri = dp.siapkan_seri(df, "t", "v", "Tahunan").seri
    spek = SpesifikasiModel(order=(1, 0, 0), seasonal_order=(0, 0, 0, 1))
    peng = Pengaturan(spek=spek, transformasi="tanpa", awal_latih=seri.index[0], akhir_latih=seri.index[-6],
                      akhir_uji=seri.index[-1], horizon=5)
    h = jalankan(seri, peng)
    assert spek.label() == "ARIMA(1,0,0) + C"
    assert h.koefisien_akhir.loc["AR(1)", "Koefisien"] == pytest.approx(0.6, abs=0.25)
    assert "Naif (nilai terakhir)" in h.akurasi.index and "Naif musiman" not in h.akurasi.index
    assert dp.label_periode(h.ramalan.index[0]) == "2025"


def test_musiman_butuh_s_lebih_dari_1():
    from modules.pipeline import Pengaturan, periksa
    seri = pd.Series(np.arange(1.0, 41.0), index=pd.period_range("1980", periods=40, freq="Y"))
    peng = Pengaturan(spek=SpesifikasiModel(seasonal_order=(1, 0, 0, 1)), transformasi="tanpa",
                      awal_latih=seri.index[0], akhir_latih=seri.index[-6], akhir_uji=seri.index[-1])
    assert any("s ≥ 2" in g for g in periksa(seri, peng))


def test_model_manual_lain_dan_ekspor(seri_sar12):
    pytest.importorskip("statsmodels")
    from modules import export_results as exr
    from modules.pipeline import jalankan, tabel_hasil
    h = jalankan(seri_sar12, _pengaturan(seri_sar12, spek=SpesifikasiModel(order=(1, 0, 1),
                                                                         seasonal_order=(1, 0, 0, 12))))
    assert {"AR(1)", "MA(1)", "SAR(12)", "C", "SIGMASQ"} <= set(h.koefisien_akhir.index)
    assert len(tabel_hasil(h)) == 24 + 12
    xls = pd.read_excel(io.BytesIO(exr.excel_laporan(h)), sheet_name=None)
    assert {"Pengaturan", "Hasil", "Evaluasi", "Koefisien", "Statistik model"} <= set(xls)
    assert exr.csv_hasil(h).startswith("\ufeffPeriode".encode("utf-8"))
    assert b"MODEL UNTUK PROYEKSI" in exr.teks_ringkasan(h)


# ------------------------------------------------------------------ SARIMAX (variabel independen)
def _data_sarimax(n=120, beta=(1.3, 0.8), phi=0.5, c=10.0, seed=11, horizon_x=0):
    """y = C + b1·D1 + b2·X2 + u, u_t = phi·u_{t-12} + e. D1 dummy, X2 kontinu. Bisa diberi baris X masa depan."""
    rng = np.random.default_rng(seed)
    total = n + horizon_x
    idx = pd.period_range("2010-01", periods=total, freq="M")
    d1 = (rng.random(total) < 0.15).astype(float)
    x2 = np.cumsum(rng.normal(0, 0.3, total)) * 0.2 + rng.normal(0, 1, total)
    u = np.zeros(total + 120)
    e = rng.normal(0, 0.4, total + 120)
    for t in range(12, total + 120):
        u[t] = phi * u[t - 12] + e[t]
    y = c + beta[0] * d1 + beta[1] * x2 + u[120:]
    y[n:] = np.nan
    X = pd.DataFrame({"D1": d1, "X2": x2}, index=idx)
    return pd.Series(y, index=idx, name="y"), X


def test_siapkan_x_dan_transformasi():
    df = pd.DataFrame({"tgl": [f"2021M{m:02d}" for m in range(1, 13)],
                       "y": [5.0] * 10 + [None, None],
                       "dummy": [0, 1] * 6, "harga": np.arange(1.0, 13.0), "teks": ["a"] * 12})
    assert dp.kolom_x_otomatis(df, "tgl", "y") == ["dummy", "harga"]
    v = dp.siapkan_seri(df, "tgl", "y", "Bulanan")
    assert v.ok and any("masa depan" in p for p in v.peringatan) and not any("nilai target kosong" in p for p in v.peringatan)
    vx = dp.siapkan_x(df, "tgl", ["dummy", "harga"], "Bulanan", v.seri.index)
    assert vx.ok and len(vx.X) == 12 and vx.X["harga"].iloc[-1] == 12.0
    Xm, g = dp.transformasi_x(vx.X, "log")
    assert list(Xm.columns) == ["dummy", "log(harga)"] and not g  # dummy tidak di-log
    assert list(dp.bentuk_lag(Xm, (0, 1)).columns) == ["dummy", "dummy(-1)", "log(harga)", "log(harga)(-1)"]
    _, g = dp.transformasi_x(pd.DataFrame({"z": [1.0, -1.0, 2.0]}), "log")
    assert g


def test_likelihood_eksak_dengan_x():
    pytest.importorskip("statsmodels")
    from scipy.linalg import toeplitz
    from scipy.stats import multivariate_normal
    from statsmodels.tsa.arima_process import arma_acovf
    from modules import arima_model as am
    y, X = _data_sarimax()
    spek = SpesifikasiModel(exog=("D1", "X2"))
    m = am.estimasi(y, spek, X)
    p = m.params
    assert p["D1"] == pytest.approx(1.3, abs=0.3) and p["X2"] == pytest.approx(0.8, abs=0.15)
    rata = p["C"] + p["D1"] * X["D1"].iloc[:120] + p["X2"] * X["X2"].iloc[:120]
    gamma = arma_acovf(np.r_[1, np.zeros(11), -p["ar.S.L12"]], [1], nobs=120, sigma2=p["sigma2"])
    ll = multivariate_normal(mean=rata.to_numpy(), cov=toeplitz(gamma)).logpdf(y.iloc[:120].to_numpy())
    assert m.llf == pytest.approx(ll, rel=1e-7)
    st = am.statistik_ringkas(m)
    assert st["Sum squared resid"] == pytest.approx(m.nobs * p["sigma2"], rel=1e-4)
    n, k = m.nobs, len(p)
    assert st["Adjusted R-squared"] == pytest.approx(1 - (1 - st["R-squared"]) * (n - 1) / (n - k))
    assert st["S.E. of regression"] == pytest.approx(np.sqrt(st["Sum squared resid"] / (n - k)))
    tk = am.tabel_koefisien(m)
    assert list(tk.index) == ["C", "D1", "X2", "SAR(12)", "SIGMASQ"]
    from scipy import stats as sst
    assert tk.loc["X2", "Prob."] == pytest.approx(2 * sst.t.sf(abs(tk.loc["X2", "t-Statistic"]), n - k))


def test_sarimax_differencing_ramalan_level():
    """d = 1 dengan X: Δy = C + βΔx + u; ramalan level = integrasi manual."""
    pytest.importorskip("statsmodels")
    from modules import arima_model as am
    y, X = _data_sarimax(horizon_x=12)
    y = y.cumsum() / 10
    spek = SpesifikasiModel(order=(1, 1, 0), seasonal_order=(0, 0, 0, 12), exog=("X2",))
    m = am.estimasi(y.iloc[:120], spek, X)
    assert m.nobs == 119
    ramal = fc.proyeksi(m, 12, "tanpa", X_depan=X.iloc[120:])
    dx = X["X2"].diff().iloc[120:].to_numpy()
    f = np.asarray(m.res.get_forecast(12, exog=np.column_stack([np.ones(12), dx])).predicted_mean)
    np.testing.assert_allclose(ramal["ramalan"].to_numpy(), y.iloc[119] + np.cumsum(f), rtol=1e-6)
    with pytest.raises(ValueError):
        fc.proyeksi(m, 12, "tanpa")  # X masa depan wajib


def test_pipeline_sarimax_bandingkan():
    pytest.importorskip("statsmodels")
    from modules.pipeline import (NAMA_EX_ANTE, NAMA_EX_POST_DEPAN, NAMA_TANPA_X, Pengaturan, jalankan,
                                  tabel_perbandingan_model, tabel_perbandingan_ramalan)
    y, X = _data_sarimax(horizon_x=6)
    peng = Pengaturan(spek=SpesifikasiModel(), transformasi="tanpa", awal_latih=y.index[0],
                      akhir_latih=y.index[107], akhir_uji=y.index[119], horizon=6, kolom_x=("D1", "X2"),
                      lag_x=(0,), maks_orde_x=(1, 0, 1, 0))
    h = jalankan(y, peng, X)
    assert h.pakai_x and h.spek.exog == ("D1", "X2")
    baris = list(h.akurasi.index)
    assert any("ex-post" in b for b in baris) and any("ex-ante" in b for b in baris) and any("tanpa X" in b for b in baris)
    assert set(h.ramalan_varian) == {NAMA_EX_POST_DEPAN, NAMA_EX_ANTE, NAMA_TANPA_X}
    assert h.nama_ramalan_utama == NAMA_EX_POST_DEPAN and len(h.ramalan) == 6
    # sampel model tanpa X sama dengan model dengan X
    assert h.hasil_tanpa_x.nobs == h.hasil_akhir.nobs
    assert h.backtest_tanpa_x["hasil"].nobs == h.backtest["hasil"].nobs
    assert len(tabel_perbandingan_ramalan(h)) == 12 + 6
    assert tabel_perbandingan_model(h).shape[1] == 2
    # model dengan X yang benar harus lebih akurat (ex-post) daripada tanpa X
    mae = h.akurasi["MAE"]
    assert mae[[b for b in baris if "ex-post" in b][0]] < mae[[b for b in baris if "tanpa X" in b][0]]


def test_pipeline_sarimax_lag_dan_tanpa_x_manual():
    pytest.importorskip("statsmodels")
    from modules.pipeline import NAMA_EX_ANTE, Pengaturan, jalankan
    y, X = _data_sarimax()
    peng = Pengaturan(spek=SpesifikasiModel(), transformasi="tanpa", awal_latih=y.index[0],
                      akhir_latih=y.index[107], akhir_uji=y.index[119], horizon=4, kolom_x=("X2",),
                      lag_x=(0, 1), maks_orde_x=(1, 0, 0, 0))
    h = jalankan(y, peng, X)
    assert h.spek.exog == ("X2", "X2(-1)")
    assert h.hasil_akhir.nobs == 119  # satu observasi awal hilang karena lag
    assert h.nama_ramalan_utama == NAMA_EX_ANTE  # tidak ada X manual ke depan
    peng2 = Pengaturan(**{**peng.__dict__, "proyeksi_x_sarima": False})
    from modules.pipeline import GalatPengaturan
    with pytest.raises(GalatPengaturan):
        jalankan(y, peng2, X)


def test_pilih_orde_otomatis():
    pytest.importorskip("statsmodels")
    from modules import arima_model as am
    y, X = _data_sarimax(n=96)
    tabel, daftar = am.pilih_orde(y.iloc[:96], 12, X=X, exog=("D1", "X2"), maks_p=2, maks_q=2, maks_P=1, maks_Q=1)
    assert len(tabel) == 36 == len(daftar)
    aic = tabel["AIC"].dropna().to_numpy()
    assert (np.diff(aic) >= 0).all()
    assert daftar[0].exog == ("D1", "X2")
