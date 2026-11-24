"""
Backtest dan proyeksi.

- Pembagian latih/uji kronologis (tidak diacak).
- Backtest: model diestimasi HANYA dengan data latih, lalu diramal dinamis multi-langkah
  sepanjang periode uji (tidak memakai nilai Y aktual periode uji).
- Untuk model dengan X (SARIMAX) ada dua jenis ramalan:
    * ex-post : nilai X periode ramalan dianggap diketahui (X aktual / X diisi manual);
    * ex-ante : nilai X periode ramalan sendiri diproyeksikan dengan SARIMA dari data X sebelumnya.
- Proyeksi ke depan: model diestimasi ulang dengan data s.d. observasi terakhir.
- Ramalan dibuat di level (differencing ditangani model), lalu dikembalikan ke skala asli
  dengan exp() bila model memakai log.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import arima_model as am
from .data_preprocessing import adalah_dummy, bentuk_lag, ke_skala_asli


def proyeksi(model: am.ModelTerestimasi, langkah: int, mode_transformasi: str,
             alpha: float = 0.05, koreksi_bias: bool = False,
             X_depan: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ramalan dinamis sebanyak langkah periode setelah akhir data estimasi, dalam skala asli.

    X_depan: nilai X (skala model, kolom = spek.exog) untuk periode ramalan; wajib untuk SARIMAX.
    """
    spek = model.spek
    y = model.y_model
    index_depan = pd.period_range(y.index[-1] + 1, periods=langkah, freq=y.index.freq)
    if spek.exog:
        if X_depan is None:
            raise ValueError("Model memakai X: nilai X untuk periode ramalan wajib diberikan.")
        X_depan = X_depan.reindex(index_depan)[list(spek.exog)]
        if X_depan.isna().any().any():
            kosong = X_depan.index[X_depan.isna().any(axis=1)]
            raise ValueError(f"Nilai X untuk periode ramalan belum lengkap ({', '.join(str(p) for p in kosong[:6])}).")
    x_c = am.regresor_konstanta(len(y) + langkah, spek)[len(y):]
    exog_depan = am.matriks_exog(index_depan, x_c, X_depan, spek)

    ramalan = model.model_level().get_forecast(steps=langkah, exog=exog_depan)
    rata = np.asarray(ramalan.predicted_mean)
    se = np.asarray(ramalan.se_mean)
    ci = np.asarray(ramalan.conf_int(alpha=alpha))
    pakai_exp = mode_transformasi in ("log", "sudah_log")
    faktor = np.exp(se ** 2 / 2) if (koreksi_bias and pakai_exp) else 1.0
    return pd.DataFrame({
        "ramalan_skala_model": rata,
        "se_skala_model": se,
        "ramalan": ke_skala_asli(rata, mode_transformasi) * faktor,
        "batas_bawah": ke_skala_asli(ci[:, 0], mode_transformasi),
        "batas_atas": ke_skala_asli(ci[:, 1], mode_transformasi),
    }, index=index_depan)


# --------------------------------------------------------------------------- proyeksi variabel X

def ramal_x_sarima(X_model: pd.DataFrame, akhir: pd.Period, langkah: int, s: int,
                   maks_orde: tuple[int, int, int, int] = (1, 1, 1, 1)) -> tuple[pd.DataFrame, dict]:
    """Proyeksikan setiap kolom X (skala model, tanpa lag) dengan SARIMA + C.

    Orde tiap variabel dipilih otomatis (AIC terkecil) dari data X s.d. akhir.
    Mengembalikan (DataFrame ramalan X untuk langkah periode setelah akhir, info orde per kolom).
    """
    index_depan = pd.period_range(akhir + 1, periods=langkah, freq=X_model.index.freq)
    hasil, info = {}, {}
    for k in X_model.columns:
        x = X_model[k].loc[:akhir].dropna()
        if x.nunique() <= 1:  # konstan: ramalan = nilai itu
            hasil[k] = np.full(langkah, x.iloc[-1] if len(x) else np.nan)
            info[k] = "konstan"
            continue
        maks_p, maks_q, maks_P, maks_Q = maks_orde
        tabel, daftar = am.pilih_orde(x, s, maks_p=maks_p, maks_q=maks_q, maks_P=maks_P, maks_Q=maks_Q)
        terbaik = next((sp for sp, ok in zip(daftar, tabel["Konvergen"]) if ok), daftar[0])
        m = am.estimasi(x, terbaik)
        hasil[k] = proyeksi(m, langkah, "tanpa")["ramalan"].to_numpy()
        info[k] = terbaik.label() + (" — dummy diproyeksi SARIMA; nilai kalender sebaiknya diisi manual"
                                     if adalah_dummy(x) else "")
    return pd.DataFrame(hasil, index=index_depan), info


def desain_x_depan(X_model: pd.DataFrame, X_ramalan: pd.DataFrame, lags: tuple[int, ...]) -> pd.DataFrame:
    """Gabungkan X historis + X periode ramalan, lalu bentuk kolom lag untuk periode ramalan."""
    gabung = pd.concat([X_model.loc[:X_ramalan.index[0] - 1], X_ramalan])
    return bentuk_lag(gabung, lags).loc[X_ramalan.index]


# --------------------------------------------------------------------------- baseline

def baseline_naif_musiman(y_model_latih: pd.Series, periode_uji: pd.PeriodIndex, s: int) -> pd.Series:
    """Ramalan = nilai periode yang sama satu musim sebelumnya (berulang bila horizon > s)."""
    nilai = y_model_latih.copy()
    for p in periode_uji:
        nilai.loc[p] = nilai.get(p - s, np.nan)
    return nilai.loc[periode_uji]


def baseline_rata_musiman(y_model_latih: pd.Series, periode_uji: pd.PeriodIndex, s: int) -> pd.Series:
    """Ramalan = rata-rata (di skala model) per posisi musim pada data latih."""
    y = y_model_latih.dropna()
    posisi = pd.Series([p.ordinal % s for p in y.index], index=y.index)
    rata = y.groupby(posisi).mean()
    return pd.Series([rata.get(p.ordinal % s, np.nan) for p in periode_uji], index=periode_uji)


# --------------------------------------------------------------------------- backtest

def _tabel(aktual: pd.Series, ramal: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "aktual": aktual,
        "ramalan": ramal["ramalan"],
        "galat": aktual - ramal["ramalan"],
        "batas_bawah": ramal["batas_bawah"],
        "batas_atas": ramal["batas_atas"],
    })


def backtest(y_model: pd.Series, spek: am.SpesifikasiModel, awal_latih, akhir_latih, akhir_uji,
             mode_transformasi: str, alpha: float = 0.05, koreksi_bias: bool = False,
             X: pd.DataFrame | None = None, X_uji_lain: dict[str, pd.DataFrame] | None = None) -> dict:
    """Estimasi pada data latih, ramal periode uji.

    X          : desain X lengkap (skala model). Untuk SARIMAX, ramalan utama memakai X aktual
                 periode uji (ex-post).
    X_uji_lain : skenario X lain untuk periode uji, mis. {"ex-ante": X hasil proyeksi}.
    """
    s = spek.seasonal_order[3]
    y_latih = y_model.loc[awal_latih:akhir_latih]
    y_uji = y_model.loc[akhir_latih + 1:akhir_uji]
    model = am.estimasi(y_latih, spek, X)
    aktual = ke_skala_asli(y_uji, mode_transformasi)
    X_uji = X.reindex(y_uji.index) if (spek.exog and X is not None) else None
    tabel = _tabel(aktual, proyeksi(model, len(y_uji), mode_transformasi, alpha, koreksi_bias, X_uji))
    varian = {}
    for nama, Xs in (X_uji_lain or {}).items():
        varian[nama] = _tabel(aktual, proyeksi(model, len(y_uji), mode_transformasi, alpha, koreksi_bias, Xs))
    y_latih_efektif = model.y_model
    pembanding = {}
    if s > 1:
        pembanding["Naif musiman"] = ke_skala_asli(baseline_naif_musiman(y_latih_efektif, y_uji.index, s),
                                                   mode_transformasi)
        pembanding["Rata-rata musiman"] = ke_skala_asli(baseline_rata_musiman(y_latih_efektif, y_uji.index, s),
                                                       mode_transformasi)
    pembanding["Naif (nilai terakhir)"] = pd.Series(
        ke_skala_asli(y_latih_efektif.dropna().iloc[-1], mode_transformasi), index=y_uji.index)
    return {"hasil": model, "y_latih": y_latih_efektif, "tabel": tabel, "tabel_varian": varian,
            "pembanding": pembanding}
