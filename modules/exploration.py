"""Ringkasan eksplorasi data (hanya informasi; tidak memengaruhi model)."""
from __future__ import annotations

import pandas as pd

from .data_preprocessing import label_periode


def statistik_deskriptif(seri: pd.Series) -> pd.Series:
    s = seri.dropna()
    return pd.Series({
        "Jumlah periode": len(seri),
        "Observasi terisi": len(s),
        "Nilai kosong": int(seri.isna().sum()),
        "Rata-rata": s.mean(),
        "Median": s.median(),
        "Simpangan baku": s.std(),
        "Minimum": s.min(),
        "Maksimum": s.max(),
        "Periode awal": label_periode(seri.index.min()),
        "Periode akhir": label_periode(seri.index.max()),
    })


def daftar_kosong(seri: pd.Series) -> list[str]:
    return [label_periode(p) for p in seri.index[seri.isna()]]


def dekomposisi(seri: pd.Series, s: int, model: str = "additive"):
    """Dekomposisi klasik (moving average). Butuh data tanpa kosong dan ≥ 2 musim penuh."""
    from statsmodels.tsa.seasonal import seasonal_decompose

    if seri.isna().any():
        raise ValueError("Dekomposisi tidak bisa dijalankan karena ada nilai kosong.")
    if len(seri) < 2 * s:
        raise ValueError(f"Dekomposisi butuh minimal {2 * s} observasi (2 musim penuh).")
    if model == "multiplicative" and (seri <= 0).any():
        raise ValueError("Dekomposisi multiplikatif butuh semua nilai > 0.")
    hasil = seasonal_decompose(seri.to_timestamp(), model=model, period=s)
    return pd.DataFrame({"Data": hasil.observed, "Tren": hasil.trend, "Musiman": hasil.seasonal,
                         "Residual": hasil.resid})
