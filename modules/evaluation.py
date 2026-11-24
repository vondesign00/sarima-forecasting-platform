"""
Ukuran akurasi ramalan:

MAE   = rata-rata |aktual − ramalan|
RMSE  = akar rata-rata (aktual − ramalan)²
MAPE  = rata-rata |aktual − ramalan| / |aktual| × 100
sMAPE = rata-rata |aktual − ramalan| / ((|aktual| + |ramalan|)/2) × 100
Theil U2 = akar[ Σ((ramalan_t+1 − aktual_t+1)/aktual_t)² / Σ((aktual_t+1 − aktual_t)/aktual_t)² ],
           dihitung di dalam periode uji.

Periode dengan aktual = 0 dikecualikan dari MAPE (pembagian dengan nol) dan jumlahnya
dilaporkan, bukan disembunyikan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PENJELASAN = {
    "MAE": "Rata-rata selisih absolut antara nilai aktual dan ramalan, dalam satuan data. Makin kecil makin baik.",
    "RMSE": "Seperti MAE tetapi memberi bobot lebih besar pada kesalahan besar. Satuan sama dengan data.",
    "MAPE (%)": "Rata-rata kesalahan dalam persen dari nilai aktual. Mudah dibaca, tetapi membengkak bila ada nilai aktual yang sangat kecil.",
    "sMAPE (%)": "Versi simetris MAPE (dibagi rata-rata aktual dan ramalan), lebih stabil saat nilai aktual kecil.",
    "Theil U2": "Perbandingan dengan ramalan naif 'sama dengan periode sebelumnya'. < 1 berarti lebih baik dari naif.",
}


def ukuran_akurasi(aktual, ramalan) -> dict:
    a = np.asarray(aktual, dtype=float)
    f = np.asarray(ramalan, dtype=float)
    sah = ~(np.isnan(a) | np.isnan(f))
    a, f = a[sah], f[sah]
    if len(a) == 0:
        return {"n": 0}
    e = a - f
    bukan_nol = a != 0
    hasil = {
        "n": int(len(a)),
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAPE (%)": float(np.mean(np.abs(e[bukan_nol] / a[bukan_nol])) * 100) if bukan_nol.any() else np.nan,
        "Aktual = 0 (dikecualikan dari MAPE)": int((~bukan_nol).sum()),
    }
    penyebut = (np.abs(a) + np.abs(f)) / 2
    ok = penyebut != 0
    hasil["sMAPE (%)"] = float(np.mean(np.abs(e[ok]) / penyebut[ok]) * 100) if ok.any() else np.nan
    if len(a) > 1:
        dasar = a[:-1]
        ok = dasar != 0
        pemb = np.sum(((f[1:][ok] - a[1:][ok]) / dasar[ok]) ** 2)
        peny = np.sum(((a[1:][ok] - dasar[ok]) / dasar[ok]) ** 2)
        hasil["Theil U2"] = float(np.sqrt(pemb / peny)) if peny > 0 else np.nan
    return hasil


def tabel_akurasi(aktual: pd.Series, ramalan: dict[str, pd.Series]) -> pd.DataFrame:
    urutan = ["MAE", "RMSE", "MAPE (%)", "sMAPE (%)", "Theil U2", "n", "Aktual = 0 (dikecualikan dari MAPE)"]
    tabel = pd.DataFrame({nama: ukuran_akurasi(aktual, f.reindex(aktual.index)) for nama, f in ramalan.items()}).T
    return tabel[[c for c in urutan if c in tabel.columns]]
