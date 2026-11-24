"""Membaca file CSV/Excel dan meringkas isinya. Tidak mengubah data."""
from __future__ import annotations

import pandas as pd


class GalatData(Exception):
    """Galat yang pesannya aman ditampilkan langsung ke pengguna."""


def baca_file(file) -> pd.DataFrame:
    """Baca CSV atau Excel. `file` boleh path atau objek unggahan Streamlit."""
    nama = getattr(file, "name", str(file)).lower()
    try:
        if nama.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        elif nama.endswith(".csv"):
            df = pd.read_csv(file, sep=None, engine="python")  # deteksi koma / titik koma
        else:
            raise GalatData("Format file tidak didukung. Gunakan .csv atau .xlsx.")
    except GalatData:
        raise
    except Exception as e:
        raise GalatData(f"File tidak bisa dibaca ({e}). Pastikan file tidak rusak dan tidak sedang dikunci.") from e

    if df.empty or df.shape[1] < 2:
        raise GalatData("File harus berisi minimal dua kolom (tanggal dan nilai) dan tidak boleh kosong.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def ringkasan_dataset(df: pd.DataFrame) -> dict:
    """Ringkasan untuk halaman unggah: jumlah baris/kolom, tipe data, nilai kosong."""
    info = pd.DataFrame({
        "Tipe data": df.dtypes.astype(str),
        "Nilai kosong": df.isna().sum(),
        "Contoh nilai": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })
    info["Contoh nilai"] = info["Contoh nilai"].astype(str)
    return {"jumlah_baris": len(df), "jumlah_kolom": df.shape[1], "info_kolom": info}


def tebak_kolom(df: pd.DataFrame) -> tuple[str, str]:
    """Tebakan awal kolom tanggal dan kolom target (pengguna tetap bisa mengganti)."""
    kolom = list(df.columns)
    kata_tanggal = ("tanggal", "periode", "date", "bulan", "waktu", "time", "tahun")
    tgl = next((c for c in kolom if any(k in c.lower() for k in kata_tanggal)), None)
    if tgl is None:
        tgl = next((c for c in kolom if pd.api.types.is_datetime64_any_dtype(df[c])), kolom[0])
    angka = [c for c in kolom if c != tgl and pd.api.types.is_numeric_dtype(df[c])]
    target = angka[0] if angka else next(c for c in kolom if c != tgl)
    return tgl, target
