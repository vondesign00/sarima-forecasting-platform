"""
Membentuk deret waktu dari tabel, memeriksa masalah data, dan transformasi log.

Aturan: data tidak pernah dihapus atau diubah diam-diam. Setiap periode yang
hilang, nilai kosong, atau nilai yang tidak bisa dibaca dilaporkan ke pengguna.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# nama tampilan -> (kode periode pandas, panjang musim bawaan)
FREKUENSI = {
    "Bulanan": ("M", 12),
    "Triwulanan": ("Q", 4),
    "Tahunan": ("Y", 1),
    "Mingguan": ("W", 52),
    "Harian": ("D", 7),
}

TRANSFORMASI = {
    "log": "Terapkan log natural (data diunggah dalam nilai asli)",
    "sudah_log": "Data sudah dalam bentuk log (jangan di-log lagi)",
    "tanpa": "Tanpa transformasi",
}

_POLA_BULANAN = re.compile(r"^\s*(\d{4})\s*[Mm]\s*(\d{1,2})\s*$")   # 2021M01
_POLA_TRIWULAN = re.compile(r"^\s*(\d{4})\s*[Qq]\s*([1-4])\s*$")   # 2021Q1
_POLA_TAHUN = re.compile(r"^\s*(\d{4})(?:\.0+)?\s*$")                  # 2021


def ke_periode(kolom: pd.Series, kode_frek: str) -> pd.PeriodIndex:
    """Ubah kolom tanggal ke PeriodIndex. Juga mengenali format 2021M01 dan 2021Q1."""
    teks = kolom.astype(str).str.strip()
    if kode_frek == "M" and teks.str.match(_POLA_BULANAN).all():
        b = teks.str.extract(_POLA_BULANAN)
        return pd.PeriodIndex(b[0] + "-" + b[1].str.zfill(2), freq="M")
    if kode_frek == "Q" and teks.str.match(_POLA_TRIWULAN).all():
        b = teks.str.extract(_POLA_TRIWULAN)
        return pd.PeriodIndex(b[0] + "Q" + b[1], freq="Q")
    if teks.str.match(_POLA_TAHUN).all():
        tahun = teks.str.extract(_POLA_TAHUN)[0]
        if kode_frek == "Y":
            return pd.PeriodIndex(tahun, freq="Y")
        raise ValueError("Kolom tanggal hanya berisi tahun; pilih frekuensi Tahunan.")
    if pd.api.types.is_datetime64_any_dtype(kolom):
        tanggal = pd.DatetimeIndex(kolom)
    else:
        # Teks tanggal: format Indonesia hari/bulan/tahun (01/02/2021 = 1 Februari). ISO (2021-02-01) tetap benar.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            tanggal = pd.DatetimeIndex(pd.to_datetime(kolom, errors="raise", dayfirst=True))
    return tanggal.to_period(kode_frek)


def tebak_frekuensi(kolom: pd.Series) -> str:
    """Tebak frekuensi data dari kolom tanggal (jarak median antar-tanggal). Bawaan: Bulanan."""
    teks = kolom.dropna().astype(str).str.strip()
    if len(teks) and teks.str.match(_POLA_BULANAN).all():
        return "Bulanan"
    if len(teks) and teks.str.match(_POLA_TRIWULAN).all():
        return "Triwulanan"
    if len(teks) and teks.str.match(_POLA_TAHUN).all():
        return "Tahunan"
    try:
        if pd.api.types.is_datetime64_any_dtype(kolom):
            tanggal = pd.DatetimeIndex(kolom.dropna())
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                tanggal = pd.DatetimeIndex(pd.to_datetime(kolom.dropna(), errors="raise", dayfirst=True))
    except Exception:
        return "Bulanan"
    jarak = pd.Series(tanggal.sort_values()).diff().dt.days.dropna()
    if jarak.empty:
        return "Bulanan"
    hari = float(jarak.median())
    if hari <= 1.5:
        return "Harian"
    if hari <= 8:
        return "Mingguan"
    if hari <= 45:
        return "Bulanan"
    if hari <= 135:
        return "Triwulanan"
    return "Tahunan"


def label_periode(p: pd.Period) -> str:
    """Label ringkas untuk tampilan: 2021-01, 2021Q1, 2021, atau tanggal awal minggu/hari."""
    kode = p.freqstr[0]
    if kode in ("Y", "A"):
        return str(p.year)
    if kode == "M":
        return p.strftime("%Y-%m")
    if kode == "Q":
        return f"{p.year}Q{p.quarter}"
    return p.start_time.strftime("%Y-%m-%d")


def ke_angka(kolom: pd.Series) -> pd.Series:
    """Ubah teks ke angka. Mengenali format Indonesia (1.234.567,89)."""
    if pd.api.types.is_numeric_dtype(kolom):
        return kolom.astype(float)
    teks = kolom.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
    format_id = teks.str.contains(r",\d+$", na=False) | teks.str.contains(r"\.\d{3}(?:\.|$)", na=False)
    teks = teks.where(~format_id, teks.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(teks, errors="coerce")


@dataclass
class HasilValidasi:
    seri: pd.Series
    galat: list[str] = field(default_factory=list)
    peringatan: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.galat


def siapkan_seri(df: pd.DataFrame, kolom_tanggal: str, kolom_target: str,
                 frekuensi: str = "Bulanan") -> HasilValidasi:
    """Bentuk deret waktu reguler dan laporkan semua masalah data."""
    kode, _ = FREKUENSI[frekuensi]
    galat, peringatan = [], []
    if kolom_tanggal == kolom_target:
        return HasilValidasi(pd.Series(dtype=float), ["Kolom tanggal dan kolom target tidak boleh sama."])

    tanggal_kosong = df[kolom_tanggal].isna()
    if tanggal_kosong.any():
        peringatan.append(f"{int(tanggal_kosong.sum())} baris tanpa tanggal tidak diikutkan.")
    df = df.loc[~tanggal_kosong]

    try:
        periode = ke_periode(df[kolom_tanggal], kode)
    except Exception:
        contoh = ", ".join(df[kolom_tanggal].astype(str).head(3))
        return HasilValidasi(pd.Series(dtype=float), [
            f"Kolom tanggal '{kolom_tanggal}' tidak bisa dibaca sebagai tanggal (contoh isi: {contoh}). "
            "Gunakan format seperti 2021-01-31, 31/01/2021, 2021M01, 2021Q1, atau 2021 (tahunan)."])

    asli = df[kolom_target]
    nilai = ke_angka(asli)
    tak_terbaca = nilai.isna() & asli.notna()
    if len(nilai) and tak_terbaca.mean() > 0.5:
        return HasilValidasi(pd.Series(dtype=float), [
            f"Kolom '{kolom_target}' sebagian besar bukan angka. Pilih kolom target yang berisi nilai numerik."])
    if tak_terbaca.any():
        peringatan.append(f"{int(tak_terbaca.sum())} nilai di kolom target bukan angka dan dianggap kosong.")

    seri = pd.Series(nilai.to_numpy(), index=periode, name=kolom_target).sort_index()

    if seri.index.duplicated().any():
        dup = sorted({label_periode(p) for p in seri.index[seri.index.duplicated()]})
        galat.append(
            f"Ada {len(dup)} periode ganda untuk frekuensi {frekuensi.lower()} (misalnya {', '.join(dup[:5])}). "
            "Periksa apakah frekuensi yang dipilih sudah benar, atau agregasikan data terlebih dahulu.")
        return HasilValidasi(seri, galat, peringatan)

    lengkap = pd.period_range(seri.index.min(), seri.index.max(), freq=seri.index.freq)
    hilang = lengkap.difference(seri.index)
    if len(hilang):
        peringatan.append(
            f"{len(hilang)} periode tidak ada di data ({', '.join(label_periode(p) for p in hilang[:6])}"
            f"{' ...' if len(hilang) > 6 else ''}). Periode ini dibiarkan kosong; model tidak mengisinya dengan nilai karangan.")
    seri = seri.reindex(lengkap)

    terakhir = seri.last_valid_index()
    masa_depan = seri.index > terakhir if terakhir is not None else np.zeros(len(seri), bool)
    if masa_depan.any():
        peringatan.append(f"{int(masa_depan.sum())} periode di akhir data tidak punya nilai target "
                          f"({label_periode(seri.index[masa_depan][0])} s.d. {label_periode(seri.index[-1])}); "
                          "dibaca sebagai periode masa depan (mis. berisi nilai X untuk proyeksi).")
    kosong = seri.isna() & ~seri.index.isin(hilang) & ~masa_depan
    if kosong.any() and not tak_terbaca.any():
        peringatan.append(f"{int(kosong.sum())} nilai target kosong.")

    if seri.notna().sum() < 10:
        galat.append("Data terlalu sedikit (kurang dari 10 observasi terisi) untuk model ARIMA.")
    return HasilValidasi(seri, galat, peringatan)


# --------------------------------------------------------------------------- transformasi

def cek_transformasi(seri: pd.Series, mode: str) -> list[str]:
    """Galat yang mencegah transformasi dijalankan."""
    if mode == "log":
        buruk = seri.dropna() <= 0
        if buruk.any():
            contoh = ", ".join(label_periode(p) for p in buruk[buruk].index[:6])
            return [f"Transformasi log membutuhkan nilai > 0. Ada {int(buruk.sum())} nilai ≤ 0 (misalnya {contoh}). "
                    "Pilih 'Tanpa transformasi' atau perbaiki datanya."]
    return []


def petunjuk_log_ganda(seri: pd.Series, mode: str) -> str | None:
    """Peringatan bila data tampaknya sudah dalam log tetapi akan di-log lagi."""
    s = seri.dropna()
    if mode == "log" and len(s) and s.min() > 0 and s.max() < 30 and (s.max() / s.min()) < 3:
        return ("Nilai data kecil dan rentangnya sempit (seperti data yang sudah di-log). Jika kolom ini sudah "
                "berupa log, pilih 'Data sudah dalam bentuk log' agar tidak di-log dua kali.")
    return None


def ke_skala_model(seri: pd.Series, mode: str) -> pd.Series:
    return np.log(seri) if mode == "log" else seri.astype(float)


def ke_skala_asli(x, mode: str):
    """Kembalikan dari skala model ke skala asli data."""
    return np.exp(x) if mode in ("log", "sudah_log") else x


def potong(seri: pd.Series, awal: pd.Period, akhir: pd.Period) -> pd.Series:
    return seri.loc[awal:akhir]


# --------------------------------------------------------------------------- variabel independen (X)

def kolom_x_otomatis(df: pd.DataFrame, kolom_tanggal: str, kolom_target: str) -> list[str]:
    """Semua kolom selain tanggal dan target yang sebagian besar berisi angka."""
    hasil = []
    for k in df.columns:
        if k in (kolom_tanggal, kolom_target):
            continue
        nilai = ke_angka(df[k])
        terisi = df[k].notna()
        if terisi.any() and nilai[terisi].notna().mean() > 0.9:
            hasil.append(k)
    return hasil


def adalah_dummy(x: pd.Series) -> bool:
    """True bila nilai (yang terisi) hanya 0 dan/atau 1."""
    nilai = set(pd.Series(x).dropna().unique().tolist())
    return bool(nilai) and nilai <= {0, 1, 0.0, 1.0}


@dataclass
class HasilValidasiX:
    X: pd.DataFrame
    galat: list[str] = field(default_factory=list)
    peringatan: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.galat


def siapkan_x(df: pd.DataFrame, kolom_tanggal: str, kolom_x: list[str], frekuensi: str,
              index_target: pd.PeriodIndex) -> HasilValidasiX:
    """Bentuk tabel X dengan periode yang sama dengan target (termasuk periode masa depan bila ada)."""
    kode, _ = FREKUENSI[frekuensi]
    galat, peringatan = [], []
    df = df.loc[df[kolom_tanggal].notna()]
    periode = ke_periode(df[kolom_tanggal], kode)
    data = {}
    for k in kolom_x:
        nilai = ke_angka(df[k])
        tak_terbaca = nilai.isna() & df[k].notna()
        if tak_terbaca.mean() > 0.5:
            galat.append(f"Kolom X '{k}' sebagian besar bukan angka.")
            continue
        if tak_terbaca.any():
            peringatan.append(f"{int(tak_terbaca.sum())} nilai di kolom X '{k}' bukan angka dan dianggap kosong.")
        data[k] = pd.Series(nilai.to_numpy(), index=periode)
    X = pd.DataFrame(data)
    if X.index.duplicated().any():
        galat.append("Ada periode ganda pada data X; periksa frekuensi data.")
        return HasilValidasiX(X, galat, peringatan)
    akhir = max(index_target.max(), X.index.max()) if len(X) else index_target.max()
    X = X.reindex(pd.period_range(index_target.min(), akhir, freq=index_target.freq))
    return HasilValidasiX(X, galat, peringatan)


def transformasi_x(X: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, list[str]]:
    """mode 'tanpa' atau 'log'. Dengan 'log', kolom dummy (0/1) TIDAK di-log dan nama kolom menjadi log(X)."""
    if mode == "tanpa" or X.empty:
        return X.astype(float), []
    hasil, galat = {}, []
    for k in X.columns:
        if adalah_dummy(X[k]):
            hasil[k] = X[k].astype(float)
        elif (X[k].dropna() <= 0).any():
            galat.append(f"Kolom X '{k}' punya nilai ≤ 0 sehingga tidak bisa di-log; pilih transformasi X 'tanpa'.")
        else:
            hasil[f"log({k})"] = np.log(X[k].astype(float))
    return pd.DataFrame(hasil, index=X.index), galat


def bentuk_lag(X_model: pd.DataFrame, lags: tuple[int, ...] = (0,)) -> pd.DataFrame:
    """Matriks desain X: lag 0 memakai nama asli, lag k > 0 bernama X(-k)."""
    kolom = {}
    for k in X_model.columns:
        for lag in sorted(set(lags)):
            kolom[k if lag == 0 else f"{k}(-{lag})"] = X_model[k].shift(lag)
    return pd.DataFrame(kolom, index=X_model.index)


def nama_dasar_x(kolom_desain: str) -> str:
    """'log(A)(-2)' -> 'log(A)'."""
    return re.sub(r"\(-\d+\)$", "", kolom_desain)
