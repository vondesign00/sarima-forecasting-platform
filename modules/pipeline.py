"""Satu pintu untuk menjalankan seluruh alur: validasi → backtest → proyeksi → evaluasi.

Tanpa X  : satu model (S)ARIMA.
Dengan X : model (S)ARIMAX dibandingkan langsung dengan model yang sama TANPA X, pada sampel,
           periode latih/uji, dan orde ARMA yang sama. Ramalan SARIMAX ditampilkan dua versi:
           ex-post (X aktual / X diisi manual) dan ex-ante (X diproyeksi SARIMA).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from . import arima_model as am
from . import evaluation as ev
from . import forecasting as fc
from .data_preprocessing import (bentuk_lag, cek_transformasi, ke_skala_asli, ke_skala_model, label_periode,
                                 transformasi_x)

NAMA_EX_POST_UJI = "ex-post (X aktual)"
NAMA_EX_ANTE = "ex-ante (X diproyeksi SARIMA)"
NAMA_EX_POST_DEPAN = "ex-post (X diisi manual)"
NAMA_TANPA_X = "tanpa X"


@dataclass
class Pengaturan:
    spek: am.SpesifikasiModel          # orde model; kolom X diatur lewat kolom_x
    transformasi: str                  # "log" | "sudah_log" | "tanpa"
    awal_latih: pd.Period
    akhir_latih: pd.Period
    akhir_uji: pd.Period | None        # None = tanpa periode uji
    horizon: int = 12
    tingkat_kepercayaan: int = 95
    koreksi_bias: bool = False
    # --- variabel independen (SARIMAX)
    kolom_x: tuple[str, ...] = ()
    transformasi_x: str = "tanpa"      # "tanpa" | "log" (kolom dummy 0/1 tidak di-log)
    lag_x: tuple[int, ...] = (0,)
    proyeksi_x_sarima: bool = True     # proyeksikan X dengan SARIMA (ex-ante)
    maks_orde_x: tuple[int, int, int, int] = (1, 1, 1, 1)


@dataclass
class HasilProyeksi:
    pengaturan: Pengaturan
    y_asli: pd.Series
    backtest: dict | None
    akurasi: pd.DataFrame | None
    hasil_akhir: am.ModelTerestimasi
    koefisien_akhir: pd.DataFrame
    statistik_akhir: pd.Series
    diagnostik_akhir: dict
    ramalan: pd.DataFrame
    akhir_data_proyeksi: pd.Period
    peringatan: list[str] = field(default_factory=list)
    # --- khusus SARIMAX
    ramalan_varian: dict = field(default_factory=dict)     # nama -> DataFrame ramalan ke depan
    nama_ramalan_utama: str = ""
    hasil_tanpa_x: am.ModelTerestimasi | None = None
    backtest_tanpa_x: dict | None = None
    info_x: dict = field(default_factory=dict)             # orde SARIMA tiap X, kolom desain
    X_model: pd.DataFrame | None = None                     # X skala model (tanpa lag)

    @property
    def spek(self) -> am.SpesifikasiModel:
        return self.hasil_akhir.spek

    @property
    def pakai_x(self) -> bool:
        return bool(self.spek.exog)


class GalatPengaturan(Exception):
    pass


def panjang_uji_bawaan(n_terisi: int, s: int) -> int:
    """Panjang periode uji bawaan: satu musim penuh bila data cukup panjang, selain itu ±20% data."""
    if s > 1 and n_terisi > 3 * s:
        return s
    return max(1, n_terisi // 5)


def periode_otomatis(seri: pd.Series, s: int, pakai_uji: bool = True) -> tuple:
    """(awal_latih, akhir_latih, akhir_uji) bawaan dari data itu sendiri, tanpa tanggal tetap."""
    terisi = seri.dropna().index
    awal = terisi[0]
    if not pakai_uji:
        return awal, terisi[-1], None
    n_uji = panjang_uji_bawaan(len(terisi), s)
    return awal, terisi[-1 - n_uji], terisi[-1]


def siapkan_desain_x(X_dasar: pd.DataFrame | None, peng: Pengaturan):
    """(X_model tanpa lag, X_desain dengan lag, galat)."""
    if not peng.kolom_x:
        return None, None, []
    if X_dasar is None:
        return None, None, ["Kolom X dipilih tetapi data X tidak diberikan."]
    kurang = [k for k in peng.kolom_x if k not in X_dasar.columns]
    if kurang:
        return None, None, [f"Kolom X tidak ditemukan: {', '.join(kurang)}."]
    X_model, galat = transformasi_x(X_dasar[list(peng.kolom_x)], peng.transformasi_x)
    return X_model, bentuk_lag(X_model, peng.lag_x), galat


def periksa(seri: pd.Series, peng: Pengaturan, X_dasar: pd.DataFrame | None = None) -> list[str]:
    galat = list(cek_transformasi(seri.loc[peng.awal_latih:], peng.transformasi))
    if peng.akhir_latih < peng.awal_latih:
        galat.append("Akhir data latih harus setelah awal data latih.")
    if peng.akhir_uji is not None and peng.akhir_uji <= peng.akhir_latih:
        galat.append("Akhir data uji harus setelah akhir data latih.")
    spek = replace(peng.spek, exog=tuple(f"x{i}" for i in range(len(peng.kolom_x) * len(set(peng.lag_x)))))
    n_latih = int(seri.loc[peng.awal_latih:peng.akhir_latih].notna().sum())
    minimal = spek.minimal_observasi()
    if n_latih < minimal:
        galat.append(f"Data latih hanya {n_latih} observasi; model {peng.spek.label()} butuh minimal ±{minimal}. "
                     "Perpanjang data latih atau sederhanakan model.")
    P, D, Q, s = peng.spek.seasonal_order
    if (P or D or Q) and s < 2:
        galat.append("Komponen musiman (P, D, Q) membutuhkan panjang musim s ≥ 2. "
                     "Untuk data tanpa musim, isi P = D = Q = 0.")
    if peng.horizon < 1:
        galat.append("Horizon proyeksi minimal 1 periode.")
    if any(lag < 0 for lag in peng.lag_x):
        galat.append("Lag X tidak boleh negatif.")
    galat += siapkan_desain_x(X_dasar, peng)[2]
    return galat


def _label(spek: am.SpesifikasiModel, varian: str = "") -> str:
    return spek.label(dengan_x=False) + (f" — {varian}" if varian else "")


def jalankan(seri: pd.Series, peng: Pengaturan, X_dasar: pd.DataFrame | None = None) -> HasilProyeksi:
    """Jalankan seluruh alur. X_dasar: tabel X mentah (indeks periode, boleh memuat periode masa depan)."""
    galat = periksa(seri, peng, X_dasar)
    if galat:
        raise GalatPengaturan("\n".join(galat))

    alpha = 1 - peng.tingkat_kepercayaan / 100
    y_model = ke_skala_model(seri, peng.transformasi)
    y_asli = ke_skala_asli(y_model, peng.transformasi)
    X_model, X_desain, _ = siapkan_desain_x(X_dasar, peng)
    spek = replace(peng.spek, exog=tuple(X_desain.columns) if X_desain is not None else ())
    spek_0 = spek.tanpa_x()
    s = spek.seasonal_order[3]
    pakai_x = bool(spek.exog)
    peringatan, info_x = [], {}
    if pakai_x:
        info_x["Kolom X dalam model"] = ", ".join(spek.exog)

    # ------------------------------------------------------------------ backtest
    bt = bt0 = akurasi = None
    if peng.akhir_uji is not None:
        y_uji = y_model.loc[peng.akhir_latih + 1:peng.akhir_uji]
        lain = {}
        if pakai_x and peng.proyeksi_x_sarima:
            Xf, info = fc.ramal_x_sarima(X_model, peng.akhir_latih, len(y_uji), s, peng.maks_orde_x)
            lain[NAMA_EX_ANTE] = fc.desain_x_depan(X_model, Xf, peng.lag_x)
            info_x.update({f"Proyeksi X (backtest) — {k}": v for k, v in info.items()})
        bt = fc.backtest(y_model, spek, peng.awal_latih, peng.akhir_latih, peng.akhir_uji,
                         peng.transformasi, alpha, peng.koreksi_bias, X=X_desain, X_uji_lain=lain)
        aktual = bt["tabel"]["aktual"]
        if pakai_x:
            ramal = {_label(spek, NAMA_EX_POST_UJI): bt["tabel"]["ramalan"]}
            ramal.update({_label(spek, k): t["ramalan"] for k, t in bt["tabel_varian"].items()})
            bt0 = fc.backtest(y_model, spek_0, bt["y_latih"].index[0], peng.akhir_latih, peng.akhir_uji,
                              peng.transformasi, alpha, peng.koreksi_bias)
            ramal[_label(spek_0, NAMA_TANPA_X)] = bt0["tabel"]["ramalan"]
        else:
            ramal = {spek.label(): bt["tabel"]["ramalan"]}
        akurasi = ev.tabel_akurasi(aktual, {**ramal, **bt["pembanding"]})
        if aktual.isna().any():
            peringatan.append(f"{int(aktual.isna().sum())} periode uji tanpa nilai aktual tidak dihitung di metrik.")

    # ------------------------------------------------------------------ proyeksi ke depan
    akhir_data = y_model.loc[peng.awal_latih:].last_valid_index()
    y_final = y_model.loc[peng.awal_latih:akhir_data]
    hasil_akhir = am.estimasi(y_final, spek, X_desain)
    index_depan = pd.period_range(akhir_data + 1, periods=peng.horizon, freq=y_model.index.freq)
    ramalan_varian, hasil_0, nama_utama = {}, None, ""
    if pakai_x:
        X_manual = X_desain.reindex(index_depan)
        lengkap = X_manual.notna().all(axis=1).to_numpy()
        n_manual = int(lengkap.argmin()) if not lengkap.all() else len(lengkap)
        if n_manual:
            ramalan_varian[NAMA_EX_POST_DEPAN] = fc.proyeksi(hasil_akhir, n_manual, peng.transformasi, alpha,
                                                             peng.koreksi_bias, X_manual.iloc[:n_manual])
            if n_manual < peng.horizon:
                peringatan.append(f"Nilai X manual hanya tersedia {n_manual} dari {peng.horizon} periode ke depan; "
                                  "ramalan ex-post dibatasi sepanjang itu.")
        if peng.proyeksi_x_sarima:
            Xf, info = fc.ramal_x_sarima(X_model, akhir_data, peng.horizon, s, peng.maks_orde_x)
            info_x.update({f"Proyeksi X (ke depan) — {k}": v for k, v in info.items()})
            ramalan_varian[NAMA_EX_ANTE] = fc.proyeksi(hasil_akhir, peng.horizon, peng.transformasi, alpha,
                                                       peng.koreksi_bias, fc.desain_x_depan(X_model, Xf, peng.lag_x))
        if not ramalan_varian:
            raise GalatPengaturan("Nilai X untuk periode ke depan tidak ada. Isi nilai X pada baris tanggal ke "
                                  "depan di file data, atau aktifkan proyeksi X dengan SARIMA.")
        hasil_0 = am.estimasi(y_final.loc[hasil_akhir.y_model.index[0]:], spek_0)
        ramalan_varian[NAMA_TANPA_X] = fc.proyeksi(hasil_0, peng.horizon, peng.transformasi, alpha,
                                                   peng.koreksi_bias)
        if NAMA_EX_POST_DEPAN in ramalan_varian and n_manual == peng.horizon:
            nama_utama = NAMA_EX_POST_DEPAN
        else:
            nama_utama = NAMA_EX_ANTE if NAMA_EX_ANTE in ramalan_varian else NAMA_EX_POST_DEPAN
        ramalan = ramalan_varian[nama_utama]
    else:
        ramalan = fc.proyeksi(hasil_akhir, peng.horizon, peng.transformasi, alpha, peng.koreksi_bias)

    if peng.akhir_uji is not None and akhir_data > peng.akhir_latih:
        peringatan.append(f"Proyeksi ke depan memakai model yang diestimasi ulang dengan data "
                          f"{label_periode(hasil_akhir.y_model.index[0])} s.d. {label_periode(akhir_data)} "
                          "(termasuk periode uji), sehingga koefisiennya bisa berbeda dari model backtest.")

    return HasilProyeksi(
        pengaturan=peng, y_asli=y_asli, backtest=bt, akurasi=akurasi,
        hasil_akhir=hasil_akhir, koefisien_akhir=am.tabel_koefisien(hasil_akhir),
        statistik_akhir=am.statistik_ringkas(hasil_akhir), diagnostik_akhir=am.diagnostik(hasil_akhir),
        ramalan=ramalan, akhir_data_proyeksi=akhir_data, peringatan=peringatan,
        ramalan_varian=ramalan_varian, nama_ramalan_utama=nama_utama, hasil_tanpa_x=hasil_0,
        backtest_tanpa_x=bt0, info_x=info_x, X_model=X_model,
    )


# --------------------------------------------------------------------------- tabel ringkasan

def tabel_hasil(h: HasilProyeksi) -> pd.DataFrame:
    """Tabel gabungan: periode uji (aktual, ramalan, galat) + proyeksi ke depan (ramalan utama)."""
    bagian = []
    if h.backtest is not None:
        t = h.backtest["tabel"].copy()
        t.insert(0, "Jenis", "Uji (backtest)" + (f" — {NAMA_EX_POST_UJI}" if h.pakai_x else ""))
        bagian.append(t)
    r = h.ramalan[["ramalan", "batas_bawah", "batas_atas"]].copy()
    r.insert(0, "Jenis", "Proyeksi ke depan" + (f" — {h.nama_ramalan_utama}" if h.pakai_x else ""))
    r["aktual"], r["galat"] = float("nan"), float("nan")
    bagian.append(r)
    tabel = pd.concat(bagian)[["Jenis", "aktual", "ramalan", "galat", "batas_bawah", "batas_atas"]]
    tabel.index = [label_periode(p) for p in tabel.index]
    tabel.index.name = "Periode"
    return tabel.rename(columns={"aktual": "Aktual", "ramalan": "Ramalan", "galat": "Galat (aktual − ramalan)",
                                 "batas_bawah": "Batas bawah", "batas_atas": "Batas atas"})


def tabel_perbandingan_ramalan(h: HasilProyeksi) -> pd.DataFrame:
    """Ramalan semua varian berdampingan (kolom), untuk periode uji dan ke depan. Kosong bila tanpa X."""
    if not h.pakai_x:
        return pd.DataFrame()
    bagian = []
    if h.backtest is not None:
        kol = {"Aktual": h.backtest["tabel"]["aktual"], NAMA_EX_POST_UJI: h.backtest["tabel"]["ramalan"]}
        kol.update({k: t["ramalan"] for k, t in h.backtest["tabel_varian"].items()})
        kol[NAMA_TANPA_X] = h.backtest_tanpa_x["tabel"]["ramalan"]
        t = pd.DataFrame(kol)
        t.insert(0, "Jenis", "Uji (backtest)")
        bagian.append(t)
    kol = {k: r["ramalan"] for k, r in h.ramalan_varian.items()}
    t = pd.DataFrame(kol)
    t.insert(0, "Jenis", "Proyeksi ke depan")
    bagian.append(t)
    tabel = pd.concat(bagian)
    tabel.index = [label_periode(p) for p in tabel.index]
    tabel.index.name = "Periode"
    return tabel


def tabel_perbandingan_model(h: HasilProyeksi, pakai_backtest: bool = True) -> pd.DataFrame:
    """Statistik model dengan X vs tanpa X (sampel sama). Kosong bila tanpa X."""
    if not h.pakai_x:
        return pd.DataFrame()
    if pakai_backtest and h.backtest is not None:
        m_x, m_0 = h.backtest["hasil"], h.backtest_tanpa_x["hasil"]
    else:
        m_x, m_0 = h.hasil_akhir, h.hasil_tanpa_x
    pilih = ["Jumlah observasi", "Log likelihood", "Akaike info criterion (AIC)", "Schwarz criterion (SC/BIC)",
             "Hannan-Quinn criterion (HQ)", "R-squared", "Adjusted R-squared", "S.E. of regression",
             "Durbin-Watson stat"]
    return pd.DataFrame({
        f"Dengan X — {m_x.spek.label()}": am.statistik_ringkas(m_x)[pilih],
        f"Tanpa X — {m_0.spek.label()}": am.statistik_ringkas(m_0)[pilih],
    })
