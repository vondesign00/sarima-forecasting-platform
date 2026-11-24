"""Grafik: Plotly untuk layar (interaktif), Matplotlib untuk file PNG unduhan.

Plotly diimpor di dalam fungsi agar fungsi PNG bisa diuji tanpa Plotly."""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Grafik memakai warna kontras agar mudah dibedakan (tema aplikasi navy-biru-putih-abu).
WARNA = {
    "latih": "#0B1F3A",            # navy: data aktual
    "uji": "#16A34A",              # hijau: aktual periode uji
    "backtest": "#DC2626",         # merah: ramalan periode uji
    "proyeksi": "#DC2626",         # merah: proyeksi ke depan
    "pita_backtest": "rgba(220,38,38,0.12)",
    "pita_proyeksi": "rgba(220,38,38,0.12)",
    "acf": "#1D4ED8",
}
WARNA_VARIAN = ["#F59E0B", "#7C3AED", "#0891B2", "#DB2777"]   # oranye, ungu, sian, merah muda
WARNA_BATANG = ["#DC2626", "#F59E0B", "#7C3AED", "#16A34A", "#64748B", "#94A3B8", "#CBD5E1"]
TATA_LETAK = dict(template="plotly_white", height=460, margin=dict(t=50, b=10, l=10, r=10),
                  legend=dict(orientation="h", y=-0.18), hovermode="x unified",
                  font=dict(color="#0B1F3A"), title_font=dict(color="#0B1F3A", size=16))


def _x(index: pd.PeriodIndex):
    return index.to_timestamp()


def grafik_seri(seri: pd.Series, judul: str = "", label_y: str = "Nilai"):
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=_x(seri.index), y=seri, mode="lines+markers", name=seri.name or "Data",
                               line=dict(color=WARNA["latih"]), marker=dict(size=4)))
    fig.update_layout(title=judul, yaxis_title=label_y, **TATA_LETAK)
    return fig


def _pita(fig, index, bawah, atas, warna, nama):
    fig.add_scatter(x=_x(index), y=atas, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=_x(index), y=bawah, fill="tonexty", fillcolor=warna, line=dict(width=0), name=nama,
                    hoverinfo="skip")


def _label_model(h) -> str:
    spek = getattr(h, "spek", None) or h.pengaturan.spek
    return spek.label()


def _garis_varian(h) -> list:
    """Garis tambahan untuk SARIMAX: (index, nilai, nama) varian ramalan selain ramalan utama."""
    garis = []
    if not getattr(h, "pakai_x", False):
        return garis
    if h.backtest is not None:
        for nama, t in h.backtest["tabel_varian"].items():
            garis.append((t.index, t["ramalan"], f"Uji — {nama}"))
        t0 = h.backtest_tanpa_x["tabel"]
        garis.append((t0.index, t0["ramalan"], "Uji — tanpa X"))
    for nama, r in h.ramalan_varian.items():
        if nama != h.nama_ramalan_utama:
            garis.append((r.index, r["ramalan"], f"Proyeksi — {nama}"))
    return garis


def grafik_hasil(h, label_y: str = "Nilai"):
    import plotly.graph_objects as go
    """Data latih, data uji, ramalan backtest, proyeksi ke depan, dan selang kepercayaan."""
    peng = h.pengaturan
    tingkat = peng.tingkat_kepercayaan
    fig = go.Figure()
    latih = h.y_asli.loc[peng.awal_latih:peng.akhir_latih]
    fig.add_scatter(x=_x(latih.index), y=latih, name="Aktual — data latih", mode="lines",
                    line=dict(color=WARNA["latih"]))
    if h.backtest is not None:
        t = h.backtest["tabel"]
        _pita(fig, t.index, t["batas_bawah"], t["batas_atas"], WARNA["pita_backtest"], f"Selang {tingkat}% (uji)")
        fig.add_scatter(x=_x(t.index), y=t["aktual"], name="Aktual — data uji", mode="lines+markers",
                        line=dict(color=WARNA["uji"]))
        fig.add_scatter(x=_x(t.index), y=t["ramalan"],
                        name="Ramalan periode uji" + (" — ex-post (X aktual)" if getattr(h, "pakai_x", False) else ""),
                        mode="lines+markers", line=dict(color=WARNA["backtest"], dash="dash"))
    sesudah = h.y_asli.loc[(peng.akhir_uji or peng.akhir_latih) + 1:h.akhir_data_proyeksi]
    if len(sesudah.dropna()):
        fig.add_scatter(x=_x(sesudah.index), y=sesudah, name="Aktual — setelah periode uji", mode="lines",
                        line=dict(color=WARNA["latih"], dash="dot"))
    r = h.ramalan
    _pita(fig, r.index, r["batas_bawah"], r["batas_atas"], WARNA["pita_proyeksi"], f"Selang {tingkat}% (proyeksi)")
    utama = getattr(h, "nama_ramalan_utama", "")
    fig.add_scatter(x=_x(r.index), y=r["ramalan"], name="Proyeksi ke depan" + (f" — {utama}" if utama else ""),
                    mode="lines+markers", line=dict(color=WARNA["proyeksi"]))
    for i, (idx, nilai, nama) in enumerate(_garis_varian(h)):
        fig.add_scatter(x=_x(idx), y=nilai, name=nama, mode="lines+markers",
                        line=dict(color=WARNA_VARIAN[i % len(WARNA_VARIAN)], dash="dot"), marker=dict(size=4))
    fig.update_layout(title=f"{_label_model(h)} — aktual dan ramalan", yaxis_title=label_y, **TATA_LETAK)
    return fig


def grafik_acf(diag: dict):
    import plotly.graph_objects as go
    a = diag["acf"]
    fig = go.Figure(go.Bar(x=a["lag"], y=a["acf"], marker_color=WARNA["acf"], name="ACF"))
    for tanda in (1, -1):
        fig.add_hline(y=tanda * diag["batas_acf"], line_dash="dash", line_color="#DC2626")
    fig.update_layout(title="ACF residual (garis putus = batas signifikansi 95%)", xaxis_title="Lag",
                      **{**TATA_LETAK, "height": 320, "hovermode": "closest"})
    return fig


def grafik_akurasi(akurasi: pd.DataFrame, ukuran: str = "MAPE (%)"):
    """Batang horizontal satu ukuran akurasi untuk semua model/pembanding (makin pendek makin baik)."""
    import plotly.graph_objects as go
    data = akurasi[ukuran].dropna().sort_values(ascending=False)
    warna = [WARNA_BATANG[list(akurasi.index).index(i) % len(WARNA_BATANG)] for i in data.index]
    fig = go.Figure(go.Bar(x=data.values, y=list(data.index), orientation="h", marker_color=warna,
                           text=[f"{v:,.2f}" for v in data.values], textposition="outside"))
    fig.update_layout(title=f"{ukuran} pada periode uji (makin kecil makin baik)",
                      **{**TATA_LETAK, "height": 120 + 45 * len(data), "hovermode": "closest",
                         "margin": dict(t=50, b=10, l=10, r=60)})
    return fig


def grafik_x(X, judul: str = "Variabel X"):
    """Satu garis per variabel X (sumbu-y sendiri-sendiri lewat subplot)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=len(X.columns), cols=1, shared_xaxes=True, subplot_titles=list(X.columns))
    for i, k in enumerate(X.columns, start=1):
        fig.add_trace(go.Scatter(x=_x(X.index), y=X[k], mode="lines+markers", name=k, showlegend=False,
                                 line=dict(color=WARNA_VARIAN[(i - 1) % len(WARNA_VARIAN)]), marker=dict(size=3)),
                      row=i, col=1)
    fig.update_layout(title=judul, **{**TATA_LETAK, "height": 120 + 170 * len(X.columns), "hovermode": "closest"})
    return fig


def png_hasil(h, label_y: str = "Nilai") -> bytes:
    """Versi Matplotlib dari grafik_hasil, untuk diunduh sebagai PNG."""
    peng = h.pengaturan
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    latih = h.y_asli.loc[peng.awal_latih:peng.akhir_latih]
    ax.plot(_x(latih.index), latih, color=WARNA["latih"], label="Aktual — data latih")
    if h.backtest is not None:
        t = h.backtest["tabel"]
        ax.fill_between(_x(t.index), t["batas_bawah"], t["batas_atas"], color=WARNA["backtest"], alpha=0.15)
        ax.plot(_x(t.index), t["aktual"], color=WARNA["uji"], marker="o", ms=3, label="Aktual — data uji")
        ax.plot(_x(t.index), t["ramalan"], color=WARNA["backtest"], ls="--", marker="o", ms=3,
                label="Ramalan periode uji" + (" — ex-post (X aktual)" if getattr(h, "pakai_x", False) else ""))
    sesudah = h.y_asli.loc[(peng.akhir_uji or peng.akhir_latih) + 1:h.akhir_data_proyeksi]
    if len(sesudah.dropna()):
        ax.plot(_x(sesudah.index), sesudah, color=WARNA["latih"], ls=":", label="Aktual — setelah periode uji")
    r = h.ramalan
    ax.fill_between(_x(r.index), r["batas_bawah"], r["batas_atas"], color=WARNA["proyeksi"], alpha=0.15,
                    label=f"Selang {peng.tingkat_kepercayaan}%")
    utama = getattr(h, "nama_ramalan_utama", "")
    ax.plot(_x(r.index), r["ramalan"], color=WARNA["proyeksi"], marker="o", ms=3,
            label="Proyeksi ke depan" + (f" — {utama}" if utama else ""))
    for i, (idx, nilai, nama) in enumerate(_garis_varian(h)):
        ax.plot(_x(idx), nilai, color=WARNA_VARIAN[i % len(WARNA_VARIAN)], ls=":", marker="o", ms=2, label=nama)
    ax.set_title(f"{_label_model(h)} — aktual dan ramalan")
    ax.set_ylabel(label_y)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.ticklabel_format(axis="y", style="plain")
    fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
