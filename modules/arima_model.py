"""
Estimasi (S)ARIMA / (S)ARIMAX dengan exact maximum likelihood, beserta statistik dan diagnostiknya.

Metodologi (berlaku untuk dataset apa pun):

1. Differencing dilakukan eksplisit sebelum estimasi, pada Y dan pada setiap variabel X:
       y*_t = (1 - L)^d (1 - L^s)^D y_t,   x*_t = (1 - L)^d (1 - L^s)^D x_t
   Sampel estimasi = data setelah differencing (d + D·s observasi awal hilang).
2. Model pada y*: regresi pada konstanta (dan X) dengan galat ARMA
       y*_t = C + β'x*_t + u_t,   φ(L) Φ(L^s) u_t = θ(L) Θ(L^s) ε_t
   sehingga C adalah RATA-RATA y* bersih dari pengaruh X (bila d = D = 0) atau drift (bila ada
   differencing), bukan intercept trend='c' statsmodels.
3. Likelihood eksak: state ARMA diinisialisasi dengan distribusi stasionernya, jadi
   semua observasi ikut dihitung (tidak ada burn-in).
4. Kovarians koefisien: outer product of gradients (OPG). Uji koefisien memakai distribusi t
   dengan derajat bebas n - k (k termasuk σ²).
5. Stasioneritas/invertibilitas tidak dipaksa (bisa diaktifkan lewat opsi).
6. Residual ε̂_t = inovasi terstandar × σ̂, sehingga Σ ε̂² = n·σ̂². Statistik ringkas (R², adj. R²,
   S.E. of regression, F, Durbin-Watson) dihitung dari residual ini dan dari y*.
   Kriteria informasi per observasi: (-2·LL + 2k)/n.

Proyeksi memakai model level (dengan differencing di dalam state space) yang parameternya
dikunci ke hasil estimasi di atas, sehingga ramalan dan selangnya otomatis kembali ke level.
Untuk SARIMAX, nilai X periode ramalan wajib diberikan (diisi manual atau hasil proyeksi X).

statsmodels diimpor di dalam fungsi agar modul lain tetap bisa diuji tanpa statsmodels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- spesifikasi

@dataclass
class SpesifikasiModel:
    order: tuple[int, int, int] = (0, 0, 0)
    seasonal_order: tuple[int, int, int, int] = (1, 0, 0, 12)
    konstanta: bool = True
    paksa_stasioner: bool = False
    paksa_invertible: bool = False
    exog: tuple[str, ...] = ()          # nama kolom X (sudah ditransformasi/di-lag) yang ikut model

    def label(self, dengan_x: bool = True) -> str:
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        musiman = bool(P or D or Q)
        nama = ("SARIMA" if musiman else "ARIMA") + ("X" if self.exog else "")
        teks = f"{nama}({p},{d},{q})" + (f"({P},{D},{Q}){s}" if musiman else "")
        teks += " + C" if self.konstanta else ""
        if self.exog and dengan_x:
            teks += " + " + " + ".join(self.exog)
        return teks

    def tanpa_x(self) -> "SpesifikasiModel":
        return replace(self, exog=())

    def jumlah_parameter_arma(self) -> int:
        p, _, q = self.order
        P, _, Q, _ = self.seasonal_order
        return p + q + P + Q

    def musiman_statsmodels(self, tanpa_differencing: bool = False) -> tuple[int, int, int, int]:
        """seasonal_order untuk statsmodels; (0,0,0,0) bila tidak ada komponen musiman."""
        P, D, Q, s = self.seasonal_order
        if tanpa_differencing:
            D = 0
        return (P, D, Q, s) if (P or D or Q) else (0, 0, 0, 0)

    def observasi_hilang(self) -> int:
        """Jumlah observasi awal yang hilang karena differencing."""
        d = self.order[1]
        _, D, _, s = self.seasonal_order
        return d + D * s

    def minimal_observasi(self) -> int:
        """Batas bawah kasar agar model bisa diestimasi."""
        p, _, q = self.order
        P, _, Q, s = self.seasonal_order
        lag_maks = max(p, P * s if P else 0, q, Q * s if Q else 0)
        return max(10, self.observasi_hilang() + lag_maks + self.jumlah_parameter_arma() + len(self.exog) + 5)


# --------------------------------------------------------------------------- utilitas

def diferensiasi(y, spek: SpesifikasiModel):
    """(1 - L)^d (1 - L^s)^D untuk Series atau DataFrame, tanpa observasi awal yang hilang."""
    d = spek.order[1]
    _, D, _, s = spek.seasonal_order
    hasil = y.astype(float)
    for _ in range(D):
        hasil = hasil - hasil.shift(s)
    for _ in range(d):
        hasil = hasil - hasil.shift(1)
    return hasil.iloc[spek.observasi_hilang():]


def regresor_konstanta(n: int, spek: SpesifikasiModel) -> np.ndarray:
    """Regresor x (level) yang setelah differencing menjadi deret 1.

    Dipakai di model level agar koefisiennya sama dengan C pada data hasil differencing.
    """
    d = spek.order[1]
    _, D, _, s = spek.seasonal_order
    x = np.ones(n)
    for _ in range(D):
        for t in range(s, n):
            x[t] += x[t - s]
    for _ in range(d):
        x = np.cumsum(x)
    return x


def matriks_exog(index, konstanta_nilai, X: pd.DataFrame | None, spek: SpesifikasiModel):
    """Gabungan kolom C (bila dipakai) dan kolom X, urut seperti spek.exog. None bila kosong."""
    bagian = []
    if spek.konstanta:
        bagian.append(pd.DataFrame({"C": konstanta_nilai}, index=index))
    if spek.exog:
        bagian.append(X.loc[index, list(spek.exog)].astype(float))
    return pd.concat(bagian, axis=1) if bagian else None


def _kelas_sarimax():
    """SARIMAX dengan inisialisasi eksak: state differencing difus, state ARMA stasioner."""
    from statsmodels.tsa.statespace.initialization import Initialization
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    class SARIMAXEksak(SARIMAX):
        def initialize_default(self, approximate_diffuse_variance=None):
            if approximate_diffuse_variance is None:
                approximate_diffuse_variance = self.ssm.initial_variance
            init = Initialization(self.k_states, approximate_diffuse_variance=approximate_diffuse_variance)
            k_diff = self._k_states_diff
            if k_diff:
                init.set((0, k_diff), "diffuse")
            if self.k_states > k_diff:
                init.set((k_diff, self.k_states), "stationary")
            self.ssm.initialization = init

    return SARIMAXEksak


def selaraskan_x(y_model: pd.Series, X: pd.DataFrame | None, spek: SpesifikasiModel):
    """Potong awal sampel bila X kosong di awal (mis. akibat lag) dan periksa X lengkap.

    Mengembalikan (y_model, X) yang sudah selaras, plus catatan.
    """
    if not spek.exog:
        return y_model, None, []
    if X is None:
        raise ValueError("Model memakai variabel X, tetapi data X tidak diberikan.")
    kurang = [k for k in spek.exog if k not in X.columns]
    if kurang:
        raise ValueError(f"Kolom X tidak ditemukan: {', '.join(kurang)}.")
    Xs = X.reindex(y_model.index)[list(spek.exog)].astype(float)
    lengkap = Xs.notna().all(axis=1)
    if not lengkap.any():
        raise ValueError("Variabel X kosong di seluruh periode estimasi.")
    awal = lengkap.idxmax()
    catatan = []
    if awal != y_model.index[0]:
        catatan.append(f"Sampel estimasi dimulai {awal} karena X belum tersedia sebelumnya (mis. akibat lag X).")
    y_model, Xs = y_model.loc[awal:], Xs.loc[awal:]
    if Xs.isna().any().any():
        kosong = Xs.index[Xs.isna().any(axis=1)]
        raise ValueError(f"Variabel X kosong di dalam periode estimasi ({', '.join(str(p) for p in kosong[:6])}). "
                         "Lengkapi data X atau persempit periode.")
    return y_model, Xs, catatan


# --------------------------------------------------------------------------- estimasi

@dataclass
class ModelTerestimasi:
    """Hasil estimasi satu model.

    res      : hasil statsmodels pada data setelah differencing (koefisien, LL, residual)
    spek     : spesifikasi model
    y_model  : data skala model (mis. log) SEBELUM differencing, dipakai untuk proyeksi
    X        : matriks X (skala model, sebelum differencing) selaras dengan y_model, atau None
    catatan  : catatan proses estimasi (mis. fallback ke batasan stasioner)
    """
    res: object
    spek: SpesifikasiModel
    y_model: pd.Series
    X: pd.DataFrame | None = None
    catatan: list[str] = field(default_factory=list)

    @property
    def params(self) -> pd.Series:
        return self.res.params

    @property
    def bse(self) -> pd.Series:
        return self.res.bse

    @property
    def llf(self) -> float:
        return float(self.res.llf)

    @property
    def nobs(self) -> int:
        return int(self.res.nobs)

    @property
    def df_resid(self) -> int:
        return max(self.nobs - len(self.params), 1)

    @property
    def resid(self) -> pd.Series:
        """Residual ε̂_t = inovasi terstandar × σ̂ (Σ ε̂² = n·σ̂²)."""
        std = np.asarray(self.res.standardized_forecasts_error)[0]
        sigma = float(np.sqrt(self.res.params["sigma2"]))
        return pd.Series(std * sigma, index=self.y_estimasi.index).dropna()

    @property
    def y_estimasi(self) -> pd.Series:
        """Variabel dependen yang benar-benar diestimasi (setelah differencing)."""
        return diferensiasi(self.y_model, self.spek)

    def summary(self):
        return self.res.summary()

    def model_level(self):
        """Model level dengan parameter terkunci, untuk ramalan dalam skala model."""
        spek = self.spek
        kelas = _kelas_sarimax()
        x = regresor_konstanta(len(self.y_model), spek)
        model = kelas(
            self.y_model.astype(float),
            exog=matriks_exog(self.y_model.index, x, self.X, spek),
            order=spek.order, seasonal_order=spek.musiman_statsmodels(), trend="n",
            enforce_stationarity=False, enforce_invertibility=False,
        )
        return model.filter(np.asarray(self.res.params))


def _fit(y_est: pd.Series, X_est: pd.DataFrame | None, spek: SpesifikasiModel, paksa_stasioner: bool):
    p, _, q = spek.order
    kelas = _kelas_sarimax()
    model = kelas(
        y_est,
        exog=matriks_exog(y_est.index, np.ones(len(y_est)), X_est, spek),
        order=(p, 0, q), seasonal_order=spek.musiman_statsmodels(tanpa_differencing=True), trend="n",
        enforce_stationarity=paksa_stasioner, enforce_invertibility=spek.paksa_invertible,
    )
    return model.fit(disp=False, maxiter=2000, cov_type="opg")


def _stasioner(res) -> bool:
    akar = np.asarray(res.arroots) if len(res.arroots) else np.array([])
    return bool(np.all(np.abs(akar) > 1)) if len(akar) else True


def estimasi(y_model: pd.Series, spek: SpesifikasiModel, X: pd.DataFrame | None = None) -> ModelTerestimasi:
    """Estimasi model pada data skala model (mis. log), sebelum differencing.

    X: DataFrame variabel independen dalam skala model (indeks periode, boleh lebih panjang dari
       y_model); hanya kolom spek.exog yang dipakai.
    """
    import warnings

    y_model = y_model.astype(float)
    y_model, X_sel, catatan = selaraskan_x(y_model, X, spek)
    y_est = diferensiasi(y_model, spek)
    X_est = diferensiasi(X_sel, spek) if X_sel is not None else None
    res = None
    if not spek.paksa_stasioner:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = _fit(y_est, X_est, spek, paksa_stasioner=False)
            if not (np.isfinite(res.llf) and _stasioner(res)):
                res = None
        except Exception:
            res = None
        if res is None:
            catatan.append("Estimasi tanpa batasan keluar dari wilayah stasioner; model diestimasi ulang dengan "
                           "batasan stasioner.")
    if res is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _fit(y_est, X_est, spek, paksa_stasioner=True)
    return ModelTerestimasi(res=res, spek=spek, y_model=y_model, X=X_sel, catatan=catatan)


def konvergen(model: ModelTerestimasi) -> bool:
    retvals = getattr(model.res, "mle_retvals", None) or {}
    return bool(retvals.get("converged", True))


# --------------------------------------------------------------------------- pemilihan orde otomatis

def pilih_orde(y_model: pd.Series, s: int, X: pd.DataFrame | None = None, exog: tuple[str, ...] = (),
               konstanta: bool = True, d: int = 0, D: int = 0, maks_p: int = 2, maks_q: int = 2,
               maks_P: int = 1, maks_Q: int = 1, kriteria: str = "AIC") -> tuple[pd.DataFrame, list]:
    """Estimasi semua kombinasi (p,q)(P,Q) hingga batas maksimum, urutkan menurut kriteria informasi.

    Semua model memakai sampel yang sama (exact ML), sehingga kriterianya sebanding.
    Mengembalikan (tabel, daftar SpesifikasiModel urut terbaik).
    """
    kolom_kriteria = {"AIC": "Akaike info criterion (AIC)", "SC": "Schwarz criterion (SC/BIC)",
                      "HQ": "Hannan-Quinn criterion (HQ)"}[kriteria]
    musiman = s >= 2
    baris, spek_list = [], []
    for p, q, P, Q in product(range(maks_p + 1), range(maks_q + 1),
                              range(maks_P + 1 if musiman else 1), range(maks_Q + 1 if musiman else 1)):
        spek = SpesifikasiModel(order=(p, d, q), seasonal_order=(P, D, Q, s if (musiman or D) else 0),
                                konstanta=konstanta, exog=tuple(exog))
        try:
            m = estimasi(y_model, spek, X)
            st = statistik_ringkas(m)
            baris.append({"ARMA (p,q)(P,Q)": f"({p},{q})({P},{Q})", "Model": spek.label(dengan_x=False),
                          "AIC": st["Akaike info criterion (AIC)"], "SC": st["Schwarz criterion (SC/BIC)"],
                          "HQ": st["Hannan-Quinn criterion (HQ)"], "Log likelihood": st["Log likelihood"],
                          "Konvergen": konvergen(m), "Batasan stasioner": bool(m.catatan),
                          "_urut": st[kolom_kriteria]})
        except Exception as e:  # model gagal diestimasi tetap dilaporkan
            baris.append({"ARMA (p,q)(P,Q)": f"({p},{q})({P},{Q})", "Model": spek.label(dengan_x=False),
                          "AIC": np.nan, "SC": np.nan, "HQ": np.nan, "Log likelihood": np.nan,
                          "Konvergen": False, "Batasan stasioner": False, "_urut": np.inf,
                          "Catatan": str(e)[:80]})
        spek_list.append(spek)
    tabel = pd.DataFrame(baris)
    urutan = tabel["_urut"].argsort(kind="stable").to_numpy()
    tabel = tabel.iloc[urutan].drop(columns="_urut").reset_index(drop=True)
    tabel.index = tabel.index + 1
    return tabel, [spek_list[i] for i in urutan]


# --------------------------------------------------------------------------- tabel & statistik

def nama_parameter(nama: str) -> str:
    """Nama parameter statsmodels -> notasi ringkas (ar.S.L12 -> SAR(12), sigma2 -> SIGMASQ)."""
    if nama == "sigma2":
        return "SIGMASQ"
    m = re.match(r"^(ar|ma)\.(S\.)?L(\d+)$", nama)
    if m:
        jenis, musiman, lag = m.groups()
        return f"{'S' if musiman else ''}{jenis.upper()}({lag})"
    return nama


def _urutan_parameter(nama: str) -> int:
    """C, X, AR, SAR, MA, SMA, SIGMASQ."""
    if nama == "sigma2":
        return 9
    for i, awal in enumerate(("ar.L", "ar.S.", "ma.L", "ma.S."), start=2):
        if nama.startswith(awal):
            return i
    return 0 if nama == "C" else 1


def tabel_koefisien(model: ModelTerestimasi) -> pd.DataFrame:
    res = model.res
    t = res.params / res.bse
    tabel = pd.DataFrame({
        "Koefisien": res.params,
        "Std. Error": res.bse,
        "t-Statistic": t,
        "Prob.": 2 * stats.t.sf(np.abs(t), model.df_resid),
    })
    tabel = tabel.iloc[sorted(range(len(tabel)), key=lambda i: (_urutan_parameter(tabel.index[i]), i))]
    tabel.index = [nama_parameter(n) for n in tabel.index]
    tabel.index.name = "Variabel"
    return tabel


def statistik_ringkas(model: ModelTerestimasi) -> pd.Series:
    n = model.nobs
    k = len(model.params)  # termasuk SIGMASQ
    ll = model.llf
    e = model.resid
    y = model.y_estimasi.reindex(e.index)
    ssr = float((e ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ssr / sst if sst > 0 else np.nan
    df = n - k
    if k > 1 and df > 0 and np.isfinite(r2) and r2 < 1:
        f_stat = (r2 / (k - 1)) / ((1 - r2) / df)
        prob_f = float(stats.f.sf(f_stat, k - 1, df))
    else:
        f_stat = prob_f = np.nan
    return pd.Series({
        "Jumlah observasi": n,
        "Observasi hilang karena differencing": model.spek.observasi_hilang(),
        "R-squared": r2,
        "Adjusted R-squared": 1 - (1 - r2) * (n - 1) / df if df > 0 else np.nan,
        "S.E. of regression": float(np.sqrt(ssr / df)) if df > 0 else np.nan,
        "Sum squared resid": ssr,
        "Log likelihood": ll,
        "F-statistic": f_stat,
        "Prob(F-statistic)": prob_f,
        "Mean dependent var": float(y.mean()),
        "S.D. dependent var": float(y.std(ddof=1)),
        "Akaike info criterion (AIC)": (-2 * ll + 2 * k) / n,
        "Schwarz criterion (SC/BIC)": (-2 * ll + k * np.log(n)) / n,
        "Hannan-Quinn criterion (HQ)": (-2 * ll + 2 * k * np.log(np.log(n))) / n,
        "Durbin-Watson stat": float((np.diff(e.to_numpy()) ** 2).sum() / ssr) if ssr > 0 else np.nan,
    })


def diagnostik(model: ModelTerestimasi) -> dict:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf

    res, spek = model.res, model.spek
    resid = model.resid
    s = spek.seasonal_order[3]
    df_model = spek.jumlah_parameter_arma()
    kandidat = {s, 2 * s, 12, 24} if s > 1 else {6, 12, 24}
    lag_uji = sorted({lag for lag in kandidat if df_model < lag < len(resid)})
    ljung_box = acorr_ljungbox(resid, lags=lag_uji, model_df=df_model) if lag_uji else pd.DataFrame()

    jb = stats.jarque_bera(resid)
    nlags = max(1, min(max(2 * s, 12), 36, len(resid) - 1))
    nilai_acf = acf(resid, nlags=nlags, fft=False, missing="drop")[1:]

    akar_ar = 1 / np.abs(res.arroots) if len(res.arroots) else np.array([])
    akar_ma = 1 / np.abs(res.maroots) if len(res.maroots) else np.array([])

    return {
        "ljung_box": ljung_box,
        "jarque_bera": {"statistik": float(jb.statistic), "p_value": float(jb.pvalue)},
        "acf": pd.DataFrame({"lag": np.arange(1, nlags + 1), "acf": nilai_acf}),
        "batas_acf": 1.96 / np.sqrt(len(resid)),
        "akar_ar_invers_maks": float(akar_ar.max()) if len(akar_ar) else None,
        "akar_ma_invers_maks": float(akar_ma.max()) if len(akar_ma) else None,
        "peringatan": list(model.catatan) + peringatan_model(res.params, res.bse, akar_ar, akar_ma, s),
    }


def peringatan_model(params: pd.Series, bse: pd.Series, akar_ar, akar_ma, s: int,
                     ambang: float = 0.99) -> list[str]:
    """Deteksi model yang tidak bisa ditafsirkan seperti ARIMA biasa."""
    pesan = []
    if len(akar_ar) and np.max(akar_ar) >= ambang:
        pesan.append(f"Akar AR invers = {np.max(akar_ar):.4f} (≥ {ambang}): komponen AR di/dekat unit root, "
                     "model tidak stasioner.")
    if len(akar_ma) and np.max(akar_ma) >= ambang:
        pesan.append(f"Akar MA invers = {np.max(akar_ma):.4f} (≥ {ambang}): komponen MA tidak/nyaris tidak invertible.")
    for ar, ma in ((f"ar.S.L{s}", f"ma.S.L{s}"), ("ar.L1", "ma.L1")):
        if ar in params.index and ma in params.index:
            phi, theta = params[ar], params[ma]
            if abs(phi + theta) < 0.05 and abs(phi) > 0.5:
                pesan.append(f"{nama_parameter(ar)} = {phi:.4f} dan {nama_parameter(ma)} = {theta:.4f} hampir saling "
                             "meniadakan (pembatalan akar); t-statistik yang sangat besar tidak bermakna.")
    if (bse.drop("sigma2", errors="ignore") < 1e-3).any():
        pesan.append("Ada standard error sangat kecil (< 0,001): estimasi berada di batas ruang parameter.")
    return pesan


def ringkasan_teks(model: ModelTerestimasi, stat: pd.Series) -> str:
    """Ringkasan model sebagai teks (untuk diunduh)."""
    garis = [f"Model: {model.spek.label()}", ""]
    garis.append(tabel_koefisien(model).to_string(float_format=lambda v: f"{v:.6f}"))
    garis += ["", stat.to_string(float_format=lambda v: f"{v:.6f}"), "", "--- statsmodels summary ---",
              str(model.summary())]
    return "\n".join(garis)
