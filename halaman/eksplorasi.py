import streamlit as st

from halaman import komponen as kp
from modules.data_preprocessing import FREKUENSI
from modules.exploration import daftar_kosong, dekomposisi, statistik_deskriptif
from modules.visualization import grafik_seri, grafik_x

seri = kp.perlu_seri()
seri = seri.loc[:seri.last_valid_index()]  # tanpa baris masa depan
target = st.session_state[kp.TARGET]

st.title("2. Eksplorasi Data")
st.plotly_chart(grafik_seri(seri, f"{target} ({st.session_state[kp.FREKUENSI].lower()})", target))

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Statistik deskriptif")
    stat = statistik_deskriptif(seri)
    tampil = {k: (kp.fmt_angka(v) if isinstance(v, float) else v) for k, v in stat.items()}
    st.dataframe({"Ukuran": list(tampil), "Nilai": [str(v) for v in tampil.values()]}, hide_index=True)
with c2:
    st.subheader("Nilai kosong")
    kosong = daftar_kosong(seri)
    if kosong:
        st.warning(f"{len(kosong)} periode kosong: {', '.join(kosong)}. "
                   "Periode ini tidak diisi; model memperlakukannya sebagai data hilang.")
    else:
        st.success("Tidak ada periode kosong.")

X_dasar = st.session_state.get(kp.X_DASAR)
if X_dasar is not None:
    st.subheader("Variabel independen (X)")
    st.caption("Termasuk nilai X periode ke depan bila diisi di file.")
    st.plotly_chart(grafik_x(X_dasar, "Variabel X"))

st.subheader("Dekomposisi (opsional)")
st.caption("Memisahkan data menjadi tren, pola musiman, dan sisa. Hanya untuk informasi; tidak memengaruhi model.")
with st.expander("Tampilkan dekomposisi"):
    s_bawaan = FREKUENSI[st.session_state[kp.FREKUENSI]][1]
    c1, c2 = st.columns(2)
    s = c1.number_input("Panjang musim", 2, 365, max(2, s_bawaan))
    jenis = c2.radio("Jenis", ["additive", "multiplicative"], horizontal=True,
                     format_func=lambda x: "Aditif" if x == "additive" else "Multiplikatif")
    try:
        d = dekomposisi(seri, int(s), jenis)
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=list(d.columns))
        for i, kol in enumerate(d.columns, start=1):
            fig.add_trace(go.Scatter(x=d.index, y=d[kol], name=kol, showlegend=False), row=i, col=1)
        fig.update_layout(height=650, template="plotly_white", margin=dict(t=40, b=10))
        st.plotly_chart(fig)
    except ValueError as e:
        st.info(str(e))
