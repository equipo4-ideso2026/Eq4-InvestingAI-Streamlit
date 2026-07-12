"""
Página Mercado — equivalente a /api/mercado/{ticker} + Módulo M2 del HTML.
Candlestick real (precios_ohlcv) con indicadores técnicos.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from db import get_db, validar_ticker, TICKERS, EMPRESAS, COL_PRECIOS

st.set_page_config(page_title="Mercado — InvestAI", page_icon="📊", layout="wide")

if not st.session_state.get("logueado"):
    st.warning("Inicia sesión desde la página principal.")
    st.stop()

st.title("📊 Dashboard de Mercado")
st.caption("Gráfico candlestick con indicadores técnicos en tiempo real")

db = get_db()

col_a, col_b, col_c, col_d = st.columns([2, 1.4, 1, 1])
with col_a:
    ticker = st.selectbox(
        "Ticker", TICKERS, format_func=lambda t: f"{t} — {EMPRESAS[t]}",
        index=TICKERS.index(st.session_state.get("ticker_global", "FSM")),
        key="m2_ticker",
    )
    st.session_state.ticker_global = ticker
with col_b:
    periodo_label = st.selectbox("Período", ["30 Días", "90 Días (3 Meses)", "180 Días (6 Meses)", "1 Año (365 días)"], index=3)
    dias = {"30 Días": 30, "90 Días (3 Meses)": 90, "180 Días (6 Meses)": 180, "1 Año (365 días)": 365}[periodo_label]
with col_c:
    mostrar_sma = st.checkbox("SMA20/50", value=True)
with col_d:
    mostrar_bollinger = st.checkbox("Bollinger", value=False)

t = validar_ticker(ticker)

# ── Consulta directa a MongoDB (reemplaza /api/mercado/{ticker}) ────────
docs = list(db[COL_PRECIOS].find({"ticker": t}, {"_id": 0}).sort("fecha", 1))

if not docs:
    st.error(f"No hay datos de precios para {t}. Ejecuta el Notebook 1 (Ingesta).")
    st.stop()

df = pd.DataFrame(docs)
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").tail(dias).reset_index(drop=True)

# ── Métricas superiores ──────────────────────────────────────────────────
ultimo = df.iloc[-1]
primero = df.iloc[0]
cambio_pct = (ultimo["close"] - primero["close"]) / primero["close"] * 100
volumen_total = df["volume"].sum()

m1, m2, m3 = st.columns(3)
m1.metric("Precio Actual", f"${ultimo['close']:.2f}", f"{t}")
m2.metric("Variación del período", f"{cambio_pct:+.2f}%")
m3.metric("Volumen", f"{volumen_total/1_000_000:.1f}M", "Acciones negociadas")

# ── Gráfico candlestick ───────────────────────────────────────────────────
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["fecha"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    name="OHLCV"
))

if mostrar_sma:
    if "sma_20" in df.columns:
        fig.add_trace(go.Scatter(x=df["fecha"], y=df["sma_20"], line=dict(color="orange", width=1.5), name="SMA20"))
    if "sma_50" in df.columns:
        fig.add_trace(go.Scatter(x=df["fecha"], y=df["sma_50"], line=dict(color="purple", width=1.5), name="SMA50"))

if mostrar_bollinger:
    banda_media = df["close"].rolling(20).mean()
    banda_std = df["close"].rolling(20).std()
    fig.add_trace(go.Scatter(x=df["fecha"], y=banda_media + 2 * banda_std, line=dict(color="rgba(100,150,255,0.4)", width=1), name="Bollinger Sup"))
    fig.add_trace(go.Scatter(x=df["fecha"], y=banda_media - 2 * banda_std, line=dict(color="rgba(100,150,255,0.4)", width=1), name="Bollinger Inf", fill="tonexty"))

fig.update_layout(
    title=f"{t} — {dias} días", xaxis_rangeslider_visible=False,
    template="plotly_dark", height=520, margin=dict(l=40, r=40, t=50, b=40),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ── Volumen ────────────────────────────────────────────────────────────
fig_vol = go.Figure()
colores = ["#26A69A" if c >= o else "#EF5350" for c, o in zip(df["close"], df["open"])]
fig_vol.add_trace(go.Bar(x=df["fecha"], y=df["volume"], marker_color=colores, name="Volumen"))
fig_vol.update_layout(template="plotly_dark", height=180, margin=dict(l=40, r=40, t=10, b=30))
st.plotly_chart(fig_vol, use_container_width=True)

st.caption(f"Datos reales de Yahoo Finance · Precios OHLCV vía MongoDB Atlas · Actualizado: {ultimo['fecha'].strftime('%Y-%m-%d')}")
