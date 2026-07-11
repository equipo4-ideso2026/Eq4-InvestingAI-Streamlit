"""
pages/1_📊_Mercado.py — Dashboard de Mercado
==============================================
Reemplaza el módulo M2 (Dashboard de Mercado) del index.html original:
gráfico candlestick + indicadores técnicos + volumen, con tarjetas de
métricas (precio actual, variación % del período, volumen).

Lee el ticker seleccionado desde `st.session_state.ticker_global`
(definido en el sidebar de la app principal) y consulta `db.py`
directamente contra MongoDB Atlas (sin pasar por FastAPI/ngrok).
"""

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from db import get_mercado_data

# ────────────────────────────────────────────────────────────────
# Configuración de página
# ────────────────────────────────────────────────────────────────
try:
    st.set_page_config(page_title="Mercado · InvestAI", page_icon="📊", layout="wide")
except Exception:
    pass  # ya fue configurado por el entrypoint principal

# ────────────────────────────────────────────────────────────────
# Paleta oscura fiel al index.html original
# ────────────────────────────────────────────────────────────────
COLOR_BG = "#0f172a"
COLOR_CARD = "#1e293b"
COLOR_TEXT = "#f8fafc"
COLOR_MUTED = "#94a3b8"
COLOR_POSITIVE = "#26A69A"
COLOR_NEGATIVE = "#EF5350"
COLOR_WARNING = "#FFC107"
COLOR_ACCENT = "#2563eb"
COLOR_ACCENT_LIGHT = "#38bdf8"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {COLOR_BG};
        color: {COLOR_TEXT};
    }}
    div[data-testid="stMetric"] {{
        background-color: {COLOR_CARD};
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }}
    div[data-testid="stMetricLabel"] {{
        color: {COLOR_MUTED};
    }}
    .modulo-header p {{
        color: {COLOR_MUTED};
        font-size: 13px;
        margin-top: -8px;
    }}
    .footer-nota {{
        margin-top: 24px;
        padding-top: 16px;
        border-top: 1px solid rgba(255,255,255,0.08);
        font-size: 12px;
        color: {COLOR_MUTED};
        text-align: center;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────
# Nombres de empresa (mismos 5 tickers del proyecto)
# ────────────────────────────────────────────────────────────────
TICKER_NOMBRES = {
    "FSM": "Fortuna Silver Mines",
    "VOLCABC1.LM": "Volcan Compañía Minera",
    "ABX.TO": "Barrick Gold",
    "BVN": "Compañía de Minas Buenaventura",
    "BHP": "BHP Group",
}

# ────────────────────────────────────────────────────────────────
# Sincronización con el ticker global del sidebar
# ────────────────────────────────────────────────────────────────
ticker = st.session_state.get("ticker_global", "FSM")

st.markdown(
    f"""
    <div class="modulo-header">
        <h2>📊 Dashboard de Mercado</h2>
        <p>Gráfico candlestick con indicadores técnicos en tiempo real</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────
# Controles locales: período + indicadores técnicos
# ────────────────────────────────────────────────────────────────
PERIODOS = {
    "1 Mes (30d)": 30,
    "3 Meses (90d)": 90,
    "6 Meses (180d)": 180,
    "1 Año (365d)": 365,
}

col_periodo, col_spacer = st.columns([1, 3])
with col_periodo:
    periodo_label = st.selectbox("Período", options=list(PERIODOS.keys()), index=3)
dias = PERIODOS[periodo_label]

c1, c2, c3, c4 = st.columns(4)
with c1:
    mostrar_sma20 = st.checkbox("SMA20", value=True)
with c2:
    mostrar_sma50 = st.checkbox("SMA50", value=True)
with c3:
    mostrar_ema12 = st.checkbox("EMA12", value=False)
with c4:
    mostrar_bollinger = st.checkbox("Banda Bollinger", value=False)

# ────────────────────────────────────────────────────────────────
# Datos
# ────────────────────────────────────────────────────────────────
df = get_mercado_data(ticker, dias=dias)

if df.empty:
    st.warning(
        f"⚠ No hay datos de mercado para **{ticker}**. "
        "Verifica que el Notebook 1 (Ingesta) haya poblado la colección `precios_ohlcv`."
    )
    st.stop()

df = df.sort_values("fecha").reset_index(drop=True)

# Bandas de Bollinger (20 periodos, 2 desviaciones estándar) — no vienen
# precalculadas en `precios_ohlcv`, se calculan aquí sobre el cierre.
if mostrar_bollinger:
    bb_media = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()
    df["bb_upper"] = bb_media + 2 * bb_std
    df["bb_lower"] = bb_media - 2 * bb_std

# ────────────────────────────────────────────────────────────────
# Layout principal: gráfico (col ancha) + tarjetas de métricas
# ────────────────────────────────────────────────────────────────
col_chart, col_metrics = st.columns([3, 1])

with col_chart:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )

    # ── Candlestick ──────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df["fecha"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color=COLOR_POSITIVE,
            decreasing_line_color=COLOR_NEGATIVE,
            name=ticker,
        ),
        row=1,
        col=1,
    )

    # ── Indicadores técnicos superpuestos ────────────────────────
    if mostrar_sma20 and "sma_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["fecha"], y=df["sma_20"], name="SMA20",
                line=dict(color=COLOR_WARNING, width=1.5),
            ),
            row=1, col=1,
        )
    if mostrar_sma50 and "sma_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["fecha"], y=df["sma_50"], name="SMA50",
                line=dict(color=COLOR_ACCENT_LIGHT, width=1.5),
            ),
            row=1, col=1,
        )
    if mostrar_ema12 and "ema_12" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["fecha"], y=df["ema_12"], name="EMA12",
                line=dict(color="#a78bfa", width=1.5),
            ),
            row=1, col=1,
        )
    if mostrar_bollinger:
        fig.add_trace(
            go.Scatter(
                x=df["fecha"], y=df["bb_upper"], name="Bollinger Sup.",
                line=dict(color=COLOR_MUTED, width=1, dash="dot"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["fecha"], y=df["bb_lower"], name="Bollinger Inf.",
                line=dict(color=COLOR_MUTED, width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(148, 163, 184, 0.08)",
            ),
            row=1, col=1,
        )

    # ── Volumen (verde alcista / rojo bajista) ───────────────────
    colores_volumen = [
        COLOR_POSITIVE if row["close"] >= row["open"] else COLOR_NEGATIVE
        for _, row in df.iterrows()
    ]
    fig.add_trace(
        go.Bar(
            x=df["fecha"], y=df["volume"], name="Volumen",
            marker_color=colores_volumen,
            showlegend=False,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT),
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Precio (USD)", row=1, col=1, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Volumen", row=2, col=1, gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")

    st.plotly_chart(fig, use_container_width=True)

with col_metrics:
    precio_actual = float(df["close"].iloc[-1])
    precio_inicial = float(df["close"].iloc[0])
    variacion_pct = (
        ((precio_actual - precio_inicial) / precio_inicial) * 100
        if precio_inicial not in (0, None)
        else 0.0
    )
    volumen_actual = float(df["volume"].iloc[-1])

    def _formatear_volumen(v: float) -> str:
        if v >= 1_000_000:
            return f"{v / 1_000_000:.1f}M"
        if v >= 1_000:
            return f"{v / 1_000:.1f}K"
        return f"{v:.0f}"

    st.metric("Precio Actual", f"${precio_actual:,.2f}", help=TICKER_NOMBRES.get(ticker, ticker))
    st.metric(
        f"Variación · {periodo_label}",
        f"{variacion_pct:+.2f}%",
        delta=f"{variacion_pct:+.2f}%",
    )
    st.metric("Volumen", _formatear_volumen(volumen_actual), help="Acciones negociadas (último día)")

# ────────────────────────────────────────────────────────────────
# Pie de página
# ────────────────────────────────────────────────────────────────
fecha_actualizacion = df["fecha"].iloc[-1]
fecha_txt = (
    fecha_actualizacion.strftime("%Y-%m-%d")
    if isinstance(fecha_actualizacion, (pd.Timestamp, datetime))
    else str(fecha_actualizacion)
)
st.markdown(
    f"""
    <div class="footer-nota">
        Datos reales de Yahoo Finance · Precios OHLCV vía MongoDB Atlas ·
        {TICKER_NOMBRES.get(ticker, ticker)} · Actualizado: {fecha_txt}
    </div>
    """,
    unsafe_allow_html=True,
)
