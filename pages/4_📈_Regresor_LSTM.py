"""
pages/4_📈_Regresor_LSTM.py — LSTM Regressor: Pronóstico de Precios
======================================================================
Reemplaza el módulo M4 del index.html original: pronóstico de precio
continuo en USD (regresión, no clasificación), con curva histórica real
vs. predicha en test, y proyección futura a 7/14/30/60 días con banda de
confianza sombreada.

Lee el ticker desde `st.session_state.ticker_global` y consulta `db.py`
directamente contra MongoDB Atlas (modelo='LSTM_REG', poblado por el
Notebook 7).
"""

from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from db import get_lstm_regresor

# ────────────────────────────────────────────────────────────────
# Configuración de página
# ────────────────────────────────────────────────────────────────
try:
    st.set_page_config(page_title="Regresor LSTM · InvestAI", page_icon="📈", layout="wide")
except Exception:
    pass

# ────────────────────────────────────────────────────────────────
# Paleta oscura fiel al index.html original
# ────────────────────────────────────────────────────────────────
COLOR_BG = "#0f172a"
COLOR_CARD = "#1e293b"
COLOR_TEXT = "#f8fafc"
COLOR_MUTED = "#94a3b8"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {COLOR_BG}; color: {COLOR_TEXT}; }}
    .modulo-header p {{ color: {COLOR_MUTED}; font-size: 13px; margin-top: -8px; }}
    .card-title {{ font-size: 14px; font-weight: 600; color: {COLOR_TEXT}; margin-bottom: 12px; }}
    .params-table {{ width:100%; border-collapse: collapse; font-size:13px; }}
    .params-table tr {{ border-bottom: 1px dashed rgba(255,255,255,0.1); }}
    .params-table td {{ padding: 8px 4px; color:{COLOR_MUTED}; }}
    .params-table td:first-child {{ font-weight:600; color:{COLOR_TEXT}; width:55%; }}
    .side-card {{
        background-color: {COLOR_CARD};
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }}
    .footer-nota {{
        margin-top:24px; padding-top:16px; border-top:1px solid rgba(255,255,255,0.08);
        font-size:12px; color:{COLOR_MUTED}; text-align:center;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────
# Sincronización con el ticker global del sidebar
# ────────────────────────────────────────────────────────────────
ticker = st.session_state.get("ticker_global", "FSM")

st.markdown(
    """
    <div class="modulo-header">
        <h2>🧠 LSTM Regressor — Pronóstico de Precios</h2>
        <p>Arquitectura: LSTM(64)→Dropout(0.2)→LSTM(32)→Dropout(0.2)→Dense(16,relu)→Dense(1,linear)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────
# Control local: días de predicción a futuro (slider tipo HTML original)
# ────────────────────────────────────────────────────────────────
dias = st.select_slider("Días de predicción", options=[7, 14, 30, 60], value=14)

# ────────────────────────────────────────────────────────────────
# Datos
# ────────────────────────────────────────────────────────────────
datos = get_lstm_regresor(ticker)

if not datos:
    st.warning(
        f"⚠ No hay pronóstico LSTM disponible para **{ticker}**. "
        "Verifica que el Notebook 7 (Regresor LSTM) haya procesado este ticker."
    )
    st.stop()

m = datos.get("metricas", {})
precio_actual = datos.get("precio_actual_usd")

# ────────────────────────────────────────────────────────────────
# Layout: gráfico central (ancho) + paneles laterales de métricas
# ────────────────────────────────────────────────────────────────
col_chart, col_side = st.columns([3, 1])

with col_chart:
    historico = datos.get("historico_precios", [])
    fechas_hist = [d["fecha"] for d in historico]
    precios_real = [d["precio_real"] for d in historico]
    precios_pred = [d["precio_predicho"] for d in historico]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fechas_hist, y=precios_real, mode="lines", name="Precio Real",
            line=dict(color="#1976D2", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fechas_hist, y=precios_pred, mode="lines", name="Predicción (test)",
            line=dict(color="#FF9800", width=1.5, dash="dot"),
        )
    )

    # ── Proyección futura con banda de confianza sombreada ──────────
    hoy_fecha_str = fechas_hist[-1] if fechas_hist else datetime.now().strftime("%Y-%m-%d")
    try:
        ultima_fecha_dt = datetime.strptime(hoy_fecha_str[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        ultima_fecha_dt = datetime.now()

    prediccion_futura = datos.get("prediccion_futura", {})
    p_horizonte = prediccion_futura.get(f"{dias}_dias")

    if p_horizonte and precio_actual is not None:
        fechas_fut = [(ultima_fecha_dt + timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(dias)]

        precio_fin = p_horizonte.get("precio")
        banda_sup = p_horizonte.get("banda_sup")
        banda_inf = p_horizonte.get("banda_inf")

        linea_fut = np.linspace(precio_actual, precio_fin, dias) if precio_fin is not None else []
        linea_sup = np.linspace(precio_actual, banda_sup, dias) if banda_sup is not None else []
        linea_inf = np.linspace(precio_actual, banda_inf, dias) if banda_inf is not None else []

        if len(linea_fut) > 0:
            if len(linea_sup) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=fechas_fut, y=linea_sup, mode="lines", name="Banda superior",
                        line=dict(color="rgba(239,83,80,0)", width=0), showlegend=False,
                    )
                )
            if len(linea_inf) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=fechas_fut, y=linea_inf, mode="lines", name="Intervalo confianza",
                        fill="tonexty", fillcolor="rgba(239,83,80,0.15)",
                        line=dict(color="rgba(239,83,80,0)", width=0),
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=fechas_fut, y=linea_fut, mode="lines", name=f"Predicción LSTM ({dias}d)",
                    line=dict(color="#EF5350", width=2.5, dash="dash"),
                )
            )

    fig.update_layout(
        title=dict(text=f"{ticker} — Pronóstico LSTM ({dias} días)", font=dict(size=18)),
        template="plotly_dark",
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT),
        xaxis=dict(title="Fecha", gridcolor="#1e293b"),
        yaxis=dict(title="Precio (USD)", gridcolor="#1e293b"),
        hovermode="x unified",
        margin=dict(l=60, r=60, t=80, b=60),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        height=500,
        shapes=[
            dict(
                type="line", x0=hoy_fecha_str, x1=hoy_fecha_str, y0=0, y1=1, yref="paper",
                line=dict(color="#9E9E9E", width=1, dash="dot"),
            )
        ],
        annotations=[
            dict(
                x=hoy_fecha_str, y=1.05, yref="paper", text="Hoy",
                showarrow=False, bgcolor="#9E9E9E", font=dict(color="white"),
            )
        ],
    )
    st.plotly_chart(fig, use_container_width=True)

with col_side:
    def _fmt_usd(v):
        return "N/D" if v is None else f"${v:,.4f}"

    def _fmt_pct(v):
        return "N/D" if v is None else f"{v:.2f}%"

    def _fmt_num(v):
        return "N/D" if v is None else f"{v:.4f}"

    st.markdown(
        f"""
        <div class="side-card">
            <div class="card-title">Parámetros del Modelo</div>
            <table class="params-table">
                <tr><td>Épocas entrenadas</td><td>{m.get('epocas_entrenadas', 'N/D')}</td></tr>
                <tr><td>Dropout</td><td>0.2</td></tr>
                <tr><td>Batch Size</td><td>32</td></tr>
                <tr><td>Ventana</td><td>{datos.get('n_steps', 60)} días</td></tr>
                <tr><td>Learning Rate</td><td>0.001</td></tr>
                <tr><td>Arquitectura</td><td>LSTM(64→32)→Dense(16)</td></tr>
                <tr><td>Optimizador</td><td>Adam</td></tr>
                <tr><td>Loss</td><td>MSE</td></tr>
            </table>
        </div>
        <div class="side-card">
            <div class="card-title">Métricas de Precisión</div>
            <table class="params-table">
                <tr><td>Precio Actual</td><td>{_fmt_usd(precio_actual)}</td></tr>
                <tr><td>RMSE (USD)</td><td>{_fmt_usd(m.get('rmse_usd'))}</td></tr>
                <tr><td>RMSE (%)</td><td>{_fmt_pct(m.get('rmse_pct'))}</td></tr>
                <tr><td>MAE</td><td>{_fmt_usd(m.get('mae_usd'))}</td></tr>
                <tr><td>R²</td><td>{_fmt_num(m.get('r2'))}</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if m.get("rmse_arima_baseline") is not None:
        st.caption(f"Baseline ARIMA RMSE: ${m['rmse_arima_baseline']:.4f} (referencia de comparación)")

# ────────────────────────────────────────────────────────────────
# Pie de página
# ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="footer-nota">
        Datos reales de Yahoo Finance · Regresor LSTM con TensorFlow/Keras ·
        Actualizado: {datos.get('fecha_prediccion', '—')}
    </div>
    """,
    unsafe_allow_html=True,
)
