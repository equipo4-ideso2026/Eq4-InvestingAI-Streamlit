"""
Página Regresor LSTM — equivalente a /api/lstm/{ticker}?horizonte=N + Módulo M4.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from db import get_db, validar_ticker, TICKERS, EMPRESAS, COL_PREDICCIONES, COL_METRICAS

st.set_page_config(page_title="Regresor LSTM — InvestAI", page_icon="🔮", layout="wide")

if not st.session_state.get("logueado"):
    st.warning("Inicia sesión desde la página principal.")
    st.stop()

st.title("🔮 LSTM Regressor — Pronóstico de Precios")
st.caption("Arquitectura: LSTM(64)→Dropout(0.2)→LSTM(32)→Dropout(0.2)→Dense(16,relu)→Dense(1,linear)")

db = get_db()

col_a, col_b = st.columns([2, 2])
with col_a:
    ticker = st.selectbox("Ticker", TICKERS, format_func=lambda t: f"{t} — {EMPRESAS[t]}", key="m4_ticker")
with col_b:
    horizonte = st.select_slider("Días de predicción", options=[7, 14, 30, 60], value=14, key="m4_horizonte")

t = validar_ticker(ticker)

prediccion = db[COL_PREDICCIONES].find_one({"ticker": t, "modelo": "LSTM_REG"}, {"_id": 0})
metricas = db[COL_METRICAS].find_one({"ticker": t, "modelo": "LSTM_REG"}, {"_id": 0})

if not prediccion:
    st.error(f"No hay pronóstico LSTM_REG para {t}. Ejecuta el Notebook 7 (Regresor LSTM) para este ticker.")
    st.stop()

m = metricas or {}

col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("#### Parámetros del Modelo")
    st.table(pd.DataFrame({
        "Parámetro": ["Épocas entrenadas", "Dropout", "Batch Size", "Ventana", "Learning Rate", "Optimizador", "Loss"],
        "Valor": [m.get("epocas_entrenadas", "—"), "0.2", "32", "60 días", "0.001", "Adam", "MSE"]
    }).set_index("Parámetro"))

    st.markdown("#### Métricas de Precisión")
    precio_actual = prediccion.get("ultimo_precio")
    st.table(pd.DataFrame({
        "Métrica": ["Precio Actual", "RMSE (USD)", "RMSE (%)", "MAE", "R²"],
        "Valor": [
            f"${precio_actual:.2f}" if precio_actual is not None else "N/D",
            f"${m.get('rmse_usd'):.4f}" if m.get("rmse_usd") is not None else "N/D",
            f"{m.get('rmse_pct'):.2f}%" if m.get("rmse_pct") is not None else "N/D",
            f"${m.get('mae_usd'):.4f}" if m.get("mae_usd") is not None else "N/D",
            f"{m.get('r2'):.4f}" if m.get("r2") is not None else "N/D",
        ]
    }).set_index("Métrica"))

with col1:
    hist_pred = prediccion.get("historico_predicciones", [])
    proyecciones = prediccion.get("proyecciones_horizonte", {})
    clave_h = f"{horizonte}d"
    datos_h = proyecciones.get(clave_h)

    fig = go.Figure()

    if hist_pred:
        df_hist = pd.DataFrame(hist_pred)
        df_hist["fecha"] = pd.to_datetime(df_hist["fecha"])
        fig.add_trace(go.Scatter(x=df_hist["fecha"], y=df_hist["precio_real"], mode="lines", name="Precio Real", line=dict(color="#42A5F5", width=2)))
        fig.add_trace(go.Scatter(x=df_hist["fecha"], y=df_hist["precio_predicho"], mode="lines", name="Predicción LSTM (test)", line=dict(color="#FFA726", dash="dot")))
        ultima_fecha = df_hist["fecha"].iloc[-1]
    else:
        ultima_fecha = pd.Timestamp.now()

    if datos_h:
        fechas_futuras = pd.date_range(start=ultima_fecha, periods=horizonte + 1, freq="D")[1:]
        precio_proy = [datos_h["precio_estimado"]] * horizonte  # línea recta hasta el horizonte (ver nota)
        fig.add_trace(go.Scatter(
            x=[ultima_fecha, fechas_futuras[-1]], y=[precio_actual, datos_h["precio_estimado"]],
            mode="lines", name=f"Predicción LSTM ({horizonte}d)", line=dict(color="#EF5350", dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=[ultima_fecha, fechas_futuras[-1], fechas_futuras[-1], ultima_fecha],
            y=[precio_actual, datos_h["banda_superior"], datos_h["banda_inferior"], precio_actual],
            fill="toself", fillcolor="rgba(239,83,80,0.15)", line=dict(color="rgba(0,0,0,0)"),
            name="Intervalo de confianza", showlegend=True
        ))

    fig.add_vline(x=ultima_fecha, line_dash="dot", line_color="gray", annotation_text="Hoy")
    fig.update_layout(title=f"{t} — Pronóstico LSTM ({horizonte} días)", template="plotly_dark",
                       height=520, hovermode="x unified", margin=dict(l=40, r=40, t=50, b=40))
    st.plotly_chart(fig, use_container_width=True)

    if datos_h:
        st.info(
            f"**Proyección a {horizonte} días:** \\${datos_h['precio_estimado']:.2f} "
            f"(banda: \\${datos_h['banda_inferior']:.2f} – \\${datos_h['banda_superior']:.2f})"
        )
    else:
        st.warning(f"No hay proyección calculada para el horizonte de {horizonte} días.")

st.caption(f"Datos reales de Yahoo Finance · Regresor LSTM con TensorFlow/Keras · Actualizado: {prediccion.get('fecha_prediccion', '—')}")
