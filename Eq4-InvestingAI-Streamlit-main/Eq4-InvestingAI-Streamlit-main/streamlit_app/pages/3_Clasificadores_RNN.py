"""
Página Clasificadores RNN — equivalente a /api/rnns/{ticker}?modelo=X + Módulo M3.
Selector de arquitectura: LSTM / BiLSTM / GRU / SimpleRNN.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from db import get_db, validar_ticker, TICKERS, EMPRESAS, MODELOS_RNN, COL_PREDICCIONES, COL_METRICAS

st.set_page_config(page_title="Clasificadores RNN — InvestAI", page_icon="🧠", layout="wide")

if not st.session_state.get("logueado"):
    st.warning("Inicia sesión desde la página principal.")
    st.stop()

st.title("🧠 Clasificadores RNN")
st.caption("LSTM · BiLSTM · GRU · SimpleRNN — Modelos entrenados con TensorFlow/Keras · Datos reales vía MongoDB")

db = get_db()

col_a, col_b, col_c = st.columns([2, 1.3, 1])
with col_a:
    ticker = st.selectbox("Ticker", TICKERS, format_func=lambda t: f"{t} — {EMPRESAS[t]}", key="m3_ticker")
with col_b:
    arch = st.selectbox("Arquitectura", MODELOS_RNN, key="m3_arch")
with col_c:
    dias = st.slider("Días a mostrar", 30, 365, 180, key="m3_dias")

t = validar_ticker(ticker)

prediccion = db[COL_PREDICCIONES].find_one({"ticker": t, "modelo": arch}, {"_id": 0})
metricas = db[COL_METRICAS].find_one({"ticker": t, "modelo": arch}, {"_id": 0})

if not prediccion:
    st.error(f"No hay predicción {arch} para {t}. Ejecuta el notebook correspondiente para este ticker.")
    st.stop()

m = metricas or {}

col1, col2 = st.columns([1, 2])
with col1:
    senal = prediccion.get("senal", "HOLD")
    color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(senal, "⚪")
    st.markdown(f"## {color} {senal}")
    conf = prediccion.get("confianza")
    st.caption(f"Confianza: {conf*100:.1f}%" if conf is not None else "Confianza: N/D")
    st.caption(f"Arquitectura: {arch} · N_STEPS: {prediccion.get('n_steps', '—')}")

with col2:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m.get('accuracy', 0)*100:.1f}%" if m.get("accuracy") is not None else "N/D")
    c2.metric("F1-Score", f"{m.get('f1', 0):.3f}" if m.get("f1") is not None else "N/D")
    c3.metric("Precision", f"{m.get('precision', 0):.3f}" if m.get("precision") is not None else "N/D")
    c4.metric("Recall", f"{m.get('recall', 0):.3f}" if m.get("recall") is not None else "N/D")

st.markdown("---")

hist = prediccion.get("historico_senales", [])
if hist:
    df_hist = pd.DataFrame(hist).tail(dias)
    df_hist["fecha"] = pd.to_datetime(df_hist["fecha"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist["fecha"], y=df_hist["precio"], mode="lines", name="Precio", line=dict(color="#42A5F5")))
    buys = df_hist[df_hist["prediccion"] == "BUY"]
    sells = df_hist[df_hist["prediccion"] == "SELL"]
    fig.add_trace(go.Scatter(x=buys["fecha"], y=buys["precio"], mode="markers", name="BUY",
                              marker=dict(symbol="triangle-up", size=10, color="#26A69A")))
    fig.add_trace(go.Scatter(x=sells["fecha"], y=sells["precio"], mode="markers", name="SELL",
                              marker=dict(symbol="triangle-down", size=10, color="#EF5350")))
    fig.update_layout(title=f"{t} — Señales {arch} (últimos {len(df_hist)} días)", template="plotly_dark",
                       height=420, hovermode="x unified", margin=dict(l=40, r=40, t=50, b=40))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sin histórico de señales disponible para este ticker/modelo.")

col3, col4 = st.columns(2)
with col3:
    st.subheader("Matriz de Confusión 2×2")
    cm = m.get("matriz_confusion")
    if cm and len(cm) == 2:
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=["Pred SELL", "Pred BUY"], y=["Real SELL", "Real BUY"],
            colorscale="Blues", text=cm, texttemplate="%{text}", showscale=False
        ))
        fig_cm.update_layout(template="plotly_dark", height=300, margin=dict(l=40, r=40, t=20, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("Sin matriz de confusión disponible.")

with col4:
    st.subheader(f"Historial de Entrenamiento — {arch}")
    hist_epocas = m.get("historial_epocas")
    if hist_epocas and hist_epocas.get("loss"):
        epocas = list(range(1, len(hist_epocas["loss"]) + 1))
        fig_h = make_subplots(specs=[[{"secondary_y": True}]])
        fig_h.add_trace(go.Scatter(x=epocas, y=hist_epocas["loss"], name="Loss (train)", line=dict(color="#EF5350")), secondary_y=False)
        fig_h.add_trace(go.Scatter(x=epocas, y=hist_epocas["accuracy"], name="Accuracy (train)", line=dict(color="#26A69A")), secondary_y=True)
        if hist_epocas.get("val_loss"):
            fig_h.add_trace(go.Scatter(x=epocas, y=hist_epocas["val_loss"], name="Loss (val)", line=dict(color="#EF5350", dash="dot")), secondary_y=False)
        if hist_epocas.get("val_accuracy"):
            fig_h.add_trace(go.Scatter(x=epocas, y=hist_epocas["val_accuracy"], name="Accuracy (val)", line=dict(color="#26A69A", dash="dot")), secondary_y=True)
        fig_h.update_layout(template="plotly_dark", height=300, margin=dict(l=40, r=40, t=20, b=20),
                             legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("Sin historial de épocas disponible.")

st.caption(f"Datos reales de Yahoo Finance · Modelos entrenados con TensorFlow/Keras · Actualizado: {prediccion.get('fecha_prediccion', '—')}")
