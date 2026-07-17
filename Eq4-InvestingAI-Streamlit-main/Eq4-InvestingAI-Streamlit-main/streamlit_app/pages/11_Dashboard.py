"""
Página Dashboard — vista agregada de todos los modelos, equivalente al
Módulo M9. Combina mercado + 5 clasificadores + regresor + NLP + backtest
en una sola pantalla, con voto mayoritario calculado en Python.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from db import get_db, validar_ticker, TICKERS, EMPRESAS, MODELOS_TODOS, PERFILES_RIESGO, \
    COL_PRECIOS, COL_PREDICCIONES, COL_METRICAS, COL_NOTICIAS, COL_BACKTESTS

st.set_page_config(page_title="Dashboard — InvestAI", page_icon="🏆", layout="wide")

if not st.session_state.get("logueado"):
    st.warning("Inicia sesión desde la página principal.")
    st.stop()

st.title("🏆 Dashboard Central")
st.caption("Vista agregada de todos los módulos · Datos reales vía MongoDB Atlas")

db = get_db()

col_a, col_b, col_c = st.columns(3)
with col_a:
    ticker = st.selectbox("Ticker", TICKERS, format_func=lambda t: f"{t} — {EMPRESAS[t]}",
                           index=TICKERS.index(st.session_state.get("ticker_global", "FSM")), key="m9_ticker")
    st.session_state.ticker_global = ticker
with col_b:
    modelo_bt = st.selectbox("Modelo (backtest)", MODELOS_TODOS, index=1, key="m9_modelo")
with col_c:
    perfil_bt = st.selectbox("Perfil (backtest)", PERFILES_RIESGO, index=1, key="m9_perfil")

t = validar_ticker(ticker)

# ── Precio actual ──────────────────────────────────────────────────────
ultimo = db[COL_PRECIOS].find_one({"ticker": t}, sort=[("fecha", -1)])
precio_actual = ultimo["close"] if ultimo else None

c1, c2 = st.columns(2)
c1.metric("Precio Actual", f"${precio_actual:.2f}" if precio_actual else "N/D")
c2.metric("Última fecha de precio", ultimo["fecha"] if ultimo else "—")

st.markdown("---")

# ── Señales de los 5 clasificadores + voto mayoritario ───────────────────
señales = []
conteo_buy, conteo_sell = 0, 0
for clf in MODELOS_TODOS:
    pred = db[COL_PREDICCIONES].find_one({"ticker": t, "modelo": clf}, {"_id": 0})
    met = db[COL_METRICAS].find_one({"ticker": t, "modelo": clf}, {"_id": 0})
    if pred:
        senal = pred.get("senal")
        if senal == "BUY":
            conteo_buy += 1
        elif senal == "SELL":
            conteo_sell += 1
        señales.append({"Modelo": clf, "Señal": senal or "—",
                         "Confianza": f"{pred.get('confianza', 0)*100:.0f}%" if pred.get("confianza") is not None else "—",
                         "Accuracy": f"{met.get('accuracy', 0)*100:.1f}%" if met and met.get("accuracy") is not None else "—",
                         "F1": f"{met.get('f1', 0):.2f}" if met and met.get("f1") is not None else "—"})
    else:
        señales.append({"Modelo": clf, "Señal": "Sin datos", "Confianza": "—", "Accuracy": "—", "F1": "—"})

if conteo_buy > conteo_sell:
    señal_ensamblada = "BUY 🟢"
elif conteo_sell > conteo_buy:
    señal_ensamblada = "SELL 🔴"
else:
    señal_ensamblada = "HOLD 🟡"

col_izq, col_der = st.columns([1.5, 1])
with col_izq:
    st.markdown("#### Consola de Señales IA (5 clasificadores)")
    st.dataframe(pd.DataFrame(señales), use_container_width=True, hide_index=True)

with col_der:
    st.markdown("#### Señal Ensamblada (voto mayoritario)")
    st.markdown(f"## {señal_ensamblada}")
    st.caption(f"BUY: {conteo_buy} · SELL: {conteo_sell} · HOLD: {len(MODELOS_TODOS) - conteo_buy - conteo_sell}")

# ── Regresor + NLP ────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Regresor LSTM")
    met_reg = db[COL_METRICAS].find_one({"ticker": t, "modelo": "LSTM_REG"}, {"_id": 0})
    if met_reg:
        st.metric("RMSE", f"${met_reg.get('rmse_usd', 0):.2f}")
        st.metric("R²", f"{met_reg.get('r2', 0):.3f}")
    else:
        st.info("Sin datos del regresor para este ticker.")

with col2:
    st.markdown("#### Sentimiento NLP")
    noticias_t = list(db[COL_NOTICIAS].find({"ticker": t}, {"compound": 1, "_id": 0}))
    if noticias_t:
        compound_prom = sum(n["compound"] for n in noticias_t) / len(noticias_t)
        sent = "BULLISH" if compound_prom > 0.05 else ("BEARISH" if compound_prom < -0.05 else "NEUTRAL")
        st.metric("Sentimiento consolidado", sent)
        st.metric("Compound promedio", f"{compound_prom:+.3f}")
    else:
        st.info("Sin noticias para este ticker.")

# ── Curva de equity del backtest elegido ─────────────────────────────────
st.markdown("---")
st.markdown(f"#### Curva de Equity — {modelo_bt} / {perfil_bt}")
doc_bt = db[COL_BACKTESTS].find_one({"modelo": modelo_bt, "perfil_riesgo": perfil_bt}, {"_id": 0})
if doc_bt and doc_bt.get("equity_curve"):
    df_eq = pd.DataFrame(doc_bt["equity_curve"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_eq["fecha"], y=df_eq["valor"], mode="lines", fill="tozeroy", line=dict(color="#2563eb")))
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=40, r=40, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(f"No hay backtest para {modelo_bt}/{perfil_bt}. Ejecuta el Notebook 10.")

st.caption(f"Vista agregada de todos los módulos · Datos reales vía MongoDB Atlas · Actualizado: {ultimo['fecha'] if ultimo else '—'}")
