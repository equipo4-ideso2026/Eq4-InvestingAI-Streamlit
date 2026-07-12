"""
Página Clasificador SVC — equivalente a /api/svc/{ticker} + Módulo MSVC.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from db import get_db, validar_ticker, TICKERS, EMPRESAS, COL_PREDICCIONES, COL_METRICAS

st.set_page_config(page_title="Clasificador SVC — InvestAI", page_icon="🎯", layout="wide")

if not st.session_state.get("logueado"):
    st.warning("Inicia sesión desde la página principal.")
    st.stop()

st.title("🎯 Clasificador SVC — Support Vector Machine")
st.caption("Clasificación binaria BUY / SELL · GridSearchCV · scikit-learn · Datos reales vía MongoDB")

db = get_db()

col_a, col_b = st.columns([2, 1])
with col_a:
    ticker = st.selectbox("Ticker", TICKERS, format_func=lambda t: f"{t} — {EMPRESAS[t]}", key="msvc_ticker")
with col_b:
    dias = st.slider("Días de histórico a mostrar", 30, 365, 180, key="msvc_dias")

t = validar_ticker(ticker)

prediccion = db[COL_PREDICCIONES].find_one({"ticker": t, "modelo": "SVC"}, {"_id": 0})
metricas = db[COL_METRICAS].find_one({"ticker": t, "modelo": "SVC"}, {"_id": 0})

if not prediccion:
    st.error(f"No hay predicción SVC para {t}. Ejecuta el Notebook 2 (SVC) para este ticker.")
    st.stop()

m = metricas or {}

# ── Semáforo de señal + métricas principales ─────────────────────────────
col1, col2 = st.columns([1, 2])
with col1:
    senal = prediccion.get("senal", "HOLD")
    color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(senal, "⚪")
    st.markdown(f"## {color} {senal}")
    conf = prediccion.get("confianza")
    st.caption(f"Confianza: {conf*100:.1f}%" if conf is not None else "Confianza: N/D")
    st.caption(f"Features: {prediccion.get('tipo_features', 'precio')}")

with col2:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m.get('accuracy', 0)*100:.1f}%" if m.get("accuracy") is not None else "N/D")
    c2.metric("F1-Score", f"{m.get('f1', 0):.3f}" if m.get("f1") is not None else "N/D")
    c3.metric("Precision", f"{m.get('precision', 0):.3f}" if m.get("precision") is not None else "N/D")
    c4.metric("Recall", f"{m.get('recall', 0):.3f}" if m.get("recall") is not None else "N/D")

st.markdown("---")

# ── Gráfico de señales históricas ─────────────────────────────────────────
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
    fig.update_layout(title=f"{t} — Señales SVC (últimos {len(df_hist)} días)", template="plotly_dark",
                       height=420, hovermode="x unified", margin=dict(l=40, r=40, t=50, b=40))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sin histórico de señales disponible para este ticker.")

# ── Matriz de confusión + hiperparámetros ────────────────────────────────
col3, col4 = st.columns(2)
with col3:
    st.subheader("Matriz de Confusión (2×2)")
    cm = m.get("matriz_confusion")
    if cm and len(cm) == 2:
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=["Pred SELL", "Pred BUY"], y=["Real SELL", "Real BUY"],
            colorscale="Blues", text=cm, texttemplate="%{text}", showscale=False
        ))
        fig_cm.update_layout(template="plotly_dark", height=320, margin=dict(l=40, r=40, t=20, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("Sin matriz de confusión disponible.")

with col4:
    st.subheader("Hiperparámetros Óptimos (GridSearchCV)")
    st.table(pd.DataFrame({
        "Parámetro": ["Kernel", "C", "Gamma", "CV Folds", "Test size", "Scoring"],
        "Valor": [
            m.get("mejor_kernel", "—"), m.get("mejor_C", "—"), m.get("mejor_gamma", "—"),
            "5 (fijo)", "20% (fijo)", "F1"
        ]
    }).set_index("Parámetro"))

st.caption(f"Datos reales de Yahoo Finance · Clasificador SVC con scikit-learn · Actualizado: {prediccion.get('fecha_prediccion', '—')}")
