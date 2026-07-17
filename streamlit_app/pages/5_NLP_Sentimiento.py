"""
Página NLP Sentimiento — equivalente a /api/noticias + Módulo M5.
"""

import streamlit as st
import plotly.graph_objects as go
from db import get_db, COL_NOTICIAS

st.set_page_config(page_title="NLP Sentimiento — InvestAI", page_icon="📰", layout="wide")

if not st.session_state.get("logueado"):
    st.warning("Inicia sesión desde la página principal.")
    st.stop()

st.title("📰 Análisis de Sentimiento NLP")
st.caption("Noticias reales (yfinance) analizadas con VADER (NLTK)")

db = get_db()

col_a, col_b, col_c = st.columns(3)
with col_a:
    fuente = st.selectbox("Fuente", ["Todas", "Bloomberg", "CNBC", "Reuters", "MarketWatch", "Yahoo Finance"])
with col_b:
    sentimiento = st.selectbox("Sentimiento", ["Todos", "BULLISH", "BEARISH", "NEUTRAL"])
with col_c:
    ticker_filtro = st.selectbox("Ticker", ["Todos", "FSM", "VOLCABC1.LM", "ABX.TO", "BVN", "BHP"])

filtro = {}
if fuente != "Todas":
    filtro["fuente"] = fuente
if sentimiento != "Todos":
    filtro["sentimiento"] = sentimiento
if ticker_filtro != "Todos":
    filtro["ticker"] = ticker_filtro

noticias = list(db[COL_NOTICIAS].find(filtro, {"_id": 0}).sort("fecha_publicacion", -1).limit(50))

if not noticias:
    st.info("📭 No hay noticias que coincidan con este filtro. Ejecuta el Notebook 8 (NLP) si la colección está vacía.")
    st.stop()

compound_prom = sum(n.get("compound", 0) for n in noticias) / len(noticias)
sentimiento_consolidado = "BULLISH" if compound_prom > 0.05 else ("BEARISH" if compound_prom < -0.05 else "NEUTRAL")

col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("#### Sentimiento consolidado")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=compound_prom,
        number={"valueformat": "+.3f"},
        gauge={
            "axis": {"range": [-1, 1]},
            "bar": {"color": "white"},
            "steps": [
                {"range": [-1, -0.05], "color": "#EF5350"},
                {"range": [-0.05, 0.05], "color": "#FFC107"},
                {"range": [0.05, 1], "color": "#26A69A"},
            ],
        }
    ))
    fig.update_layout(template="plotly_dark", height=260, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Etiqueta consolidada", sentimiento_consolidado)
    st.caption(f"{len(noticias)} noticias en este filtro")

with col2:
    st.markdown("#### Feed de noticias")
    for n in noticias:
        icono = {"BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "➡️"}.get(n.get("sentimiento"), "•")
        with st.container(border=True):
            c1, c2 = st.columns([0.08, 0.92])
            c1.markdown(f"### {icono}")
            with c2:
                st.markdown(f"**{n.get('titulo', 'Sin título')}**")
                st.caption(n.get("texto", ""))
                origen = n.get("origen", "real")
                badge_origen = "🟢 real" if origen == "real" else "🟡 simulado (respaldo)"
                st.caption(f"{n.get('fuente', '—')} · {n.get('ticker', '—')} · {n.get('fecha_publicacion', '—')} · {badge_origen}")

st.caption("Datos reales de noticias (yfinance) · Análisis de sentimiento VADER (NLTK)")
