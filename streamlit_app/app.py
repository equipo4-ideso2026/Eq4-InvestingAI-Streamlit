"""
app.py — Página de inicio de Ernesto Investing AI (versión Streamlit).

Sustituye a index.html + Notebook11 (FastAPI + ngrok): esta app se conecta
DIRECTO a MongoDB Atlas con pymongo, sin ningún servidor intermedio que
haya que mantener corriendo. Las 11 pestañas del HTML original se
convierten en 11 páginas dentro de la carpeta pages/ (multi-page app
nativa de Streamlit).
"""

import streamlit as st
from datetime import datetime
from db import get_db, TICKERS, EMPRESAS, COL_PRECIOS, COL_PREDICCIONES, COL_METRICAS, \
    COL_NOTICIAS, COL_ESTRATEGIAS, COL_BACKTESTS, COL_ORDENES

st.set_page_config(
    page_title="InvestAI — Ernesto Investing AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Login simple (mismo criterio "demo" que index.html) ─────────────────
if "logueado" not in st.session_state:
    st.session_state.logueado = False

if not st.session_state.logueado:
    st.title("📈 InvestAI — Ernesto Investing AI")
    st.caption("Sistema de Predicción Bursátil Inteligente (SPBI) · iDeSo · UNMSM")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.subheader("Iniciar sesión")
        email = st.text_input("Correo electrónico", value="demo@investai.com")
        password = st.text_input("Contraseña", value="demo123", type="password")
        if st.button("Ingresar", type="primary", use_container_width=True):
            if email == "demo@investai.com" and password == "demo123":
                st.session_state.logueado = True
                st.rerun()
            else:
                st.error("Credenciales incorrectas. Usa demo@investai.com / demo123.")
    st.stop()

# ── Ticker global compartido entre páginas (equivalente a tickerActual) ──
if "ticker_global" not in st.session_state:
    st.session_state.ticker_global = "FSM"

# ── Capital compartido entre Estrategias / Portafolio / Backtesting ──────
if "capital" not in st.session_state:
    st.session_state.capital = 100_000.0

with st.sidebar:
    st.markdown("## 📈 InvestAI")
    st.caption("Ernesto Investing AI · SPBI")
    st.markdown("---")

    ticker_sel = st.selectbox(
        "Ticker global (Mercado / Dashboard)",
        TICKERS,
        format_func=lambda t: f"{t} — {EMPRESAS[t]}",
        index=TICKERS.index(st.session_state.ticker_global),
        key="ticker_global_selector",
    )
    st.session_state.ticker_global = ticker_sel

    capital_sel = st.number_input(
        "💰 Capital (Estrategias / Portafolio / Backtesting)",
        min_value=1000.0, value=st.session_state.capital, step=1000.0,
        key="capital_selector",
        help="Compartido entre estas 3 páginas. Los montos en USD se reescalan "
             "proporcionalmente al backtest guardado (que se calculó con $100,000).",
    )
    st.session_state.capital = capital_sel

    st.markdown("---")
    if st.button("🔌 Verificar conexión a MongoDB"):
        db = get_db()
        st.success("✓ Conectado a MongoDB Atlas")

    if st.button("🚪 Cerrar sesión"):
        st.session_state.logueado = False
        st.rerun()

# ── Contenido de la home ─────────────────────────────────────────────────
st.title("📈 InvestAI — Panel Principal")
st.caption("Sistema de Predicción Bursátil Inteligente · Datos reales vía MongoDB Atlas")

db = get_db()

st.markdown("### Estado de las colecciones")
colecciones = {
    "Precios OHLCV (Notebook 1)": COL_PRECIOS,
    "Predicciones (Notebooks 2-6)": COL_PREDICCIONES,
    "Métricas de modelos (Notebooks 2-7)": COL_METRICAS,
    "Noticias / NLP (Notebook 8)": COL_NOTICIAS,
    "Estrategias / Markowitz (Notebook 9)": COL_ESTRATEGIAS,
    "Backtests (Notebook 10)": COL_BACKTESTS,
    "Órdenes manuales (Broker)": COL_ORDENES,
}

cols = st.columns(4)
for i, (label, coll_name) in enumerate(colecciones.items()):
    count = db[coll_name].count_documents({})
    with cols[i % 4]:
        st.metric(label, f"{count:,} docs")

st.markdown("---")
st.markdown("""
### 🧭 Navegación

Usa el menú de la izquierda para moverte entre los módulos:

| Módulo | Contenido |
|---|---|
| **Mercado** | Gráfico candlestick + indicadores técnicos reales |
| **Clasificador SVC** | Señal BUY/SELL con Support Vector Machine |
| **Clasificadores RNN** | LSTM / BiLSTM / GRU / SimpleRNN, seleccionables |
| **Regresor LSTM** | Pronóstico de precio futuro con bandas de confianza |
| **NLP Sentimiento** | Noticias reales + análisis VADER (BULLISH/BEARISH/NEUTRAL) |
| **Estrategias** | Optimización de portafolio (Markowitz) |
| **Portafolio** | Resultado del backtest diversificado (P&L, posiciones) |
| **Backtesting** | Reporte técnico completo (Sharpe, Drawdown, trades) |
| **Broker** | Libro de órdenes manual (paper trading) |
| **Dashboard** | Vista agregada de todos los modelos |
| **Modelos** | Consola de ejecución/consulta de los 5 clasificadores |
""")

st.markdown("---")
st.caption(
    f"Datos reales vía MongoDB Atlas + pymongo (sin API intermedia) · "
    f"Sesión iniciada: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)
