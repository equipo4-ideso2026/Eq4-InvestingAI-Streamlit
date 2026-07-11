"""
streamlit_app.py — InvestAI (Entry Point)
============================================
Punto de entrada de la aplicación:
  1. Pantalla de login simulada (cuenta demo, igual que el index.html original).
  2. Sidebar con el selector de ticker global — `st.session_state.ticker_global` —
     que todas las páginas de `pages/` leen para sincronizarse.
  3. Pantalla de bienvenida con resumen rápido del activo seleccionado y
     accesos directos a los 11 módulos.

El menú de navegación hacia las páginas lo genera Streamlit automáticamente
a partir de la carpeta `pages/`: el prefijo numérico de cada archivo (1_…
11_) define el orden, replicando el de las pestañas del navbar original.
"""

import streamlit as st

from db import get_mercado_data, get_noticias

# ────────────────────────────────────────────────────────────────
# Configuración de página (debe ser el primer comando de Streamlit)
# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InvestAI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────
# Paleta oscura fiel al index.html original
# ────────────────────────────────────────────────────────────────
COLOR_BG = "#0f172a"
COLOR_CARD = "#1e293b"
COLOR_TEXT = "#f8fafc"
COLOR_MUTED = "#94a3b8"
COLOR_ACCENT = "#38bdf8"
COLOR_POSITIVE = "#26A69A"
COLOR_NEGATIVE = "#EF5350"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {COLOR_BG}; color: {COLOR_TEXT}; }}
    section[data-testid="stSidebar"] {{ background-color: {COLOR_CARD}; }}

    /* Tarjeta de login: le da al <form> la misma apariencia que
       .login-container en el index.html original (cristal esmerilado). */
    div[data-testid="stForm"] {{
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 32px 28px;
    }}

    .login-header {{ text-align: center; margin-bottom: 12px; }}
    .login-header h1 {{ font-size: 32px; font-weight: 700; color: {COLOR_ACCENT}; margin-bottom: 4px; }}
    .login-header p {{ font-size: 14px; color: {COLOR_MUTED}; }}

    .home-header p {{ color: {COLOR_MUTED}; font-size: 13px; margin-top: -8px; }}
    .card-title {{ font-size: 14px; font-weight: 600; color: {COLOR_TEXT}; margin-bottom: 12px; }}

    div[data-testid="stMetric"] {{
        background-color: {COLOR_CARD};
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }}
    div[data-testid="stMetricLabel"] {{ color: {COLOR_MUTED}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────
# Estado de sesión
# ────────────────────────────────────────────────────────────────
st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("user_email", None)
st.session_state.setdefault("ticker_global", "FSM")

DEMO_EMAIL = "demo@investai.com"
DEMO_PASSWORD = "demo123"

TICKERS_INFO = {
    "FSM": "FSM — Fortuna Silver Mines",
    "VOLCABC1.LM": "VOLCABC1.LM — Volcan Compañía Minera",
    "ABX.TO": "ABX.TO — Barrick Gold",
    "BVN": "BVN — Compañía de Minas Buenaventura",
    "BHP": "BHP — BHP Group",
}

# Páginas en el mismo orden que las pestañas del navbar original (M2, MSVC,
# M3, M4, M5, M6, M7, M11, M9, M10, M8)
PAGINAS = [
    ("pages/1_📊_Mercado.py", "📊 Mercado"),
    ("pages/2_🎯_Clasificador_SVC.py", "🎯 Clasificador SVC"),
    ("pages/3_🧠_Clasificadores_RNN.py", "🧠 Clasificadores RNN"),
    ("pages/4_📈_Regresor_LSTM.py", "📈 Regresor LSTM"),
    ("pages/5_📰_NLP_Noticias.py", "📰 NLP Noticias"),
    ("pages/6_⚡_Estrategias_Markowitz.py", "⚡ Estrategias Markowitz"),
    ("pages/7_💼_Gestion_Portafolio.py", "💼 Gestión de Portafolio"),
    ("pages/8_📋_Backtesting_Detallado.py", "📋 Backtesting Detallado"),
    ("pages/9_🏆_Dashboard_Central.py", "🏆 Dashboard Central"),
    ("pages/10_⚙️_Consola_de_Modelos.py", "⚙️ Consola de Modelos"),
    ("pages/11_🚀_Terminal_Broker.py", "🚀 Terminal Broker"),
]


# ────────────────────────────────────────────────────────────────
# Pantalla de login
# ────────────────────────────────────────────────────────────────
def pantalla_login() -> None:
    _, col_centro, _ = st.columns([1, 1.2, 1])
    with col_centro:
        st.markdown(
            """
            <div class="login-header">
                <h1>📈 InvestAI</h1>
                <p>Sistema Inteligente de Inversión</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            email = st.text_input("Correo Electrónico", placeholder="demo@investai.com")
            password = st.text_input("Contraseña", type="password", placeholder="demo123")
            enviado = st.form_submit_button("Iniciar Sesión", use_container_width=True, type="primary")

        st.caption(f"Cuenta demo: `{DEMO_EMAIL}` / `{DEMO_PASSWORD}`")

        if enviado:
            if not email or not password:
                st.error("Completa tu correo y contraseña.")
            elif email.strip().lower() == DEMO_EMAIL and password == DEMO_PASSWORD:
                st.session_state.authenticated = True
                st.session_state.user_email = email.strip().lower()
                st.rerun()
            else:
                st.error("Credenciales incorrectas. Usa la cuenta demo indicada arriba.")


# ────────────────────────────────────────────────────────────────
# Sidebar de la app (visible en todas las páginas una vez autenticado)
# ────────────────────────────────────────────────────────────────
def construir_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 📈 InvestAI")
        st.caption(f"Sesión: {st.session_state.user_email}")
        st.divider()

        tickers_lista = list(TICKERS_INFO.keys())
        idx_actual = (
            tickers_lista.index(st.session_state.ticker_global)
            if st.session_state.ticker_global in tickers_lista
            else 0
        )
        ticker_seleccionado = st.selectbox(
            "Activo (Ticker Global)",
            options=tickers_lista,
            index=idx_actual,
            format_func=lambda t: TICKERS_INFO.get(t, t),
            key="selector_ticker_sidebar",
        )
        if ticker_seleccionado != st.session_state.ticker_global:
            st.session_state.ticker_global = ticker_seleccionado
            st.rerun()

        st.caption("El ticker elegido aquí se usa automáticamente en todos los módulos.")
        st.divider()

        if st.button("Cerrar Sesión", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_email = None
            st.rerun()


# ────────────────────────────────────────────────────────────────
# Pantalla principal (home) tras iniciar sesión
# ────────────────────────────────────────────────────────────────
def pantalla_home() -> None:
    construir_sidebar()
    ticker = st.session_state.ticker_global

    st.markdown(
        f"""
        <div class="home-header">
            <h2>🏆 InvestAI — Sistema Inteligente de Inversión</h2>
            <p>Ensamblado de Modelos IA sobre 5 activos mineros · Datos reales vía MongoDB Atlas</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Resumen rápido del activo seleccionado ────────────────────
    df = get_mercado_data(ticker, dias=30)
    noticias = get_noticias(fuente="all", sentimiento="all", ticker=ticker)

    c1, c2, c3 = st.columns(3)
    with c1:
        if not df.empty:
            precio = float(df["close"].iloc[-1])
            st.metric("Precio Actual", f"${precio:,.2f}", help=TICKERS_INFO.get(ticker, ticker))
        else:
            st.metric("Precio Actual", "Sin datos")
    with c2:
        if not df.empty and len(df) > 1:
            cambio = ((df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]) * 100
            st.metric("Variación (30 días)", f"{cambio:+.2f}%")
        else:
            st.metric("Variación (30 días)", "—")
    with c3:
        sentimiento = noticias.get("sentimiento_consolidado")
        st.metric("Sentimiento de Noticias", sentimiento or "Sin datos")

    if df.empty:
        st.warning(
            "⚠ No hay datos de mercado para este ticker todavía. Ve a **⚙️ Consola de Modelos** "
            "y ejecuta el pipeline de ingesta para poblar MongoDB Atlas."
        )

    st.divider()

    # ── Accesos directos a los 11 módulos (equivalente al navbar original) ──
    st.markdown('<div class="card-title">Módulos disponibles</div>', unsafe_allow_html=True)
    columnas = st.columns(4)
    for i, (ruta, etiqueta) in enumerate(PAGINAS):
        with columnas[i % 4]:
            st.page_link(ruta, label=etiqueta, use_container_width=True)


# ────────────────────────────────────────────────────────────────
# Enrutamiento principal
# ────────────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    pantalla_login()
else:
    pantalla_home()
