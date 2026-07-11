import streamlit as st
from db import get_mercado_data # Usaremos esto para validar la conexión

# Configuración inicial de la pestaña del navegador
st.set_page_config(
    page_title="InvestAI — Sistema Inteligente",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar los estados de sesión si no existen
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "ticker_global" not in st.session_state:
    st.session_state.ticker_global = "FSM"

# ====================================================================
# PANTALLA DE LOGIN (Extraído de index.html)
# ====================================================================
if not st.session_state.logged_in:
    # Ocultar la barra de navegación lateral si no se ha logueado
    st.markdown("""
        <style>
        [data-testid="stSidebarNav"] {display: none !important;}
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.write("")
        st.write("")
        st.markdown("<h1 style='text-align: center; color: #38BDF8;'>📈 InvestAI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94A3B8;'>Sistema Inteligente de Inversión</p>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.subheader("Iniciar Sesión")
            email = st.text_input("Correo Electrónico", value="demo@investai.com", placeholder="demo@investai.com")
            password = st.text_input("Contraseña", value="demo123", type="password", placeholder="******")
            submit_button = st.form_submit_button("Ingresar al Sistema", use_container_width=True)
            
            if submit_button:
                if email == "demo@investai.com" and password == "demo123":
                    st.session_state.logged_in = True
                    st.success("✓ Autenticación exitosa.")
                    st.rerun()
                else:
                    st.error("❌ Credenciales inválidas. Usa la cuenta demo.")

# ====================================================================
# ENTORNO PRINCIPAL (Una vez autenticado)
# ====================================================================
else:
    # Barra lateral global que verás en todas las páginas
    with st.sidebar:
        st.markdown("<h2 style='color: #38BDF8;'>📈 InvestAI</h2>", unsafe_allow_html=True)
        st.write("👤 **Usuario:** demo@investai.com")
        st.divider()
        
        # Selector de Ticker Global compartido
        tickers = ["FSM", "VOLCABC1.LM", "ABX.TO", "BVN", "BHP"]
        ticker_seleccionado = st.selectbox(
            "Ticker Activo:",
            options=tickers,
            index=tickers.index(st.session_state.ticker_global)
        )
        st.session_state.ticker_global = ticker_seleccionado
        st.divider()
        
        if st.button("Cerrar Sesión", type="primary", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    # Contenido de la Página de Inicio
    st.title("🏆 ¡Bienvenido a InvestAI!")
    st.markdown("### Plataforma de Analítica Avanzada Cuantitativa")
    st.write(f"Has iniciado sesión correctamente. Actualmente estás consultando el activo global: **{st.session_state.ticker_global}**.")
    st.write("Explora los modelos predictivos de Machine Learning, el optimizador Markowitz y las simulaciones en el menú de la izquierda.")
