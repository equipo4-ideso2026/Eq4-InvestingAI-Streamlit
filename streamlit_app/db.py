"""
db.py — Conexión centralizada a MongoDB Atlas para toda la app Streamlit.

Reemplaza la capa FastAPI + ngrok (Notebook 11): en vez de que el frontend
haga fetch() a una API HTTP, Streamlit se conecta DIRECTO a MongoDB con
pymongo — mismo patrón de lectura que usaban los endpoints, sin el
intermediario. Esto elimina la dependencia de tener el Notebook 11
corriendo y con un túnel ngrok activo.

Los nombres de colecciones y el nombre de la base de datos ('spbi') son
EXACTAMENTE los mismos que usan los 10 notebooks del proyecto — no cambia
nada del lado de la ingesta ni del entrenamiento de modelos.
"""

import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

DB_NAME = "spbi"

COL_PRECIOS      = "precios_ohlcv"        # Notebook 1
COL_PREDICCIONES = "predicciones"         # Notebooks 2-6
COL_METRICAS     = "metricas_modelos"     # Notebooks 2-7
COL_NOTICIAS     = "noticias"             # Notebook 8
COL_ESTRATEGIAS  = "estrategias"          # Notebook 9
COL_BACKTESTS    = "backtests"            # Notebook 10
COL_ORDENES      = "ordenes_simuladas"    # Broker (manual, sin notebook)

TICKERS = ["FSM", "VOLCABC1.LM", "ABX.TO", "BVN", "BHP"]
EMPRESAS = {
    "FSM": "Fortuna Silver Mines",
    "VOLCABC1.LM": "Volcan Compañía Minera",
    "ABX.TO": "Barrick Gold",
    "BVN": "Compañía de Minas Buenaventura",
    "BHP": "BHP Group",
}
MODELOS_RNN = ["LSTM", "BiLSTM", "GRU", "SimpleRNN"]
MODELOS_TODOS = ["SVC", "LSTM", "BiLSTM", "GRU", "SimpleRNN"]
PERFILES_RIESGO = ["conservador", "moderado", "agresivo"]
HORIZONTES = ["3m", "6m", "1y", "3y"]


@st.cache_resource(show_spinner=False)
def get_client():
    """
    Crea (y cachea, una sola vez por sesión de servidor de Streamlit) el
    cliente de MongoDB. st.cache_resource es el equivalente en Streamlit
    a mantener una conexión abierta reutilizable, en vez de reconectar
    en cada rerun del script (Streamlit re-ejecuta el script completo
    en cada interacción del usuario).
    """
    uri = st.secrets.get("MONGO_URI")
    if not uri:
        st.error(
            "⚠ No se encontró el secret `MONGO_URI`. "
            "Configúralo en `.streamlit/secrets.toml` (local) o en "
            "**Settings → Secrets** si ya desplegaste en Streamlit Cloud."
        )
        st.stop()
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    return client


def get_db():
    """Devuelve el objeto de base de datos 'spbi', verificando la conexión."""
    client = get_client()
    try:
        client.admin.command("ping")
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        st.error(f"⚠ No se pudo conectar a MongoDB Atlas: {e}")
        st.info(
            "Verifica que `MONGO_URI` sea correcta y que tu IP esté permitida "
            "en Network Access de MongoDB Atlas (o que esté configurado 0.0.0.0/0)."
        )
        st.stop()
    return client[DB_NAME]


def escalar_backtest(doc: dict, capital_deseado: float) -> dict:
    """
    Reescala PROPORCIONALMENTE los montos en dólares de un documento de
    'backtests' (que se calculó con un capital_base fijo, ej. $100,000)
    a un capital distinto elegido por el usuario en Streamlit.

    Esto es matemáticamente válido porque el Notebook 10 no usa
    apalancamiento ni redondea a acciones enteras — todo (equity_curve,
    valor_actual, pnl_usd) es una función LINEAL del capital inicial.
    Los campos que son porcentajes o ratios (accuracy, sharpe, retorno_pct,
    win_rate, drawdown, precios de acciones) NO se tocan, porque son
    invariantes de escala — solo se reescalan los montos absolutos en USD.

    No modifica el documento original en MongoDB, solo la copia usada
    para mostrar en pantalla.
    """
    import copy
    doc = copy.deepcopy(doc)
    capital_base = doc.get("capital_base", 100_000.0)
    if not capital_base:
        return doc
    escala = capital_deseado / capital_base

    doc["capital_base"] = capital_deseado

    for punto in doc.get("equity_curve", []):
        punto["valor"] = punto["valor"] * escala

    for pos in doc.get("posiciones_finales", []):
        pos["capital_inicial_sleeve"] = pos.get("capital_inicial_sleeve", 0) * escala
        pos["valor_actual"] = pos.get("valor_actual", 0) * escala
        pos["pnl_usd"] = pos.get("pnl_usd", 0) * escala
        # cantidad de acciones también escala (mismo % del capital, más plata → más acciones)
        pos["cantidad"] = pos.get("cantidad", 0) * escala
        # precio_entrada / precio_actual son precios de mercado reales: NO se tocan

    return doc


def validar_ticker(ticker: str) -> str:
    """Normaliza y valida un ticker contra la lista de los 5 tickers del proyecto."""
    t = ticker.strip()
    if t not in TICKERS:
        st.error(f"Ticker '{ticker}' no reconocido. Válidos: {TICKERS}")
        st.stop()
    return t
