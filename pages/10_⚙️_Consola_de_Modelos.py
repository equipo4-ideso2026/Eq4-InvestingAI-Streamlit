"""
pages/10_⚙️_Consola_de_Modelos.py — AI Model Console
========================================================
Reemplaza el módulo M10 del index.html original: selección masiva de
modelos, ejecución simulada con barra de progreso, y tabla de resultados
reales leídos de MongoDB (última predicción guardada por cada notebook).

Consulta `db.py` directamente contra MongoDB Atlas, sin pasar por
FastAPI/ngrok.
"""

import time
from datetime import datetime

import pandas as pd
import streamlit as st

from db import get_lstm_regresor, get_rnn_data, get_svc_data

# ────────────────────────────────────────────────────────────────
# Configuración de página
# ────────────────────────────────────────────────────────────────
try:
    st.set_page_config(page_title="Consola de Modelos · InvestAI", page_icon="⚙️", layout="wide")
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
    .seccion-label {{
        font-size: 11px; font-weight: 700; color: {COLOR_MUTED};
        margin: 12px 0 4px; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.1);
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
    f"""
    <div class="modulo-header">
        <h2>⚙️ AI Model Console — Configuración y Control de los Modelos</h2>
        <p>Ticker: <strong>{ticker}</strong> · Selecciona modelos y ejecuta contra MongoDB Atlas</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────
# Catálogo de modelos disponibles (checkbox por modelo)
# ────────────────────────────────────────────────────────────────
MODELOS_CLASIFICADORES = [
    ("SVC", "SVC", "Clasificador", "scikit-learn"),
    ("LSTM", "LSTM Clf", "Clasificador", "TensorFlow"),
    ("BiLSTM", "BiLSTM", "Clasificador", "TensorFlow"),
    ("GRU", "GRU", "Clasificador", "TensorFlow"),
    ("SimpleRNN", "SimpleRNN", "Clasificador", "TensorFlow"),
]
MODELO_REGRESOR = ("LSTM_REG", "LSTM Regresor", "Regresor", "TensorFlow")
TODOS_LOS_MODELOS = MODELOS_CLASIFICADORES + [MODELO_REGRESOR]

for clave, *_ in TODOS_LOS_MODELOS:
    st.session_state.setdefault(f"chk_{clave}", True)

# ────────────────────────────────────────────────────────────────
# Control de procesos: seleccionar / deseleccionar / ejecutar
# ────────────────────────────────────────────────────────────────
col_sel, col_desel, col_run = st.columns(3)
with col_sel:
    if st.button("Seleccionar Todos", use_container_width=True):
        for clave, *_ in TODOS_LOS_MODELOS:
            st.session_state[f"chk_{clave}"] = True
        st.rerun()
with col_desel:
    if st.button("Deseleccionar Todos", use_container_width=True):
        for clave, *_ in TODOS_LOS_MODELOS:
            st.session_state[f"chk_{clave}"] = False
        st.rerun()
with col_run:
    ejecutar = st.button("▶ Ejecutar Modelos Seleccionados", use_container_width=True, type="primary")

# ────────────────────────────────────────────────────────────────
# Layout: lista de checkboxes (izq) + tabla de resultados (der)
# ────────────────────────────────────────────────────────────────
col_lista, col_resultados = st.columns([1, 2])

with col_lista:
    st.markdown('<div class="card-title">Selección de Modelos</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="seccion-label">CLASIFICADORES (BUY/SELL)</div>', unsafe_allow_html=True)
        for clave, nombre, _tipo, libreria in MODELOS_CLASIFICADORES:
            st.checkbox(f"{nombre}  ·  _{libreria}_", key=f"chk_{clave}")

        st.markdown('<div class="seccion-label">REGRESOR (PRECIO FUTURO)</div>', unsafe_allow_html=True)
        clave_r, nombre_r, _tipo_r, lib_r = MODELO_REGRESOR
        st.checkbox(f"{nombre_r}  ·  _{lib_r}_", key=f"chk_{clave_r}")

with col_resultados:
    st.markdown('<div class="card-title">Resultados de la Última Ejecución</div>', unsafe_allow_html=True)
    tabla_placeholder = st.empty()

    if "m10_resultados" not in st.session_state:
        st.session_state["m10_resultados"] = None
        st.session_state["m10_ultima_ejecucion"] = None

    if ejecutar:
        seleccionados = [
            (clave, nombre, tipo) for clave, nombre, tipo, _lib in TODOS_LOS_MODELOS
            if st.session_state.get(f"chk_{clave}")
        ]

        if not seleccionados:
            st.warning("Selecciona al menos un modelo.")
        else:
            progreso = st.progress(0, text="Iniciando ejecución…")
            filas = []

            for i, (clave, nombre, tipo) in enumerate(seleccionados):
                progreso.progress(
                    int(((i + 1) / len(seleccionados)) * 100),
                    text=f"Consultando {nombre}…",
                )
                time.sleep(0.25)  # simulación visual de latencia de ejecución

                if clave == "SVC":
                    datos = get_svc_data(ticker)
                elif clave == "LSTM_REG":
                    datos = get_lstm_regresor(ticker)
                else:
                    datos = get_rnn_data(ticker, clave)

                if not datos:
                    filas.append(
                        {
                            "Modelo": nombre, "Tipo": tipo, "Señal/Predicción": "—",
                            "Accuracy/RMSE": "—", "F1/R²": "—",
                            "Fecha Predicción": "—", "Estado Conexión": "✕ Sin datos",
                        }
                    )
                    continue

                if clave == "LSTM_REG":
                    precio = datos.get("precio_actual_usd")
                    m = datos.get("metricas", {})
                    filas.append(
                        {
                            "Modelo": nombre, "Tipo": tipo,
                            "Señal/Predicción": f"${precio:,.2f}" if precio is not None else "—",
                            "Accuracy/RMSE": f"${m.get('rmse_usd'):.2f}" if m.get("rmse_usd") is not None else "—",
                            "F1/R²": f"{m.get('r2'):.3f}" if m.get("r2") is not None else "—",
                            "Fecha Predicción": datos.get("fecha_prediccion", "—"),
                            "Estado Conexión": "✓ OK",
                        }
                    )
                else:
                    senal = datos.get("prediccion", {}).get("senal", "—")
                    m = datos.get("metricas", {})
                    filas.append(
                        {
                            "Modelo": nombre, "Tipo": tipo,
                            "Señal/Predicción": senal,
                            "Accuracy/RMSE": f"{m.get('accuracy') * 100:.1f}%" if m.get("accuracy") is not None else "—",
                            "F1/R²": f"{m.get('f1'):.2f}" if m.get("f1") is not None else "—",
                            "Fecha Predicción": datos.get("fecha_prediccion", "—"),
                            "Estado Conexión": "✓ OK",
                        }
                    )

            progreso.progress(100, text="Ejecución completada ✓")
            time.sleep(0.3)
            progreso.empty()

            st.session_state["m10_resultados"] = filas
            st.session_state["m10_ultima_ejecucion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    resultados = st.session_state.get("m10_resultados")
    if resultados:
        df_resultados = pd.DataFrame(resultados)
        tabla_placeholder.dataframe(df_resultados, use_container_width=True, hide_index=True)
    else:
        tabla_placeholder.info('Presiona "▶ Ejecutar Modelos Seleccionados" para consultar MongoDB Atlas.')

# ────────────────────────────────────────────────────────────────
# Pie de página
# ────────────────────────────────────────────────────────────────
ultima_ejecucion = st.session_state.get("m10_ultima_ejecucion") or "Aún no se ha ejecutado ningún modelo en esta sesión"
st.markdown(
    f"""
    <div class="footer-nota">
        Ejecución real contra MongoDB Atlas · Última ejecución: {ultima_ejecucion}
    </div>
    """,
    unsafe_allow_html=True,
)
