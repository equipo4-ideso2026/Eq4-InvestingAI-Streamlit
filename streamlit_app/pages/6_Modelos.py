"""
Página Modelos — consola que consulta (no reentrena) los 5 clasificadores
para el ticker elegido, equivalente al Módulo M10.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from db import get_db, validar_ticker, TICKERS, EMPRESAS, MODELOS_TODOS, COL_PREDICCIONES, COL_METRICAS

st.set_page_config(page_title="Modelos — InvestAI", page_icon="⚙️", layout="wide")

if not st.session_state.get("logueado"):
    st.warning("Inicia sesión desde la página principal.")
    st.stop()

st.title("⚙️ Consola de Modelos")
st.caption("Consulta en vivo de los 5 clasificadores entrenados (SVC, LSTM, BiLSTM, GRU, SimpleRNN)")

db = get_db()

ticker = st.selectbox("Ticker", TICKERS, format_func=lambda t: f"{t} — {EMPRESAS[t]}",
                       index=TICKERS.index(st.session_state.get("ticker_global", "FSM")), key="m10_ticker")
t = validar_ticker(ticker)

st.markdown("#### Selecciona los modelos a consultar")
cols = st.columns(5)
seleccionados = []
for i, modelo in enumerate(MODELOS_TODOS):
    with cols[i]:
        if st.checkbox(modelo, value=True, key=f"chk_{modelo}"):
            seleccionados.append(modelo)

if st.button("▶ Consultar modelos seleccionados", type="primary"):
    if not seleccionados:
        st.warning("Selecciona al menos un modelo.")
    else:
        filas = []
        progress = st.progress(0, text="Consultando...")
        for i, modelo in enumerate(seleccionados):
            pred = db[COL_PREDICCIONES].find_one({"ticker": t, "modelo": modelo}, {"_id": 0})
            met = db[COL_METRICAS].find_one({"ticker": t, "modelo": modelo}, {"_id": 0})
            if pred:
                estado = "✓ OK"
                senal = pred.get("senal", "—")
                metrica1 = f"Acc: {met.get('accuracy', 0)*100:.1f}%" if met and met.get("accuracy") is not None else "—"
                metrica2 = f"F1: {met.get('f1', 0):.3f}" if met and met.get("f1") is not None else "—"
                fecha = pred.get("fecha_prediccion", "—")
            else:
                estado = "✕ Sin datos"
                senal = "—"
                metrica1 = "—"
                metrica2 = "—"
                fecha = "—"

            filas.append({"Modelo": modelo, "Señal": senal, "Métrica 1": metrica1, "Métrica 2": metrica2,
                           "Fecha": fecha, "Estado": estado})
            progress.progress((i + 1) / len(seleccionados), text=f"Consultando {modelo}...")

        progress.empty()
        st.session_state["m10_resultados"] = filas
        st.session_state["m10_fecha_ejecucion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if "m10_resultados" in st.session_state:
    st.markdown("#### Resultados")
    df_res = pd.DataFrame(st.session_state["m10_resultados"])

    def resaltar(row):
        color = "background-color: rgba(38,166,154,0.15)" if row["Estado"] == "✓ OK" else "background-color: rgba(239,83,80,0.15)"
        return [color] * len(row)

    st.dataframe(df_res.style.apply(resaltar, axis=1), use_container_width=True, hide_index=True)
    st.caption(f"Última ejecución: {st.session_state.get('m10_fecha_ejecucion', '—')}")
else:
    st.info("Selecciona modelos y presiona 'Consultar' para ver resultados.")

st.caption(f"Ejecución real contra MongoDB Atlas · Ticker: {t}")
