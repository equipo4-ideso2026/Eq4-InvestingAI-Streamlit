"""
Página Estrategias — equivalente a /api/estrategias + Módulo M6.
Optimización de portafolio (Markowitz, Notebook 9).
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from db import get_db, COL_ESTRATEGIAS, PERFILES_RIESGO, HORIZONTES

st.set_page_config(page_title="Estrategias — InvestAI", page_icon="⚡", layout="wide")

if not st.session_state.get("logueado"):
    st.warning("Inicia sesión desde la página principal.")
    st.stop()

st.title("⚡ Generación de Estrategias de Inversión")
st.caption("Perfil inversor + distribución de portafolio (Optimización de Markowitz — Notebook 9)")

db = get_db()

st.markdown("#### Configuración de Estrategia")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Capital Inicial (USD)", f"${st.session_state.get('capital', 100000):,.0f}")
    st.caption("Se edita desde el sidebar (⚡ compartido con Portafolio y Backtesting)")
    capital = st.session_state.get("capital", 100_000.0)
with c2:
    horizonte_label = st.selectbox("Horizonte Temporal", ["3 Meses", "6 Meses", "1 Año", "3 Años"], index=2)
    horizonte = {"3 Meses": "3m", "6 Meses": "6m", "1 Año": "1y", "3 Años": "3y"}[horizonte_label]
with c3:
    perfil_label = st.selectbox("Nivel de Riesgo", ["Conservador", "Moderado", "Agresivo"], index=1)
    perfil = perfil_label.lower()

doc = db[COL_ESTRATEGIAS].find_one({"perfil_riesgo": perfil, "horizonte": horizonte}, {"_id": 0})

if not doc:
    st.error(f"No hay estrategia calculada para perfil='{perfil}', horizonte='{horizonte}'. Ejecuta el Notebook 9 (Markowitz).")
    st.stop()

if not doc.get("optimizacion_exitosa", True):
    st.warning("⚠ La optimización no convergió para esta combinación — se usaron pesos iguales como respaldo.")

col1, col2, col3 = st.columns(3)
col1.metric("Retorno Esperado Anual", f"{doc['retorno_esperado_anual']*100:.2f}%")
col2.metric("Volatilidad Anual", f"{doc['volatilidad_anual']*100:.2f}%")
col3.metric("Sharpe Ratio", f"{doc['sharpe_ratio']:.2f}")

st.markdown("---")

activos = sorted(doc.get("activos", []), key=lambda a: -a["asignacion_pct"])

col_izq, col_der = st.columns([1.6, 1])

with col_izq:
    st.markdown("#### Asignación de Activos")
    df_activos = pd.DataFrame(activos)
    df_activos["Monto (USD)"] = df_activos["asignacion_pct"] / 100 * capital
    df_activos_display = df_activos.rename(columns={
        "ticker": "Activo", "tipo": "Tipo", "asignacion_pct": "Asignación %",
        "riesgo": "Riesgo", "rendimiento_estimado_pct": "Rendimiento Est. %"
    })
    st.dataframe(
        df_activos_display[["Activo", "Tipo", "Asignación %", "Riesgo", "Rendimiento Est. %", "Monto (USD)"]],
        use_container_width=True, hide_index=True,
        column_config={
            "Asignación %": st.column_config.NumberColumn(format="%.2f%%"),
            "Rendimiento Est. %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Monto (USD)": st.column_config.NumberColumn(format="$%.2f"),
        }
    )

with col_der:
    st.markdown("#### Distribución del Portafolio")
    con_asignacion = [a for a in activos if a["asignacion_pct"] > 0]
    fig = go.Figure(data=[go.Pie(
        labels=[a["ticker"] for a in con_asignacion],
        values=[a["asignacion_pct"] for a in con_asignacion],
        hole=0.5,
    )])
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ── Frontera eficiente ────────────────────────────────────────────────────
frontera = doc.get("frontera_eficiente", [])
if frontera:
    st.markdown("#### Frontera Eficiente")
    df_f = pd.DataFrame(frontera)
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df_f["volatilidad"] * 100, y=df_f["retorno"] * 100, mode="markers",
                                marker=dict(size=6, color="#42A5F5"), name="Portafolios simulados"))
    fig_f.add_trace(go.Scatter(x=[doc["volatilidad_anual"] * 100], y=[doc["retorno_esperado_anual"] * 100],
                                mode="markers", marker=dict(size=16, color="#EF5350", symbol="star"),
                                name=f"Perfil {perfil_label}"))
    fig_f.update_layout(template="plotly_dark", height=380, xaxis_title="Volatilidad anual (%)",
                         yaxis_title="Retorno esperado anual (%)", margin=dict(l=40, r=40, t=20, b=40))
    st.plotly_chart(fig_f, use_container_width=True)

st.caption(f"Datos reales vía MongoDB Atlas · Optimización de Portafolio (Markowitz) · Actualizado: {doc.get('fecha_calculo', '—')}")
