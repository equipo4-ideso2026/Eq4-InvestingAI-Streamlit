"""
pages/6_⚡_Estrategias_Markowitz.py — Optimizador de Portafolio (Markowitz)
=============================================================================
Reemplaza el módulo M6 del index.html original: asignación óptima de
activos, distribución del portafolio y frontera eficiente Retorno vs.
Volatilidad.

Consulta `db.py` directamente contra MongoDB Atlas (colección
`estrategias`, poblada por el Notebook 9), sin pasar por FastAPI/ngrok.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from db import get_estrategias

# ────────────────────────────────────────────────────────────────
# Configuración de página
# ────────────────────────────────────────────────────────────────
try:
    st.set_page_config(page_title="Estrategias Markowitz · InvestAI", page_icon="⚡", layout="wide")
except Exception:
    pass

# ────────────────────────────────────────────────────────────────
# Paleta oscura fiel al index.html original
# ────────────────────────────────────────────────────────────────
COLOR_BG = "#0f172a"
COLOR_CARD = "#1e293b"
COLOR_TEXT = "#f8fafc"
COLOR_MUTED = "#94a3b8"
COLOR_POSITIVE = "#26A69A"
COLOR_NEGATIVE = "#EF5350"

RIESGO_COLOR = {
    "bajo": "#26A69A",
    "medio": "#FFC107",
    "moderado": "#FFC107",
    "alto": "#EF5350",
}
DISTRIBUCION_COLORES = ["#2563eb", "#38bdf8", "#06b6d4", "#1d4ed8", "#7dd3fc"]

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {COLOR_BG}; color: {COLOR_TEXT}; }}
    div[data-testid="stMetric"] {{
        background-color: {COLOR_CARD};
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }}
    div[data-testid="stMetricLabel"] {{ color: {COLOR_MUTED}; }}
    .modulo-header p {{ color: {COLOR_MUTED}; font-size: 13px; margin-top: -8px; }}
    .card-title {{ font-size: 14px; font-weight: 600; color: {COLOR_TEXT}; margin-bottom: 12px; }}
    .footer-nota {{
        margin-top:24px; padding-top:16px; border-top:1px solid rgba(255,255,255,0.08);
        font-size:12px; color:{COLOR_MUTED}; text-align:center;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="modulo-header">
        <h2>⚡ Optimizador de Portafolio — Markowitz</h2>
        <p>Frontera eficiente sobre los 5 activos mineros del proyecto (Notebook 9)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────
# Controles locales: capital, horizonte, perfil de riesgo
# ────────────────────────────────────────────────────────────────
HORIZONTES = {"3 Meses": "3m", "6 Meses": "6m", "1 Año": "1y", "3 Años": "3y"}
PERFILES = {"Conservador": "conservador", "Moderado": "moderado", "Agresivo": "agresivo"}

col_capital, col_horizonte, col_riesgo = st.columns(3)
with col_capital:
    capital = st.number_input("Capital Inicial (USD)", min_value=0.0, value=100000.0, step=1000.0)
with col_horizonte:
    horizonte_label = st.selectbox("Horizonte Temporal", options=list(HORIZONTES.keys()), index=2)
    horizonte = HORIZONTES[horizonte_label]
with col_riesgo:
    riesgo_label = st.selectbox("Nivel de Riesgo", options=list(PERFILES.keys()), index=1)
    perfil = PERFILES[riesgo_label]

# ────────────────────────────────────────────────────────────────
# Datos
# ────────────────────────────────────────────────────────────────
doc = get_estrategias(perfil_riesgo=perfil, horizonte=horizonte, capital=capital)

if not doc:
    st.warning(
        f"⚠ No hay estrategia calculada para **{riesgo_label} / {horizonte_label}**. "
        "Verifica que el Notebook 9 (Optimizador) haya guardado esta combinación en `estrategias`."
    )
    st.stop()

# ────────────────────────────────────────────────────────────────
# 4 tarjetas de métricas
# ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Capital Asignado", f"${doc.get('capital_inicial_usd', capital):,.0f}")
with c2:
    ret = doc.get("retorno_esperado_anual")
    st.metric("Retorno Esperado (Anual)", f"{ret * 100:.2f}%" if ret is not None else "—")
with c3:
    vol = doc.get("volatilidad_anual")
    st.metric("Volatilidad Anual", f"{vol * 100:.2f}%" if vol is not None else "—")
with c4:
    sharpe = doc.get("sharpe_ratio")
    st.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe is not None else "—")

st.caption(f"Actualizado: {doc.get('fecha_calculo', '—')}")

# ────────────────────────────────────────────────────────────────
# Tabla de asignación + gráfico de distribución
# ────────────────────────────────────────────────────────────────
col_tabla, col_pie = st.columns([2, 1])

activos = sorted(doc.get("activos", []), key=lambda a: a.get("asignacion_pct", 0), reverse=True)

with col_tabla:
    st.markdown('<div class="card-title">Asignación de Activos (Markowitz — sin venta en corto)</div>', unsafe_allow_html=True)
    if activos:
        df_activos = pd.DataFrame(activos)
        columnas = ["ticker", "tipo", "asignacion_pct", "riesgo", "rendimiento_estimado_pct", "monto_asignado_usd"]
        columnas_disponibles = [c for c in columnas if c in df_activos.columns]
        df_mostrar = df_activos[columnas_disponibles].rename(
            columns={
                "ticker": "Activo",
                "tipo": "Tipo",
                "asignacion_pct": "Asignación %",
                "riesgo": "Riesgo",
                "rendimiento_estimado_pct": "Rendimiento Est. %",
                "monto_asignado_usd": "Monto Asignado (USD)",
            }
        )
        st.dataframe(df_mostrar, use_container_width=True, hide_index=True)
    else:
        st.info("Sin datos de asignación.")

with col_pie:
    st.markdown('<div class="card-title">Distribución de Portafolio</div>', unsafe_allow_html=True)
    con_asignacion = [a for a in activos if a.get("asignacion_pct", 0) > 0]
    if con_asignacion:
        fig_pie = px.pie(
            names=[a["ticker"] for a in con_asignacion],
            values=[a["asignacion_pct"] for a in con_asignacion],
            color_discrete_sequence=DISTRIBUCION_COLORES,
            hole=0.35,
        )
        fig_pie.update_traces(textinfo="label+percent", marker=dict(line=dict(color=COLOR_BG, width=2)))
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
            font=dict(color=COLOR_TEXT),
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Sin asignación positiva para graficar.")

# ────────────────────────────────────────────────────────────────
# Frontera eficiente
# ────────────────────────────────────────────────────────────────
st.markdown('<div class="card-title">Frontera Eficiente — Retorno vs. Volatilidad</div>', unsafe_allow_html=True)

frontera = doc.get("frontera_eficiente", [])
if frontera:
    fig_frontera = go.Figure()
    fig_frontera.add_trace(
        go.Scatter(
            x=[p["volatilidad"] * 100 for p in frontera],
            y=[p["retorno"] * 100 for p in frontera],
            mode="lines",
            name="Frontera Eficiente",
            line=dict(color="#38bdf8", width=2),
        )
    )
    if doc.get("volatilidad_anual") is not None and doc.get("retorno_esperado_anual") is not None:
        fig_frontera.add_trace(
            go.Scatter(
                x=[doc["volatilidad_anual"] * 100],
                y=[doc["retorno_esperado_anual"] * 100],
                mode="markers",
                name=f"Perfil {perfil}",
                marker=dict(size=16, color="#FFD700", symbol="star", line=dict(color=COLOR_TEXT, width=1)),
            )
        )
    fig_frontera.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT),
        xaxis=dict(title="Volatilidad Anual (%)", gridcolor="#1e293b"),
        yaxis=dict(title="Retorno Esperado Anual (%)", gridcolor="#1e293b"),
        legend=dict(orientation="h", y=1.15),
        margin=dict(l=60, r=30, t=20, b=50),
        height=380,
    )
    st.plotly_chart(fig_frontera, use_container_width=True)
else:
    st.info("Sin datos de frontera eficiente para esta combinación.")

# ────────────────────────────────────────────────────────────────
# Pie de página
# ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="footer-nota">
        Datos reales vía MongoDB Atlas · Optimización de Portafolio (Markowitz) ·
        Actualizado: {doc.get('fecha_calculo', '—')}
    </div>
    """,
    unsafe_allow_html=True,
)
