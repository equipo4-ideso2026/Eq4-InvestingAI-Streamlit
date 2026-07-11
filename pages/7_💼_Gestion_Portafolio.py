"""
pages/7_💼_Gestion_Portafolio.py — Gestión de Portafolio
============================================================
Reemplaza el módulo M7 del index.html original: salud actual del
portafolio simulado (valor total, PnL, retorno %, Sharpe), posiciones
por sleeve de activo, distribución de capital y curva de equity.

Consulta `db.py` directamente contra MongoDB Atlas (colección
`backtests`, poblada por el Notebook 10), sin pasar por FastAPI/ngrok.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from db import get_backtest_report

# ────────────────────────────────────────────────────────────────
# Configuración de página
# ────────────────────────────────────────────────────────────────
try:
    st.set_page_config(page_title="Gestión de Portafolio · InvestAI", page_icon="💼", layout="wide")
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
        <h2>💼 Gestión de Portafolio</h2>
        <p>Distribución, posiciones y curva de equity — simulación diversificada (Notebook 10)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────
# Controles locales: modelo de señales + perfil de riesgo
# ────────────────────────────────────────────────────────────────
MODELOS = ["SVC", "LSTM", "BiLSTM", "GRU", "SimpleRNN"]
PERFILES = {"Conservador": "conservador", "Moderado": "moderado", "Agresivo": "agresivo"}

col_modelo, col_riesgo = st.columns(2)
with col_modelo:
    modelo = st.selectbox("Modelo de Señales", options=MODELOS, index=1)
with col_riesgo:
    riesgo_label = st.selectbox("Perfil de Riesgo", options=list(PERFILES.keys()), index=1)
    perfil = PERFILES[riesgo_label]

# ────────────────────────────────────────────────────────────────
# Datos
# ────────────────────────────────────────────────────────────────
doc = get_backtest_report(modelo=modelo, perfil_riesgo=perfil)

if not doc:
    st.warning(
        f"⚠ No hay backtest disponible para **{modelo} / {riesgo_label}**. "
        "Verifica que el Notebook 10 haya guardado esta combinación en `backtests`."
    )
    st.stop()

posiciones = doc.get("posiciones_finales", [])
m = doc.get("metricas", {})

# ────────────────────────────────────────────────────────────────
# 4 tarjetas de salud del portafolio
# ────────────────────────────────────────────────────────────────
valor_total = sum(p.get("valor_actual", 0) for p in posiciones)
pnl_total = sum(p.get("pnl_usd", 0) for p in posiciones)
capital_inicial = sum(p.get("capital_inicial_sleeve", 0) for p in posiciones)
retorno_pct = (pnl_total / capital_inicial * 100) if capital_inicial > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Valor Total", f"${valor_total:,.2f}")
with c2:
    st.metric("Ganancia/Pérdida", f"{'+' if pnl_total >= 0 else '-'}${abs(pnl_total):,.2f}", delta=f"{pnl_total:,.2f}")
with c3:
    st.metric("Retorno %", f"{retorno_pct:+.2f}%", delta=f"{retorno_pct:+.2f}%")
with c4:
    sharpe = m.get("sharpe_ratio")
    st.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe is not None else "—")

st.caption(f"Actualizado: {doc.get('fecha_fin', '—')}")

# ────────────────────────────────────────────────────────────────
# Tabla de posiciones + distribución del portafolio
# ────────────────────────────────────────────────────────────────
col_tabla, col_pie = st.columns(2)

with col_tabla:
    st.markdown('<div class="card-title">Posiciones (5 sleeves — 1 por ticker)</div>', unsafe_allow_html=True)
    if posiciones:
        df_pos = pd.DataFrame(posiciones)
        columnas = ["ticker", "cantidad", "precio_entrada", "precio_actual", "pnl_usd"]
        columnas_disponibles = [c for c in columnas if c in df_pos.columns]
        df_mostrar = df_pos[columnas_disponibles].rename(
            columns={
                "ticker": "Activo",
                "cantidad": "Cantidad",
                "precio_entrada": "Precio Entrada",
                "precio_actual": "Precio Actual",
                "pnl_usd": "P&L (USD)",
            }
        )
        st.dataframe(
            df_mostrar,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Precio Entrada": st.column_config.NumberColumn(format="$%.2f"),
                "Precio Actual": st.column_config.NumberColumn(format="$%.2f"),
                "P&L (USD)": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
    else:
        st.info("Sin posiciones registradas.")

with col_pie:
    st.markdown('<div class="card-title">Distribución del Portafolio</div>', unsafe_allow_html=True)
    if posiciones:
        fig_pie = px.pie(
            names=[p["ticker"] for p in posiciones],
            values=[p.get("valor_actual", 0) for p in posiciones],
            color_discrete_sequence=DISTRIBUCION_COLORES,
        )
        fig_pie.update_traces(textinfo="label+percent", marker=dict(line=dict(color=COLOR_BG, width=2)))
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
            font=dict(color=COLOR_TEXT),
            legend=dict(orientation="h", y=-0.1),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Sin datos de distribución.")

# ────────────────────────────────────────────────────────────────
# Curva de equity
# ────────────────────────────────────────────────────────────────
st.markdown('<div class="card-title">Curva de Equity</div>', unsafe_allow_html=True)

curva = doc.get("equity_curve", [])
if curva:
    fig_equity = go.Figure()
    fig_equity.add_trace(
        go.Scatter(
            x=[p["fecha"] for p in curva],
            y=[p["valor"] for p in curva],
            mode="lines",
            name="Equity",
            line=dict(color="#2563eb", width=2),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.1)",
        )
    )
    fig_equity.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT),
        xaxis=dict(title="Fecha", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Valor (USD)", gridcolor="rgba(255,255,255,0.1)"),
        margin=dict(l=50, r=30, t=10, b=40),
        height=320,
    )
    st.plotly_chart(fig_equity, use_container_width=True)
else:
    st.info("Sin curva de equity para esta combinación.")

# ────────────────────────────────────────────────────────────────
# Pie de página
# ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="footer-nota">
        Datos reales vía MongoDB Atlas · Backtesting de Portafolio Diversificado ·
        Actualizado: {doc.get('fecha_fin', '—')}
    </div>
    """,
    unsafe_allow_html=True,
)
