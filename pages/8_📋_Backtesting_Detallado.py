"""
pages/8_📋_Backtesting_Detallado.py — Backtesting de Portafolio Diversificado
================================================================================
Reemplaza el módulo M11 del index.html original: métricas técnicas de
rendimiento, curva de equity, histograma de retornos por trade y log
completo de operaciones con exportación a CSV.

Consulta `db.py` directamente contra MongoDB Atlas (colección
`backtests`, poblada por el Notebook 10), sin pasar por FastAPI/ngrok.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from db import get_backtest_report

# ────────────────────────────────────────────────────────────────
# Configuración de página
# ────────────────────────────────────────────────────────────────
try:
    st.set_page_config(page_title="Backtesting Detallado · InvestAI", page_icon="📋", layout="wide")
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
        <h2>📋 Backtesting de Portafolio Diversificado</h2>
        <p>Simulación día a día sobre datos reales · Comisión: 0.10% · Slippage: 0.05%</p>
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

m = doc.get("metricas", {})
trades = doc.get("trades", [])

st.caption(
    f"Período: {doc.get('fecha_inicio', '—')} → {doc.get('fecha_fin', '—')} · "
    f"Comisión: {(doc.get('comision_pct') or 0) * 100:.2f}% · "
    f"Slippage: {(doc.get('slippage_pct') or 0) * 100:.2f}%"
)

# ────────────────────────────────────────────────────────────────
# 6 tarjetas de métricas técnicas
# ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    tr = m.get("total_return_pct")
    st.metric("Total Return", f"{tr:+.1f}%" if tr is not None else "—")
with c2:
    sh = m.get("sharpe_ratio")
    st.metric("Sharpe Ratio", f"{sh:.2f}" if sh is not None else "—")
with c3:
    dd = m.get("max_drawdown_pct")
    st.metric("Max Drawdown", f"{dd:.1f}%" if dd is not None else "—")
    st.markdown(f"<div style='color:{COLOR_NEGATIVE}; font-size:12px; margin-top:-12px;'>Riesgo</div>", unsafe_allow_html=True)
with c4:
    wr = m.get("win_rate_pct")
    st.metric("Win Rate", f"{wr:.1f}%" if wr is not None else "—")
with c5:
    pf = m.get("profit_factor")
    st.metric("Profit Factor", f"{pf:.2f}" if pf is not None else "—")
with c6:
    st.metric("Total Trades", m.get("total_trades", "—"))

# ────────────────────────────────────────────────────────────────
# Curva de equity + tabla de métricas detalladas
# ────────────────────────────────────────────────────────────────
col_equity, col_metricas = st.columns([2, 1])

with col_equity:
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
            height=380,
        )
        st.plotly_chart(fig_equity, use_container_width=True)
    else:
        st.info("Sin curva de equity para esta combinación.")

with col_metricas:
    st.markdown('<div class="card-title">Métricas Detalladas</div>', unsafe_allow_html=True)

    def _fmt_pct(v, signo=False):
        if v is None:
            return "—"
        return f"{'+' if signo and v >= 0 else ''}{v:.2f}%"

    filas = [
        ("Total Return", _fmt_pct(m.get("total_return_pct"), signo=True)),
        ("Ann. Return", _fmt_pct(m.get("retorno_anualizado_pct"), signo=True)),
        ("Sharpe Ratio", f"{m.get('sharpe_ratio'):.2f}" if m.get("sharpe_ratio") is not None else "—"),
        ("Sortino Ratio", f"{m.get('sortino_ratio'):.2f}" if m.get("sortino_ratio") is not None else "—"),
        ("Max Drawdown", _fmt_pct(m.get("max_drawdown_pct"))),
        ("Win Rate", _fmt_pct(m.get("win_rate_pct"))),
        ("Profit Factor", f"{m.get('profit_factor'):.2f}" if m.get("profit_factor") is not None else "—"),
        ("Total Trades", str(m.get("total_trades", "—"))),
    ]
    filas_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in filas)
    st.markdown(
        f"""
        <table class="params-table" style="width:100%; font-size:12px; border-collapse:collapse;">
        {filas_html}
        </table>
        <style>
        .params-table tr {{ border-bottom: 1px dashed rgba(255,255,255,0.1); }}
        .params-table td {{ padding: 8px 4px; color:{COLOR_MUTED}; }}
        .params-table td:first-child {{ font-weight:600; color:{COLOR_TEXT}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ────────────────────────────────────────────────────────────────
# Histograma de retornos + log de últimos 20 trades
# ────────────────────────────────────────────────────────────────
col_hist, col_trades = st.columns(2)

with col_hist:
    st.markdown('<div class="card-title">Histograma de Retornos por Trade</div>', unsafe_allow_html=True)
    retornos = [t.get("retorno_pct") for t in trades if t.get("retorno_pct") is not None]

    if retornos:
        n_bins = 12
        min_r, max_r = min(retornos), max(retornos)
        ancho = (max_r - min_r) / n_bins or 1
        conteo, bordes = np.histogram(retornos, bins=n_bins, range=(min_r, min_r + ancho * n_bins))
        etiquetas = [f"{bordes[i]:.1f}%" for i in range(n_bins)]
        colores = [COLOR_NEGATIVE if bordes[i] < 0 else COLOR_POSITIVE for i in range(n_bins)]

        fig_hist = go.Figure(
            data=go.Bar(x=etiquetas, y=conteo, marker_color=colores, name="Frecuencia")
        )
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
            font=dict(color=COLOR_TEXT),
            xaxis=dict(title="Retorno por trade", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Frecuencia", gridcolor="rgba(255,255,255,0.1)"),
            margin=dict(l=40, r=20, t=10, b=60),
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No hay trades cerrados para graficar.")

with col_trades:
    st.markdown('<div class="card-title">Últimos 20 Trades</div>', unsafe_allow_html=True)
    ultimos = list(reversed(trades[-20:])) if trades else []

    if ultimos:
        df_trades = pd.DataFrame(ultimos)
        columnas = ["numero", "fecha_entrada", "fecha_salida", "ticker", "retorno_pct", "duracion_dias"]
        columnas_disponibles = [c for c in columnas if c in df_trades.columns]
        df_mostrar = df_trades[columnas_disponibles].rename(
            columns={
                "numero": "#",
                "fecha_entrada": "Entrada",
                "fecha_salida": "Salida",
                "ticker": "Activo",
                "retorno_pct": "Retorno %",
                "duracion_dias": "Duración (días)",
            }
        )
        st.dataframe(
            df_mostrar,
            use_container_width=True,
            hide_index=True,
            height=300,
            column_config={"Retorno %": st.column_config.NumberColumn(format="%.2f%%")},
        )
    else:
        st.info("Sin trades cerrados en esta combinación.")

    # ── Botón de descarga: exporta el historial completo de trades ──
    if trades:
        df_export = pd.DataFrame(trades)
        columnas_export = [
            "numero", "ticker", "fecha_entrada", "fecha_salida",
            "precio_entrada", "precio_salida", "retorno_pct", "duracion_dias",
        ]
        columnas_export_disponibles = [c for c in columnas_export if c in df_export.columns]
        csv_bytes = df_export[columnas_export_disponibles].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Exportar CSV",
            data=csv_bytes,
            file_name=f"backtesting_trades_{modelo}_{perfil}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.button("⬇ Exportar CSV", disabled=True, use_container_width=True, help="No hay trades cargados para exportar.")

# ────────────────────────────────────────────────────────────────
# Pie de página
# ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="footer-nota">
        Datos reales vía MongoDB Atlas · Comisión 0.10% + Slippage 0.05% aplicados en cada trade ·
        Actualizado: {doc.get('fecha_fin', '—')}
    </div>
    """,
    unsafe_allow_html=True,
)
