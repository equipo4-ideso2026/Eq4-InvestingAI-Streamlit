"""
Página Backtesting — equivalente a /api/backtests + Módulo M11 (reporte
técnico completo: Sharpe, Sortino, Max Drawdown, Win Rate, trades).
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from db import get_db, COL_BACKTESTS, MODELOS_TODOS, PERFILES_RIESGO, escalar_backtest

st.set_page_config(page_title="Backtesting — InvestAI", page_icon="📋", layout="wide")

if not st.session_state.get("logueado"):
    st.warning("Inicia sesión desde la página principal.")
    st.stop()

st.title("📋 Backtesting de Portafolio")
st.caption("Comisión 0.10% + Slippage 0.05% aplicados en cada trade · Datos reales vía MongoDB Atlas")

db = get_db()

c1, c2 = st.columns(2)
with c1:
    modelo = st.selectbox("Modelo", MODELOS_TODOS, index=1, key="m11_modelo")
with c2:
    perfil = st.selectbox("Perfil de riesgo", PERFILES_RIESGO, index=1, key="m11_perfil")

doc = db[COL_BACKTESTS].find_one({"modelo": modelo, "perfil_riesgo": perfil}, {"_id": 0})

if not doc:
    st.error(f"No hay backtest para modelo='{modelo}', perfil='{perfil}'. Ejecuta el Notebook 10.")
    st.stop()

capital = st.session_state.get("capital", 100_000.0)
if capital != doc.get("capital_base", 100_000.0):
    st.caption(f"💰 Montos en USD reescalados de ${doc.get('capital_base', 100000):,.0f} a ${capital:,.0f} — las métricas en % (Sharpe, Win Rate, etc.) no cambian, son invariantes de escala")
doc = escalar_backtest(doc, capital)

m = doc.get("metricas", {})

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Retorno Total", f"{m.get('total_return_pct', 0):+.2f}%")
c2.metric("Sharpe Ratio", f"{m.get('sharpe_ratio', 0):.2f}")
c3.metric("Max Drawdown", f"{m.get('max_drawdown_pct', 0):.2f}%")
c4.metric("Win Rate", f"{m.get('win_rate_pct', 0):.0f}%")
c5.metric("Profit Factor", f"{m.get('profit_factor', 0):.2f}")
c6.metric("Total Trades", f"{m.get('total_trades', 0)}")

st.markdown("---")

st.markdown("#### Curva de Equity")
equity = doc.get("equity_curve", [])
if equity:
    df_eq = pd.DataFrame(equity)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_eq["fecha"], y=df_eq["valor"], mode="lines", fill="tozeroy",
                              line=dict(color="#2563eb"), name="Equity"))
    fig.add_hline(y=doc.get("capital_base", 100000), line_dash="dash", line_color="gray", annotation_text="Capital inicial")
    fig.update_layout(template="plotly_dark", height=380, margin=dict(l=40, r=40, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)

trades = doc.get("trades", [])
col_hist, col_tabla = st.columns([1, 1.4])

with col_hist:
    st.markdown("#### Histograma de Retornos por Trade")
    if trades:
        retornos = [tr["retorno_pct"] for tr in trades]
        fig_h = go.Figure(data=[go.Histogram(x=retornos, nbinsx=15,
                                               marker_color=["#26A69A" if r >= 0 else "#EF5350" for r in retornos])])
        fig_h.update_layout(template="plotly_dark", height=320, margin=dict(l=40, r=40, t=20, b=40),
                             xaxis_title="Retorno por trade (%)", yaxis_title="Frecuencia")
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("Sin trades registrados en este backtest.")

with col_tabla:
    st.markdown(f"#### Últimos Trades ({len(trades)} total)")
    if trades:
        df_tr = pd.DataFrame(trades[-20:][::-1])
        st.dataframe(
            df_tr[["numero", "ticker", "fecha_entrada", "fecha_salida", "retorno_pct", "duracion_dias"]]
            .rename(columns={"numero": "#", "ticker": "Ticker", "fecha_entrada": "Entrada",
                              "fecha_salida": "Salida", "retorno_pct": "Retorno %", "duracion_dias": "Días"}),
            use_container_width=True, hide_index=True,
            column_config={"Retorno %": st.column_config.NumberColumn(format="%+.2f%%")}
        )
        csv = pd.DataFrame(trades).to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Exportar CSV completo", csv, file_name=f"trades_{modelo}_{perfil}.csv", mime="text/csv")
    else:
        st.info("Sin trades para mostrar.")

st.caption(f"Datos reales vía MongoDB Atlas · Comisión 0.10% + Slippage 0.05% · Actualizado: {doc.get('fecha_fin', '—')}")
