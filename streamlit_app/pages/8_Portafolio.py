"""
Página Portafolio — equivalente a /api/backtests + Módulo M7 (vista resumida).
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from db import get_db, COL_BACKTESTS, MODELOS_TODOS, PERFILES_RIESGO, escalar_backtest

st.set_page_config(page_title="Portafolio — InvestAI", page_icon="💼", layout="wide")

if not st.session_state.get("logueado"):
    st.warning("Inicia sesión desde la página principal.")
    st.stop()

st.title("💼 Portafolio")
st.caption("Resultado del backtest de portafolio diversificado (Notebook 10)")

db = get_db()

c1, c2 = st.columns(2)
with c1:
    modelo = st.selectbox("Modelo (señales)", MODELOS_TODOS, index=1, key="m7_modelo")
with c2:
    perfil = st.selectbox("Perfil de riesgo", PERFILES_RIESGO, index=1, key="m7_perfil")

doc = db[COL_BACKTESTS].find_one({"modelo": modelo, "perfil_riesgo": perfil}, {"_id": 0})

if not doc:
    st.error(f"No hay backtest para modelo='{modelo}', perfil='{perfil}'. Ejecuta el Notebook 10 (Backtesting).")
    st.stop()

capital = st.session_state.get("capital", 100_000.0)
if capital != doc.get("capital_base", 100_000.0):
    st.caption(f"💰 Montos reescalados de ${doc.get('capital_base', 100000):,.0f} (backtest original) a ${capital:,.0f} (tu capital elegido)")
doc = escalar_backtest(doc, capital)

posiciones = doc.get("posiciones_finales", [])
m = doc.get("metricas", {})

valor_total = sum(p["valor_actual"] for p in posiciones)
pnl_total = sum(p["pnl_usd"] for p in posiciones)
capital_inicial = sum(p["capital_inicial_sleeve"] for p in posiciones)
retorno_pct = (pnl_total / capital_inicial * 100) if capital_inicial > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Valor Total", f"${valor_total:,.2f}")
c2.metric("Ganancia/Pérdida", f"{'+' if pnl_total >= 0 else '-'}${abs(pnl_total):,.2f}")
c3.metric("Retorno", f"{retorno_pct:+.2f}%")
c4.metric("Sharpe Ratio", f"{m.get('sharpe_ratio', 0):.2f}")

st.markdown("---")

col_izq, col_der = st.columns([1.6, 1])
with col_izq:
    st.markdown("#### Posiciones")
    df_pos = pd.DataFrame(posiciones)
    if not df_pos.empty:
        df_pos_display = df_pos.rename(columns={
            "ticker": "Ticker", "cantidad": "Cantidad", "precio_entrada": "Precio Entrada",
            "precio_actual": "Precio Actual", "pnl_usd": "P&L (USD)"
        })
        st.dataframe(
            df_pos_display[["Ticker", "Cantidad", "Precio Entrada", "Precio Actual", "P&L (USD)"]],
            use_container_width=True, hide_index=True,
            column_config={
                "Precio Entrada": st.column_config.NumberColumn(format="$%.2f"),
                "Precio Actual": st.column_config.NumberColumn(format="$%.2f"),
                "P&L (USD)": st.column_config.NumberColumn(format="$%.2f"),
            }
        )

with col_der:
    st.markdown("#### Distribución (valor actual)")
    if posiciones:
        fig = go.Figure(data=[go.Pie(
            labels=[p["ticker"] for p in posiciones],
            values=[p["valor_actual"] for p in posiciones], hole=0.5
        )])
        fig.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Curva de Equity")
equity = doc.get("equity_curve", [])
if equity:
    df_eq = pd.DataFrame(equity)
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=df_eq["fecha"], y=df_eq["valor"], mode="lines", fill="tozeroy",
                                 line=dict(color="#2563eb"), name=f"Portafolio ({modelo}/{perfil})"))
    fig_eq.update_layout(template="plotly_dark", height=350, margin=dict(l=40, r=40, t=20, b=40))
    st.plotly_chart(fig_eq, use_container_width=True)

st.caption(f"Datos reales vía MongoDB Atlas · Backtesting de Portafolio Diversificado · Actualizado: {doc.get('fecha_fin', '—')}")
