"""
pages/9_🏆_Dashboard_Central.py — InvestAI Dashboard Central Integrado
=========================================================================
Reemplaza el módulo M9 del index.html original: ensamblado en vivo de
los 5 clasificadores (SVC, LSTM, BiLSTM, GRU, SimpleRNN) para el ticker
global, con votación mayoritaria, gráfico OHLCV, curva de equity e
histograma de retornos diarios de la estrategia vigente.

Consulta `db.py` directamente contra MongoDB Atlas, sin pasar por
FastAPI/ngrok.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from db import get_backtest_report, get_mercado_data, get_rnn_data, get_svc_data

# ────────────────────────────────────────────────────────────────
# Configuración de página
# ────────────────────────────────────────────────────────────────
try:
    st.set_page_config(page_title="Dashboard Central · InvestAI", page_icon="🏆", layout="wide")
except Exception:
    pass

# ────────────────────────────────────────────────────────────────
# Paleta oscura fiel al index.html original
# ────────────────────────────────────────────────────────────────
COLOR_BG = "#0f172a"
COLOR_CARD = "#1e293b"
COLOR_CARD_DARK = "#111827"
COLOR_TEXT = "#f8fafc"
COLOR_MUTED = "#94a3b8"
COLOR_MUTED_DIM = "#64748b"
COLOR_BUY = "#26A69A"
COLOR_HOLD = "#FFC107"
COLOR_SELL = "#EF5350"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {COLOR_BG}; color: {COLOR_TEXT}; }}
    .modulo-header p {{ color: {COLOR_MUTED}; font-size: 13px; margin-top: -8px; }}
    .card-title {{ font-size: 14px; font-weight: 600; color: {COLOR_TEXT}; margin-bottom: 12px; }}
    .top-bar {{
        background-color: {COLOR_CARD_DARK};
        border-radius: 10px;
        padding: 14px 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 16px;
        font-size: 13px;
    }}
    .signal-row {{
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.1); font-size: 13px;
    }}
    .signal-badge {{
        padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: 700;
    }}
    .ensemble-panel {{
        background-color: rgba(255,255,255,0.05);
        border: 2px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        margin-top: 14px;
    }}
    .ensemble-label {{ font-size: 11px; color: {COLOR_MUTED}; margin-bottom: 8px; }}
    .ensemble-value {{ font-size: 28px; font-weight: 700; }}
    .ensemble-sub {{ font-size: 12px; color: {COLOR_MUTED}; margin-top: 8px; }}
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
    """
    <div class="modulo-header">
        <h2>🏆 InvestAI — Dashboard Central Integrado</h2>
        <p>Ensamblado de Modelos IA (SVC + LSTM/BiLSTM/GRU/SimpleRNN) · Datos reales de MongoDB Atlas</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "ejecutar_analisis" not in st.session_state:
    st.session_state["ejecutar_analisis"] = True  # auto-ejecuta en el primer render

# ────────────────────────────────────────────────────────────────
# Encabezado integrado: ticker, precio, variación + botón de ejecución
# ────────────────────────────────────────────────────────────────
df_mercado = get_mercado_data(ticker, dias=90)

col_bar, col_btn = st.columns([4, 1])
with col_bar:
    if not df_mercado.empty:
        ultimo_precio = float(df_mercado["close"].iloc[-1])
        primer_precio = float(df_mercado["close"].iloc[0])
        cambio_pct = ((ultimo_precio - primer_precio) / primer_precio) * 100 if primer_precio else 0.0
        color_cambio = COLOR_BUY if cambio_pct >= 0 else COLOR_SELL
        precio_txt = f"${ultimo_precio:,.2f}"
        cambio_txt = f"{cambio_pct:+.2f}%"
        fecha_txt = str(df_mercado["fecha"].iloc[-1])[:10]
    else:
        precio_txt, cambio_txt, color_cambio, fecha_txt = "Sin datos", "—", COLOR_MUTED, "—"

    st.markdown(
        f"""
        <div class="top-bar">
            <span>Ticker: <strong>{ticker}</strong></span>
            <span>Precio: <strong>{precio_txt}</strong></span>
            <span>Variación: <strong style="color:{color_cambio};">{cambio_txt}</strong></span>
            <span>Última actualización: <strong>{fecha_txt}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_btn:
    if st.button("▶ Ejecutar Análisis Integrado", use_container_width=True, type="primary"):
        st.session_state["ejecutar_analisis"] = True
        st.rerun()

# ────────────────────────────────────────────────────────────────
# Layout principal: gráfico OHLCV (izq) + consola de señales IA (der)
# ────────────────────────────────────────────────────────────────
col_chart, col_console = st.columns([2, 1])

with col_chart:
    st.markdown('<div class="card-title">Gráfico Central Integrado</div>', unsafe_allow_html=True)
    if not df_mercado.empty:
        fig = go.Figure(
            data=go.Candlestick(
                x=df_mercado["fecha"],
                open=df_mercado["open"],
                high=df_mercado["high"],
                low=df_mercado["low"],
                close=df_mercado["close"],
                increasing_line_color=COLOR_BUY,
                decreasing_line_color=COLOR_SELL,
                name=ticker,
            )
        )
        fig.update_layout(
            title=f"{ticker} — Análisis Integrado",
            template="plotly_dark",
            paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
            font=dict(color=COLOR_TEXT),
            xaxis=dict(rangeslider=dict(visible=False)),
            hovermode="x unified",
            margin=dict(l=60, r=60, t=40, b=40),
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"⚠ No hay datos de mercado para **{ticker}**.")

with col_console:
    st.markdown('<div class="card-title">Consola de Señales IA</div>', unsafe_allow_html=True)

    MODELOS_CLASIFICADORES = ["SVC", "LSTM", "BiLSTM", "GRU", "SimpleRNN"]
    resultados = []
    for modelo in MODELOS_CLASIFICADORES:
        datos = get_svc_data(ticker) if modelo == "SVC" else get_rnn_data(ticker, modelo)
        resultados.append({"modelo": modelo, "datos": datos})

    filas_html = ""
    votos_buy = votos_sell = 0
    accuracies = []

    for r in resultados:
        modelo, datos = r["modelo"], r["datos"]
        if not datos:
            filas_html += f"""
                <div class="signal-row">
                    <span>{modelo}</span>
                    <span style="color:{COLOR_MUTED_DIM}; font-size:11px;">Sin datos</span>
                </div>"""
            continue

        senal = (datos.get("prediccion", {}).get("senal") or "—").upper()
        confianza = datos.get("prediccion", {}).get("confianza")
        accuracy = datos.get("metricas", {}).get("accuracy")

        if senal == "BUY":
            votos_buy += 1
        elif senal == "SELL":
            votos_sell += 1
        if accuracy is not None:
            accuracies.append(accuracy)

        color = COLOR_BUY if senal == "BUY" else (COLOR_SELL if senal == "SELL" else COLOR_HOLD)
        bg = (
            "rgba(38, 166, 154, 0.2)" if senal == "BUY"
            else "rgba(239, 83, 80, 0.2)" if senal == "SELL"
            else "rgba(255, 193, 7, 0.2)"
        )
        conf_txt = f"{confianza * 100:.0f}%" if confianza is not None else "—"

        filas_html += f"""
            <div class="signal-row">
                <span>{modelo}</span>
                <span class="signal-badge" style="background-color:{bg}; color:{color};">{senal}</span>
                <span style="color:{COLOR_MUTED_DIM}; font-size:11px;">{conf_txt}</span>
            </div>"""

    st.markdown(
        f'<div style="background-color:{COLOR_CARD}; border-radius:8px; padding:12px; max-height:300px; overflow-y:auto;">{filas_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Señal de ensamblado: votación mayoritaria ────────────────
    votos_validos = votos_buy + votos_sell
    if votos_validos > 0:
        if votos_buy > votos_sell:
            senal_final = "BUY"
        elif votos_sell > votos_buy:
            senal_final = "SELL"
        else:
            senal_final = "HOLD"
    else:
        senal_final = "HOLD"

    color_final = COLOR_BUY if senal_final == "BUY" else (COLOR_SELL if senal_final == "SELL" else COLOR_HOLD)
    acc_prom = f"{(sum(accuracies) / len(accuracies)) * 100:.1f}%" if accuracies else "—"

    st.markdown(
        f"""
        <div class="ensemble-panel">
            <div class="ensemble-label">SEÑAL ENSAMBLADO ({votos_validos}/5 votos válidos):</div>
            <div class="ensemble-value" style="color:{color_final};">{senal_final}</div>
            <div class="ensemble-sub">Accuracy promedio de los modelos: <strong>{acc_prom}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ────────────────────────────────────────────────────────────────
# Sección inferior: equity curve + histograma de retornos diarios
# ────────────────────────────────────────────────────────────────
st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

MODELO_BACKTEST_DEFECTO = "SVC"
PERFIL_BACKTEST_DEFECTO = "moderado"
backtest = get_backtest_report(modelo=MODELO_BACKTEST_DEFECTO, perfil_riesgo=PERFIL_BACKTEST_DEFECTO)

col_equity, col_hist = st.columns(2)

curva = backtest.get("equity_curve", []) if backtest else []

with col_equity:
    st.markdown(
        f'<div class="card-title">Curva de Equity ({MODELO_BACKTEST_DEFECTO} · {PERFIL_BACKTEST_DEFECTO})</div>',
        unsafe_allow_html=True,
    )
    if curva:
        fig_equity = go.Figure(
            data=go.Scatter(
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
            xaxis=dict(title="Fecha", gridcolor="rgba(255,255,255,0.1)", nticks=12),
            yaxis=dict(title="Valor (USD)", gridcolor="rgba(255,255,255,0.1)"),
            margin=dict(l=50, r=20, t=10, b=40),
            height=280,
        )
        st.plotly_chart(fig_equity, use_container_width=True)
    else:
        st.info("Sin curva de equity disponible para la estrategia por defecto.")

with col_hist:
    st.markdown('<div class="card-title">Histograma de Retornos Diarios</div>', unsafe_allow_html=True)
    if len(curva) > 1:
        valores = [p["valor"] for p in curva]
        retornos_pct = [
            ((valores[i] - valores[i - 1]) / valores[i - 1]) * 100
            for i in range(1, len(valores))
            if valores[i - 1] != 0
        ]
        if retornos_pct:
            n_bins = 20
            min_r, max_r = min(retornos_pct), max(retornos_pct)
            ancho = (max_r - min_r) / n_bins or 1
            conteo, bordes = np.histogram(retornos_pct, bins=n_bins, range=(min_r, min_r + ancho * n_bins))
            etiquetas = [f"{bordes[i]:.1f}%" for i in range(n_bins)]
            colores = [COLOR_SELL if bordes[i] < 0 else COLOR_BUY for i in range(n_bins)]

            fig_hist = go.Figure(data=go.Bar(x=etiquetas, y=conteo, marker_color=colores))
            fig_hist.update_layout(
                template="plotly_dark",
                paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
                font=dict(color=COLOR_TEXT),
                xaxis=dict(title="Retorno diario", gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(title="Frecuencia", gridcolor="rgba(255,255,255,0.1)"),
                margin=dict(l=40, r=20, t=10, b=60),
                height=280,
                showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No hay suficientes puntos para calcular retornos diarios.")
    else:
        st.info("Sin datos suficientes en la curva de equity.")

# ────────────────────────────────────────────────────────────────
# Pie de página
# ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="footer-nota">
        Vista agregada de todos los módulos · Datos reales vía MongoDB Atlas ·
        Actualizado: {fecha_txt}
    </div>
    """,
    unsafe_allow_html=True,
)
