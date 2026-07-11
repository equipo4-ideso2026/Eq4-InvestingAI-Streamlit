"""
pages/3_🧠_Clasificadores_RNN.py — Clasificadores RNN (LSTM · BiLSTM · GRU · SimpleRNN)
==========================================================================================
Reemplaza el módulo M3 del index.html original: semáforo de señal
BUY/HOLD/SELL, métricas del modelo, matriz de confusión 2x2 y curvas de
entrenamiento (Loss/Accuracy train vs. validación) para diagnosticar
sobreajuste.

Lee el ticker desde `st.session_state.ticker_global`, la arquitectura se
elige localmente, y consulta `db.py` directamente contra MongoDB Atlas
(colección `predicciones` / `metricas_modelos`, poblada por los
Notebooks 3-6).
"""

import plotly.graph_objects as go
import streamlit as st

from db import get_rnn_data

# ────────────────────────────────────────────────────────────────
# Configuración de página
# ────────────────────────────────────────────────────────────────
try:
    st.set_page_config(page_title="Clasificadores RNN · InvestAI", page_icon="🧠", layout="wide")
except Exception:
    pass

# ────────────────────────────────────────────────────────────────
# Paleta oscura fiel al index.html original
# ────────────────────────────────────────────────────────────────
COLOR_BG = "#0f172a"
COLOR_CARD = "#1e293b"
COLOR_TEXT = "#f8fafc"
COLOR_MUTED = "#94a3b8"
COLOR_BUY = "#26A69A"
COLOR_HOLD = "#FFC107"
COLOR_SELL = "#EF5350"

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
    .signal-card {{
        background-color: {COLOR_CARD};
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }}
    .card-title {{ font-size: 14px; font-weight: 600; color: {COLOR_TEXT}; margin-bottom: 12px; }}
    .traffic-light {{ display:flex; flex-direction:column; gap:12px; align-items:center; margin: 16px 0; }}
    .signal-light {{
        width: 72px; height: 72px; border-radius: 50%;
        display:flex; align-items:center; justify-content:center;
        font-size: 12px; font-weight:700; color:white; opacity:0.2;
    }}
    .signal-light.active {{ opacity:1; box-shadow: 0 0 25px currentColor; }}
    .signal-light.buy {{ background-color:{COLOR_BUY}; }}
    .signal-light.hold {{ background-color:{COLOR_HOLD}; }}
    .signal-light.sell {{ background-color:{COLOR_SELL}; }}
    .traffic-light-label {{ font-size:11px; color:{COLOR_MUTED}; margin-top:-8px; }}
    .confianza-label {{ font-size:13px; color:{COLOR_MUTED}; margin-top:8px; }}
    .footer-nota {{
        margin-top:24px; padding-top:16px; border-top:1px solid rgba(255,255,255,0.08);
        font-size:12px; color:{COLOR_MUTED}; text-align:center;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────
# Sincronización con el ticker global + control local de arquitectura
# ────────────────────────────────────────────────────────────────
ticker = st.session_state.get("ticker_global", "FSM")

st.markdown(
    """
    <div class="modulo-header">
        <h2>🧠 Clasificadores RNN (LSTM · BiLSTM · GRU · SimpleRNN)</h2>
        <p>Clasificación binaria BUY / SELL · Ventana: 20 días</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_arch, col_periodo = st.columns(2)
with col_arch:
    arquitectura = st.selectbox("Arquitectura", options=["LSTM", "BiLSTM", "GRU", "SimpleRNN"])
with col_periodo:
    PERIODOS = {"90 días": 90, "180 días": 180, "365 días": 365}
    periodo_label = st.selectbox("Período (gráfico)", options=list(PERIODOS.keys()), index=1)
    dias = PERIODOS[periodo_label]

# ────────────────────────────────────────────────────────────────
# Datos
# ────────────────────────────────────────────────────────────────
datos = get_rnn_data(ticker, arquitectura)

if not datos:
    st.warning(
        f"⚠ No hay predicción **{arquitectura}** disponible para **{ticker}**. "
        "Verifica que el notebook correspondiente haya guardado resultados en MongoDB."
    )
    st.stop()

m = datos.get("metricas", {})
pred = datos.get("prediccion", {})
senal = (pred.get("senal") or "HOLD").upper()
confianza = pred.get("confianza")

# ────────────────────────────────────────────────────────────────
# Fila 1: gráfico de señales (izq) + semáforo y métricas (der)
# ────────────────────────────────────────────────────────────────
col_chart, col_signal = st.columns([2, 1])

with col_chart:
    historico = (datos.get("historico_senales") or [])[-dias:]

    if historico:
        fechas = [d["fecha"] for d in historico]
        precios = [d["precio"] for d in historico]
        buy_pts = [(d["fecha"], d["precio"]) for d in historico if d.get("prediccion") == "BUY"]
        sell_pts = [(d["fecha"], d["precio"]) for d in historico if d.get("prediccion") == "SELL"]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fechas, y=precios, mode="lines", name="Precio",
                line=dict(color="#42A5F5", width=2),
                hovertemplate="<b>%{x}</b><br>Precio: $%{y:.2f}<extra></extra>",
            )
        )
        if buy_pts:
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in buy_pts], y=[p[1] for p in buy_pts],
                    mode="markers", name="BUY",
                    marker=dict(symbol="triangle-up", size=11, color=COLOR_BUY),
                    hovertemplate="<b>BUY</b><br>%{x}<br>$%{y:.2f}<extra></extra>",
                )
            )
        if sell_pts:
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in sell_pts], y=[p[1] for p in sell_pts],
                    mode="markers", name="SELL",
                    marker=dict(symbol="triangle-down", size=11, color=COLOR_SELL),
                    hovertemplate="<b>SELL</b><br>%{x}<br>$%{y:.2f}<extra></extra>",
                )
            )

        fig.update_layout(
            title=f"{ticker} — Señales {arquitectura} (últimos {dias} días)",
            template="plotly_dark",
            paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
            font=dict(color=COLOR_TEXT),
            xaxis=dict(title="Fecha", gridcolor="#1e293b"),
            yaxis=dict(title="Precio (USD)", gridcolor="#1e293b"),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=60, r=30, t=50, b=40),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay histórico de señales para graficar en este período.")

with col_signal:
    conf_txt = f"{confianza * 100:.1f}%" if confianza is not None else "—"

    st.markdown(
        f"""
        <div class="signal-card">
            <div class="card-title">Señal Actual RNN ({arquitectura})</div>
            <div class="traffic-light">
                <div class="signal-light buy {'active' if senal == 'BUY' else ''}">BUY</div>
                <div class="traffic-light-label">BUY</div>
                <div class="signal-light hold {'active' if senal == 'HOLD' else ''}">HOLD</div>
                <div class="traffic-light-label">HOLD</div>
                <div class="signal-light sell {'active' if senal == 'SELL' else ''}">SELL</div>
                <div class="traffic-light-label">SELL</div>
            </div>
            <div class="confianza-label">Confianza: <strong>{conf_txt}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    mc1, mc2 = st.columns(2)
    with mc1:
        st.metric("Accuracy", f"{m.get('accuracy', 0) * 100:.1f}%" if m.get("accuracy") is not None else "—")
        st.metric("Precision", f"{m.get('precision'):.3f}" if m.get("precision") is not None else "—")
    with mc2:
        st.metric("F1-Score", f"{m.get('f1'):.3f}" if m.get("f1") is not None else "—")
        st.metric("Recall", f"{m.get('recall'):.3f}" if m.get("recall") is not None else "—")

# ────────────────────────────────────────────────────────────────
# Fila 2: matriz de confusión (izq) + curvas de entrenamiento (der)
# ────────────────────────────────────────────────────────────────
col_cm, col_curvas = st.columns(2)

with col_cm:
    st.markdown('<div class="card-title">Matriz de Confusión 2×2</div>', unsafe_allow_html=True)
    cm = m.get("matriz_confusion")
    if cm and len(cm) == 2:
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Pred SELL", "Pred BUY"],
                y=["Real SELL", "Real BUY"],
                colorscale=[[0, "#1e293b"], [0.5, "#1d4ed8"], [1, "#38bdf8"]],
                text=cm,
                texttemplate="%{text}",
                textfont=dict(size=22, color="white"),
                hovertemplate="Real: %{y}<br>Pred: %{x}<br>N: %{z}<extra></extra>",
            )
        )
        fig_cm.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
            font=dict(color=COLOR_TEXT),
            xaxis=dict(title="Predicho"), yaxis=dict(title="Real"),
            margin=dict(l=90, r=20, t=20, b=50),
            height=340,
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("Matriz de confusión no disponible.")

with col_curvas:
    st.markdown('<div class="card-title">Curvas de Entrenamiento (Loss / Accuracy)</div>', unsafe_allow_html=True)
    hist_epocas = m.get("historial_epocas") or {}
    loss = hist_epocas.get("loss", [])
    val_loss = hist_epocas.get("val_loss", [])
    accuracy = hist_epocas.get("accuracy", [])
    val_accuracy = hist_epocas.get("val_accuracy", [])

    if loss:
        epocas_x = list(range(1, len(loss) + 1))
        fig_curvas = go.Figure()
        fig_curvas.add_trace(
            go.Scatter(x=epocas_x, y=loss, mode="lines", name="Loss (train)",
                       line=dict(color="#EF5350", width=2))
        )
        if val_loss:
            fig_curvas.add_trace(
                go.Scatter(x=epocas_x, y=val_loss, mode="lines", name="Loss (val)",
                           line=dict(color="#EF5350", width=2, dash="dot"))
            )
        if accuracy:
            fig_curvas.add_trace(
                go.Scatter(x=epocas_x, y=accuracy, mode="lines", name="Accuracy (train)",
                           line=dict(color="#38bdf8", width=2), yaxis="y2")
            )
        if val_accuracy:
            fig_curvas.add_trace(
                go.Scatter(x=epocas_x, y=val_accuracy, mode="lines", name="Accuracy (val)",
                           line=dict(color="#38bdf8", width=2, dash="dot"), yaxis="y2")
            )

        fig_curvas.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
            font=dict(color=COLOR_TEXT),
            xaxis=dict(title="Época", gridcolor="#1e293b"),
            yaxis=dict(title="Loss", gridcolor="#1e293b"),
            yaxis2=dict(title="Accuracy", overlaying="y", side="right", range=[0, 1]),
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=50, r=50, t=20, b=50),
            height=340,
        )
        st.plotly_chart(fig_curvas, use_container_width=True)

        if val_loss and len(val_loss) > 3 and val_loss[-1] > val_loss[len(val_loss) // 2] * 1.15:
            st.caption("⚠ La curva de *val_loss* sube hacia el final del entrenamiento: posible señal de sobreajuste.")
    else:
        st.info("No hay historial de épocas disponible para este modelo.")

# ────────────────────────────────────────────────────────────────
# Pie de página
# ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="footer-nota">
        Datos reales de Yahoo Finance · Modelos entrenados con TensorFlow/Keras ·
        Actualizado: {datos.get('fecha_prediccion', '—')}
    </div>
    """,
    unsafe_allow_html=True,
)
