"""
pages/11_🚀_Terminal_Broker.py — Envío de Señales al Broker (Paper Trading)
==============================================================================
Reemplaza el módulo M8 del index.html original: formulario de orden
manual con preview de costos en vivo, flujo de confirmación explícita, y
libro de órdenes leído/escrito directamente en MongoDB Atlas a través de
`db.py` (colección `ordenes_simuladas`).
"""

import pandas as pd
import streamlit as st

from db import (
    COMISION_ORDEN_PCT,
    get_historial_ordenes,
    get_mercado_data,
    registrar_orden_manual,
)

# ────────────────────────────────────────────────────────────────
# Configuración de página
# ────────────────────────────────────────────────────────────────
try:
    st.set_page_config(page_title="Terminal Broker · InvestAI", page_icon="🚀", layout="wide")
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
COLOR_WARNING = "#FFC107"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {COLOR_BG}; color: {COLOR_TEXT}; }}
    .modulo-header p {{ color: {COLOR_MUTED}; font-size: 13px; margin-top: -8px; }}
    .card-title {{ font-size: 14px; font-weight: 600; color: {COLOR_TEXT}; margin-bottom: 12px; }}
    .status-bar {{
        background-color: {COLOR_CARD_DARK};
        border-radius: 10px;
        padding: 12px 16px;
        display: flex;
        gap: 20px;
        flex-wrap: wrap;
        font-size: 12px;
        color: {COLOR_MUTED};
        margin-bottom: 12px;
    }}
    .warning-banner {{
        background-color: rgba(255, 193, 7, 0.1);
        border: 1px solid rgba(255, 193, 7, 0.3);
        border-radius: 10px;
        padding: 12px 16px;
        color: {COLOR_WARNING};
        font-size: 13px;
        margin-bottom: 16px;
    }}
    .preview-box {{
        background-color: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
        font-size: 13px;
    }}
    .preview-row {{
        display: flex; justify-content: space-between; padding: 6px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }}
    .preview-row.total {{ font-weight: 700; border-bottom: none; }}
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
ticker_global = st.session_state.get("ticker_global", "FSM")

st.markdown(
    """
    <div class="modulo-header">
        <h2>🚀 Envío de Señales al Broker</h2>
        <p>Formulario de orden + historial de ejecuciones (Paper Trading)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="status-bar">
        <div>● Conectado a Interactive Brokers (simulado)</div>
        <div>Latencia TWS: 12ms</div>
        <div>Cuenta: DU123456 (Paper Trading)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="warning-banner">
        ⚠️ PAPER TRADING — Órdenes simuladas sin dinero real.
    </div>
    """,
    unsafe_allow_html=True,
)

TICKERS_INFO = {
    "FSM": "FSM — Fortuna Silver",
    "VOLCABC1.LM": "VOLCABC1.LM — Volcan",
    "ABX.TO": "ABX.TO — Barrick Gold",
    "BVN": "BVN — Buenaventura",
    "BHP": "BHP — BHP Billiton",
}

# ────────────────────────────────────────────────────────────────
# Layout: formulario de orden (izq) + historial (der)
# ────────────────────────────────────────────────────────────────
col_form, col_historial = st.columns(2)

with col_form:
    st.markdown('<div class="card-title">Nueva Orden</div>', unsafe_allow_html=True)

    tickers_lista = list(TICKERS_INFO.keys())
    idx_default = tickers_lista.index(ticker_global) if ticker_global in tickers_lista else 0

    ticker_orden = st.selectbox(
        "Ticker", options=tickers_lista, index=idx_default,
        format_func=lambda t: TICKERS_INFO.get(t, t),
    )
    direccion = st.selectbox("Dirección", options=["BUY", "SELL"])
    tipo_orden = st.selectbox("Tipo de Orden", options=["MARKET", "LIMIT"])
    cantidad = st.number_input("Cantidad (acciones)", min_value=1, value=10, step=1)

    # ── Preview de costos dinámico (precio real más reciente) ────────
    df_precio = get_mercado_data(ticker_orden, dias=5)
    precio_est = float(df_precio["close"].iloc[-1]) if not df_precio.empty else None

    if precio_est is not None:
        subtotal = precio_est * cantidad
        comision = subtotal * COMISION_ORDEN_PCT
        total_neto = subtotal + comision if direccion == "BUY" else subtotal - comision

        st.markdown(
            f"""
            <div class="preview-box">
                <div class="preview-row"><span>Precio Est.:</span><strong>${precio_est:,.2f}</strong></div>
                <div class="preview-row"><span>Costo Total Est.:</span><strong>${subtotal:,.2f}</strong></div>
                <div class="preview-row"><span>Comisión Est. (0.10%):</span><strong>${comision:,.2f}</strong></div>
                <div class="preview-row total"><span>Total {'con' if direccion == 'BUY' else 'neto de'} Comisión:</span><strong>${total_neto:,.2f}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"⚠ No hay precio reciente disponible para **{ticker_orden}**.")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    confirmar_check = st.checkbox("Confirmo que he revisado los detalles de esta orden")
    enviar = st.button(
        "CONFIRMAR Y ENVIAR",
        type="primary",
        use_container_width=True,
        disabled=not confirmar_check or precio_est is None,
    )

    if enviar:
        resultado = registrar_orden_manual(
            ticker=ticker_orden,
            direccion=direccion,
            cantidad=cantidad,
            tipo_orden=tipo_orden,
        )
        if resultado:
            st.success(
                f"✓ Orden registrada: {direccion} {cantidad} {ticker_orden} @ ${resultado['precio_ejecucion']:.2f} "
                f"(total: ${resultado['total']:,.2f})"
            )
            st.rerun()
        else:
            st.error("✕ No se pudo registrar la orden. Verifica la conexión con MongoDB Atlas y los datos del ticker.")

with col_historial:
    st.markdown('<div class="card-title">Historial de Órdenes</div>', unsafe_allow_html=True)
    ordenes = get_historial_ordenes()

    if ordenes:
        df_ordenes = pd.DataFrame(ordenes)
        columnas = ["fecha", "ticker", "direccion", "tipo_orden", "cantidad", "precio_ejecucion", "total"]
        columnas_disponibles = [c for c in columnas if c in df_ordenes.columns]
        df_mostrar = df_ordenes[columnas_disponibles].rename(
            columns={
                "fecha": "Hora",
                "ticker": "Ticker",
                "direccion": "Dir.",
                "tipo_orden": "Tipo",
                "cantidad": "Qty",
                "precio_ejecucion": "Precio",
                "total": "Total",
            }
        )
        st.dataframe(
            df_mostrar,
            use_container_width=True,
            hide_index=True,
            height=420,
            column_config={
                "Precio": st.column_config.NumberColumn(format="$%.2f"),
                "Total": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
    else:
        st.info("Sin órdenes registradas todavía. Envía la primera orden desde el formulario.")

# ────────────────────────────────────────────────────────────────
# Pie de página
# ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer-nota">
        Libro de órdenes manual · Precio real de ejecución vía MongoDB Atlas
    </div>
    """,
    unsafe_allow_html=True,
)
