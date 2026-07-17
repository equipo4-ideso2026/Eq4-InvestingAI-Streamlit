"""
Página Broker — libro de órdenes manual. En la versión FastAPI esto era
POST/GET /api/ordenes + /api/cuenta; aquí Streamlit escribe y lee
DIRECTO en MongoDB (pymongo permite escritura, no solo lectura).
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from db import get_db, validar_ticker, TICKERS, EMPRESAS, COL_PRECIOS, COL_ORDENES

st.set_page_config(page_title="Broker — InvestAI", page_icon="🚀", layout="wide")

if not st.session_state.get("logueado"):
    st.warning("Inicia sesión desde la página principal.")
    st.stop()

st.title("🚀 Broker — Paper Trading")
st.caption("Libro de órdenes manual · Precio real de ejecución vía MongoDB Atlas")

db = get_db()
COMISION_PCT = 0.001

def calcular_cuenta():
    """Recalcula caja + posiciones desde el historial de órdenes (misma lógica que /api/cuenta)."""
    ordenes = list(db[COL_ORDENES].find({}, {"_id": 0}).sort("created_at", 1))
    caja = 100_000.0
    posiciones = {}
    for o in ordenes:
        if o["direccion"] == "BUY":
            caja -= o["total"]
            posiciones[o["ticker"]] = posiciones.get(o["ticker"], 0) + o["cantidad"]
        else:
            caja += o["total"]
            posiciones[o["ticker"]] = posiciones.get(o["ticker"], 0) - o["cantidad"]
    return caja, posiciones, ordenes

caja, posiciones, ordenes = calcular_cuenta()

# ── Valor de posiciones a precio actual ──────────────────────────────────
valor_posiciones = 0.0
filas_posiciones = []
for ticker, cantidad in posiciones.items():
    if cantidad <= 0:
        continue
    ultimo = db[COL_PRECIOS].find_one({"ticker": ticker}, sort=[("fecha", -1)])
    precio = ultimo["close"] if ultimo else 0
    valor = precio * cantidad
    valor_posiciones += valor
    filas_posiciones.append({"Ticker": ticker, "Cantidad": cantidad, "Precio Actual": precio, "Valor": valor})

col1, col2, col3 = st.columns(3)
col1.metric("Poder Adquisitivo", f"${caja:,.2f}")
col2.metric("Valor en Posiciones", f"${valor_posiciones:,.2f}")
col3.metric("Valor Total de Cuenta", f"${caja + valor_posiciones:,.2f}")

st.markdown("---")

col_form, col_hist = st.columns([1, 1.6])

with col_form:
    st.markdown("#### Nueva Orden")
    ticker = st.selectbox("Ticker", TICKERS, format_func=lambda t: f"{t} — {EMPRESAS[t]}", key="m8_ticker")
    direccion = st.radio("Dirección", ["BUY", "SELL"], horizontal=True)
    tipo_orden = st.selectbox("Tipo de Orden", ["MARKET", "LIMIT"])
    cantidad = st.number_input("Cantidad (acciones)", min_value=1, value=10, step=1)

    t = validar_ticker(ticker)
    ultimo = db[COL_PRECIOS].find_one({"ticker": t}, sort=[("fecha", -1)])
    precio_actual = ultimo["close"] if ultimo else None

    if precio_actual:
        subtotal = precio_actual * cantidad
        comision = subtotal * COMISION_PCT
        total = subtotal + comision if direccion == "BUY" else subtotal - comision
        st.info(f"Precio real: \\${precio_actual:.2f} · Comisión (0.10%): \\${comision:.2f} · **Total: \\${total:.2f}**")

        if st.button("Enviar Orden", type="primary", use_container_width=True):
            # ── Validaciones server-side (mismo criterio que /api/ordenes) ──
            if direccion == "BUY" and total > caja:
                st.error(f"Fondos insuficientes. Poder adquisitivo: \\${caja:,.2f}, se necesitan \\${total:,.2f}.")
            elif direccion == "SELL" and posiciones.get(t, 0) < cantidad:
                st.error(f"Posición insuficiente. Tienes {posiciones.get(t, 0)} acciones de {t}, intentas vender {cantidad}.")
            else:
                doc = {
                    "ticker": t, "direccion": direccion, "cantidad": cantidad,
                    "tipo_orden": tipo_orden, "precio_ejecucion": round(precio_actual, 4),
                    "subtotal": round(subtotal, 2), "comision": round(comision, 2), "total": round(total, 2),
                    "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "created_at": datetime.now(),
                }
                db[COL_ORDENES].insert_one(doc)
                st.success(f"✓ Orden ejecutada: {direccion} {cantidad} {t} @ \\${precio_actual:.2f}")
                st.rerun()
    else:
        st.error(f"No hay precio disponible para {t}.")

with col_hist:
    st.markdown("#### Historial de Órdenes")
    if ordenes:
        df_ord = pd.DataFrame(ordenes[::-1][:30])  # más recientes primero
        st.dataframe(
            df_ord[["fecha", "ticker", "direccion", "tipo_orden", "cantidad", "precio_ejecucion", "total"]]
            .rename(columns={"fecha": "Fecha", "ticker": "Ticker", "direccion": "Dir.", "tipo_orden": "Tipo",
                              "cantidad": "Cant.", "precio_ejecucion": "Precio", "total": "Total"}),
            use_container_width=True, hide_index=True,
            column_config={
                "Precio": st.column_config.NumberColumn(format="$%.2f"),
                "Total": st.column_config.NumberColumn(format="$%.2f"),
            }
        )
    else:
        st.info("Aún no has enviado ninguna orden.")

    if filas_posiciones:
        st.markdown("#### Posiciones Abiertas")
        st.dataframe(pd.DataFrame(filas_posiciones), use_container_width=True, hide_index=True,
                      column_config={
                          "Precio Actual": st.column_config.NumberColumn(format="$%.2f"),
                          "Valor": st.column_config.NumberColumn(format="$%.2f"),
                      })

st.caption(f"Libro de órdenes manual · Precio real de ejecución vía MongoDB Atlas · Última orden: {ordenes[-1]['fecha'] if ordenes else '—'}")
