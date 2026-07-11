"""
pages/5_📰_NLP_Noticias.py — Análisis de Noticias NLP
=======================================================
Reemplaza el módulo M5 (Feed de Noticias NLP) del index.html original:
feed de noticias con sentimiento BULLISH/BEARISH/NEUTRAL (VADER),
filtrable por fuente y sentimiento.

Consulta `db.py` directamente contra MongoDB Atlas (colección `noticias`,
poblada por el Notebook 8), sin pasar por FastAPI/ngrok.
"""

from datetime import datetime

import streamlit as st

from db import get_noticias

# ────────────────────────────────────────────────────────────────
# Configuración de página
# ────────────────────────────────────────────────────────────────
try:
    st.set_page_config(page_title="NLP Noticias · InvestAI", page_icon="📰", layout="wide")
except Exception:
    pass  # ya fue configurado por el entrypoint principal

# ────────────────────────────────────────────────────────────────
# Paleta oscura fiel al index.html original
# ────────────────────────────────────────────────────────────────
COLOR_BG = "#0f172a"
COLOR_CARD = "#1e293b"
COLOR_TEXT = "#f8fafc"
COLOR_MUTED = "#94a3b8"
COLOR_MUTED_DIM = "#64748b"
COLOR_POSITIVE = "#26A69A"
COLOR_NEGATIVE = "#EF5350"
COLOR_WARNING = "#FFC107"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {COLOR_BG};
        color: {COLOR_TEXT};
    }}
    .modulo-header p {{
        color: {COLOR_MUTED};
        font-size: 13px;
        margin-top: -8px;
    }}
    .news-item {{
        background-color: {COLOR_CARD};
        border-radius: 8px;
        padding: 12px;
        display: flex;
        gap: 12px;
        margin-bottom: 12px;
    }}
    .news-sentiment {{
        display: flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 16px;
        flex-shrink: 0;
    }}
    .news-content {{
        flex: 1;
    }}
    .news-title {{
        font-size: 14px;
        font-weight: 600;
        color: {COLOR_TEXT};
        margin-bottom: 4px;
    }}
    .news-text {{
        font-size: 13px;
        color: {COLOR_MUTED};
        margin-bottom: 6px;
    }}
    .news-meta {{
        font-size: 11px;
        color: {COLOR_MUTED_DIM};
    }}
    .footer-nota {{
        margin-top: 24px;
        padding-top: 16px;
        border-top: 1px solid rgba(255,255,255,0.08);
        font-size: 12px;
        color: {COLOR_MUTED};
        text-align: center;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Estilo por sentimiento: (color de borde/icono, color de fondo del icono, emoji)
ESTILO_SENTIMIENTO = {
    "BULLISH": (COLOR_POSITIVE, "rgba(38, 166, 154, 0.2)", "📈"),
    "BEARISH": (COLOR_NEGATIVE, "rgba(239, 83, 80, 0.2)", "📉"),
    "NEUTRAL": (COLOR_WARNING, "rgba(255, 193, 7, 0.2)", "➡️"),
}


def formatear_fecha_relativa(fecha) -> str:
    """Traduce una fecha (str ISO o datetime) a texto relativo, igual que
    la función `formatearFechaRelativa` del index.html original."""
    if not fecha:
        return "—"

    fecha_dt = None
    if isinstance(fecha, datetime):
        fecha_dt = fecha
    elif isinstance(fecha, str):
        try:
            fecha_dt = datetime.fromisoformat(fecha.replace("Z", "+00:00"))
        except ValueError:
            return fecha

    if fecha_dt is None:
        return str(fecha)

    ahora = datetime.now(fecha_dt.tzinfo) if fecha_dt.tzinfo else datetime.now()
    try:
        diff_seg = (ahora - fecha_dt).total_seconds()
    except TypeError:
        # fechas naive vs aware mezcladas: se compara sin tz
        diff_seg = (ahora.replace(tzinfo=None) - fecha_dt.replace(tzinfo=None)).total_seconds()

    diff_min = int(diff_seg // 60)
    if diff_min < 1:
        return "hace instantes"
    if diff_min < 60:
        return f"hace {diff_min} min"
    diff_h = diff_min // 60
    if diff_h < 24:
        return f"hace {diff_h} h"
    diff_dias = diff_h // 24
    return f"hace {diff_dias} día" + ("" if diff_dias == 1 else "s")


st.markdown(
    """
    <div class="modulo-header">
        <h2>📰 Análisis de Noticias NLP</h2>
        <p>Feed de noticias con análisis de sentimiento BULLISH/BEARISH/NEUTRAL</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────
# Controles locales: fuente + sentimiento
# ────────────────────────────────────────────────────────────────
FUENTES = {"Todas": "all", "Bloomberg": "bloomberg", "CNBC": "cnbc", "Reuters": "reuters"}
SENTIMIENTOS = {"Todos": "all", "Bullish": "bullish", "Bearish": "bearish", "Neutral": "neutral"}

col_fuente, col_sentimiento = st.columns(2)
with col_fuente:
    fuente_label = st.selectbox("Fuente", options=list(FUENTES.keys()))
with col_sentimiento:
    sentimiento_label = st.selectbox("Sentimiento", options=list(SENTIMIENTOS.keys()))

fuente_valor = FUENTES[fuente_label]
sentimiento_valor = SENTIMIENTOS[sentimiento_label]

# ────────────────────────────────────────────────────────────────
# Datos
# ────────────────────────────────────────────────────────────────
resultado = get_noticias(fuente=fuente_valor, sentimiento=sentimiento_valor)
noticias = resultado.get("noticias", [])

if not noticias:
    mensaje = resultado.get("mensaje") or "No hay noticias que coincidan con este filtro."
    st.markdown(
        f"""
        <div style="text-align:center; padding:30px 0; color:{COLOR_MUTED};">
            <p>📭 {mensaje}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ────────────────────────────────────────────────────────────────
# Feed de noticias
# ────────────────────────────────────────────────────────────────
for noticia in noticias:
    sentimiento = (noticia.get("sentimiento") or "NEUTRAL").upper()
    color_borde, color_fondo_icono, emoji = ESTILO_SENTIMIENTO.get(
        sentimiento, ESTILO_SENTIMIENTO["NEUTRAL"]
    )

    titulo = noticia.get("titulo", "Sin título")
    texto = noticia.get("texto", "")
    fuente_noticia = noticia.get("fuente", "—")
    ticker_noticia = noticia.get("ticker")
    fecha_relativa = formatear_fecha_relativa(noticia.get("fecha_publicacion"))

    meta = fuente_noticia
    if ticker_noticia:
        meta += f" · {ticker_noticia}"
    meta += f" · {fecha_relativa}"

    st.markdown(
        f"""
        <div class="news-item" style="border-left: 4px solid {color_borde};">
            <div class="news-sentiment" style="background-color:{color_fondo_icono}; color:{color_borde};">
                {emoji}
            </div>
            <div class="news-content">
                <div class="news-title">{titulo}</div>
                <div class="news-text">{texto}</div>
                <div class="news-meta">{meta}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ────────────────────────────────────────────────────────────────
# Pie de página
# ────────────────────────────────────────────────────────────────
sentimiento_consolidado = resultado.get("sentimiento_consolidado")
ultima_fecha = noticias[0].get("fecha_publicacion", "—") if noticias else "—"
st.markdown(
    f"""
    <div class="footer-nota">
        Datos reales de noticias (yfinance) · Análisis de sentimiento VADER (NLTK) ·
        Sentimiento consolidado: {sentimiento_consolidado or '—'} · Actualizado: {ultima_fecha}
    </div>
    """,
    unsafe_allow_html=True,
)
