"""
db.py — Capa de acceso a datos de InvestAI
============================================

Centraliza todas las consultas a MongoDB Atlas (base de datos `spbi`) para
la aplicación Streamlit. Reemplaza a la API FastAPI/ngrok del Notebook 11:
en vez de exponer endpoints HTTP, cada función de este módulo hace
directamente la consulta a Mongo y devuelve la estructura ya lista para
que las páginas de Streamlit la consuman con `from db import ...`.

Todas las funciones son defensivas: si la colección está vacía, el
documento no existe o el servidor no responde, devuelven `None` (o una
lista/DataFrame vacío según corresponda) en vez de lanzar una excepción
que rompa la app.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# ────────────────────────────────────────────────────────────────
# Nombres de colecciones (mismos que poblaron los Notebooks 1-10)
# ────────────────────────────────────────────────────────────────
COL_PRECIOS = "precios_ohlcv"
COL_PREDICCIONES = "predicciones"
COL_METRICAS = "metricas_modelos"
COL_NOTICIAS = "noticias"
COL_ESTRATEGIAS = "estrategias"
COL_BACKTESTS = "backtests"
COL_ORDENES = "ordenes_simuladas"

TICKERS_VALIDOS = {"FSM", "VOLCABC1.LM", "ABX.TO", "BVN", "BHP"}
MODELOS_RNN_VALIDOS = {"LSTM", "BiLSTM", "GRU", "SimpleRNN"}
FUENTES_VALIDAS = {"bloomberg", "cnbc", "reuters", "marketwatch"}
SENTIMIENTOS_VALIDOS = {"bullish", "bearish", "neutral"}
PERFILES_VALIDOS = {"conservador", "moderado", "agresivo"}
HORIZONTES_VALIDOS = {"3m", "6m", "1y", "3y"}
MODELOS_BACKTEST_VALIDOS = {"SVC", "LSTM", "BiLSTM", "GRU", "SimpleRNN"}

COMISION_ORDEN_PCT = 0.001  # 0.10%, igual que el Notebook 10 / la API original
HORIZONTES_DISPONIBLES = [7, 14, 30, 60]

# Campos que nunca deben viajar al frontend
_PROYECCION_BASE = {"_id": 0, "created_at": 0}


# ────────────────────────────────────────────────────────────────
# Conexión central (cacheada como recurso — se crea una sola vez)
# ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_client() -> Optional[MongoClient]:
    """
    Crea (una única vez por sesión de servidor) el cliente de MongoDB
    usando la URI guardada en `st.secrets["MONGO_URI"]`.

    Retorna `None` si la conexión falla, para que el resto de las
    funciones puedan degradar de forma controlada en vez de crashear.
    """
    try:
        uri = st.secrets["MONGO_URI"]
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client
    except Exception as e:
        st.error(f"No se pudo conectar a MongoDB Atlas: {e}")
        return None


def get_db():
    """Devuelve el objeto de base de datos `spbi`, o `None` si no hay conexión."""
    client = get_client()
    if client is None:
        return None
    return client["spbi"]


def _coleccion(nombre: str):
    """Helper interno: devuelve la colección `nombre`, o `None` si no hay DB."""
    db = get_db()
    if db is None:
        return None
    return db[nombre]


def _validar_ticker(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    if t not in TICKERS_VALIDOS:
        raise ValueError(
            f"Ticker '{t}' no está en el sistema. "
            f"Tickers disponibles: {sorted(TICKERS_VALIDOS)}"
        )
    return t


# ────────────────────────────────────────────────────────────────
# 1) Mercado — Notebook 1 (precios_ohlcv)
# ────────────────────────────────────────────────────────────────
def get_mercado_data(ticker: str, dias: Optional[int] = None) -> pd.DataFrame:
    """
    Retorna un DataFrame con los datos OHLCV + indicadores técnicos
    (SMA-20, SMA-50, EMA-12, EMA-26, RSI-14, etc.) de un ticker,
    ordenados cronológicamente por `fecha`.

    Si se especifica `dias`, recorta a los últimos N días disponibles.
    Retorna un DataFrame vacío si no hay datos o si ocurre un error.
    """
    try:
        t = _validar_ticker(ticker)
        col = _coleccion(COL_PRECIOS)
        if col is None:
            return pd.DataFrame()

        cursor = col.find({"ticker": t}, _PROYECCION_BASE).sort("fecha", 1)
        datos: List[Dict[str, Any]] = list(cursor)

        if not datos:
            return pd.DataFrame()

        if dias is not None and dias > 0:
            datos = datos[-dias:]

        df = pd.DataFrame(datos)
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"])
        return df

    except (PyMongoError, ValueError, Exception):
        return pd.DataFrame()


# ────────────────────────────────────────────────────────────────
# 2) Clasificador SVC — Notebook 2 (predicciones + metricas_modelos)
# ────────────────────────────────────────────────────────────────
def get_svc_data(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Unifica en un solo diccionario:
      - la predicción actual (señal, confianza)
      - las métricas del modelo (accuracy, f1, matriz de confusión, hiperparámetros)
      - el histórico de señales BUY/SELL

    Retorna `None` si no hay predicción SVC para el ticker o si ocurre un error.
    """
    try:
        t = _validar_ticker(ticker)
        pred_col = _coleccion(COL_PREDICCIONES)
        met_col = _coleccion(COL_METRICAS)
        if pred_col is None or met_col is None:
            return None

        prediccion = pred_col.find_one({"ticker": t, "modelo": "SVC"}, _PROYECCION_BASE)
        if not prediccion:
            return None

        metricas = met_col.find_one(
            {"ticker": t, "modelo": "SVC"},
            {**_PROYECCION_BASE, "fecha_entrenamiento": 0},
        ) or {}

        return {
            "ticker": t,
            "modelo": "SVC",
            "fecha_prediccion": prediccion.get("fecha_prediccion", "—"),
            "tipo_features": prediccion.get("tipo_features"),
            "prediccion": {
                "senal": prediccion.get("senal"),
                "confianza": prediccion.get("confianza"),
            },
            "metricas": {
                "accuracy": metricas.get("accuracy"),
                "precision": metricas.get("precision"),
                "recall": metricas.get("recall"),
                "f1": metricas.get("f1"),
                "matriz_confusion": metricas.get("matriz_confusion"),
                "mejor_kernel": metricas.get("mejor_kernel"),
                "mejor_C": metricas.get("mejor_C"),
                "mejor_gamma": metricas.get("mejor_gamma"),
            },
            "historico_senales": prediccion.get("historico_senales", []),
        }

    except (PyMongoError, ValueError, Exception):
        return None


# ────────────────────────────────────────────────────────────────
# 3) Clasificadores RNN — Notebooks 3-6 (LSTM/BiLSTM/GRU/SimpleRNN)
# ────────────────────────────────────────────────────────────────
def get_rnn_data(ticker: str, arquitectura: str = "LSTM") -> Optional[Dict[str, Any]]:
    """
    Igual que get_svc_data, pero filtrando `predicciones` y `metricas_modelos`
    por el campo `modelo` según la arquitectura RNN elegida.

    `arquitectura` debe ser una de: 'LSTM', 'BiLSTM', 'GRU', 'SimpleRNN'.
    Retorna `None` si la arquitectura no es válida, no hay datos o hay error.
    """
    try:
        t = _validar_ticker(ticker)
        m = (arquitectura or "").strip()
        if m not in MODELOS_RNN_VALIDOS:
            raise ValueError(
                f"Arquitectura '{m}' no reconocida. Válidas: {sorted(MODELOS_RNN_VALIDOS)}"
            )

        pred_col = _coleccion(COL_PREDICCIONES)
        met_col = _coleccion(COL_METRICAS)
        if pred_col is None or met_col is None:
            return None

        prediccion = pred_col.find_one({"ticker": t, "modelo": m}, _PROYECCION_BASE)
        if not prediccion:
            return None

        metricas = met_col.find_one(
            {"ticker": t, "modelo": m},
            {**_PROYECCION_BASE, "fecha_entrenamiento": 0},
        ) or {}

        return {
            "ticker": t,
            "modelo": m,
            "fecha_prediccion": prediccion.get("fecha_prediccion", "—"),
            "n_steps": prediccion.get("n_steps"),
            "prediccion": {
                "senal": prediccion.get("senal"),
                "confianza": prediccion.get("confianza"),
            },
            "metricas": {
                "accuracy": metricas.get("accuracy"),
                "precision": metricas.get("precision"),
                "recall": metricas.get("recall"),
                "f1": metricas.get("f1"),
                "epocas_entrenadas": metricas.get("epocas_entrenadas"),
                "matriz_confusion": metricas.get("matriz_confusion"),
                "historial_epocas": metricas.get("historial_epocas"),
            },
            "historico_senales": prediccion.get("historico_senales", []),
        }

    except (PyMongoError, ValueError, Exception):
        return None


# ────────────────────────────────────────────────────────────────
# 4) Regresor LSTM — Notebook 7 (modelo='LSTM_REG')
# ────────────────────────────────────────────────────────────────
def get_lstm_regresor(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Extrae el pronóstico de precio continuo del regresor LSTM: histórico
    predicho vs. real, y proyecciones a futuro (7/14/30/60 días) con
    bandas de confianza, traduciendo el esquema interno de Mongo
    (`proyecciones_horizonte` con claves '7d'/'14d'/...) al esquema de
    salida (`prediccion_futura` con claves '7_dias'/...).

    Retorna `None` si no hay pronóstico LSTM_REG para el ticker o hay error.
    """
    try:
        t = _validar_ticker(ticker)
        pred_col = _coleccion(COL_PREDICCIONES)
        met_col = _coleccion(COL_METRICAS)
        if pred_col is None or met_col is None:
            return None

        prediccion = pred_col.find_one({"ticker": t, "modelo": "LSTM_REG"}, _PROYECCION_BASE)
        if not prediccion:
            return None

        metricas = met_col.find_one(
            {"ticker": t, "modelo": "LSTM_REG"},
            {**_PROYECCION_BASE, "fecha_entrenamiento": 0},
        ) or {}

        horizonte_interno = prediccion.get("proyecciones_horizonte", {})
        prediccion_futura: Dict[str, Any] = {}
        for h in HORIZONTES_DISPONIBLES:
            datos_h = horizonte_interno.get(f"{h}d")
            if datos_h:
                prediccion_futura[f"{h}_dias"] = {
                    "precio": datos_h.get("precio_estimado"),
                    "banda_sup": datos_h.get("banda_superior"),
                    "banda_inf": datos_h.get("banda_inferior"),
                }

        return {
            "ticker": t,
            "modelo": "LSTM_REG",
            "fecha_prediccion": prediccion.get("fecha_prediccion", "—"),
            "n_steps": prediccion.get("n_steps"),
            "precio_actual_usd": prediccion.get("ultimo_precio"),
            "metricas": {
                "rmse_usd": metricas.get("rmse_usd"),
                "rmse_pct": metricas.get("rmse_pct"),
                "mae_usd": metricas.get("mae_usd"),
                "r2": metricas.get("r2"),
                "rmse_arima_baseline": metricas.get("rmse_arima_baseline"),
                "epocas_entrenadas": metricas.get("epocas_entrenadas"),
                "historial_epocas": metricas.get("historial_epocas"),
            },
            "historico_precios": prediccion.get("historico_predicciones", []),
            "serie_diaria_60d": prediccion.get("serie_diaria_60d", []),
            "prediccion_futura": prediccion_futura,
        }

    except (PyMongoError, ValueError, Exception):
        return None


# ────────────────────────────────────────────────────────────────
# 5) Noticias / Sentimiento NLP — Notebook 8 (VADER)
# ────────────────────────────────────────────────────────────────
def get_noticias(
    fuente: str = "all",
    sentimiento: str = "all",
    ticker: Optional[str] = None,
    limite: int = 50,
) -> Dict[str, Any]:
    """
    Consulta la colección `noticias` con los mismos filtros que el
    selector del frontend:
      - fuente: 'all' | 'bloomberg' | 'cnbc' | 'reuters' | 'marketwatch' (case-insensitive)
      - sentimiento: 'all' | 'bullish' | 'bearish' | 'neutral'
      - ticker: opcional

    Ordena por `fecha_publicacion` descendente y calcula el sentimiento
    consolidado del conjunto filtrado.

    Retorna siempre una estructura válida (nunca None): si no hay
    coincidencias, `noticias` es una lista vacía.
    """
    resultado_vacio = {
        "total": 0,
        "compound_promedio": None,
        "sentimiento_consolidado": None,
        "mensaje": "No hay noticias disponibles.",
        "noticias": [],
    }
    try:
        col = _coleccion(COL_NOTICIAS)
        if col is None:
            return resultado_vacio

        query: Dict[str, Any] = {}

        if ticker:
            query["ticker"] = _validar_ticker(ticker)

        f = (fuente or "all").strip().lower()
        if f != "all":
            if f not in FUENTES_VALIDAS:
                raise ValueError(f"Fuente '{fuente}' no reconocida.")
            query["fuente"] = {"$regex": f"^{f}$", "$options": "i"}

        s = (sentimiento or "all").strip().lower()
        if s != "all":
            if s not in SENTIMIENTOS_VALIDOS:
                raise ValueError(f"Sentimiento '{sentimiento}' no reconocido.")
            query["sentimiento"] = s.upper()

        cursor = col.find(query, _PROYECCION_BASE).sort("fecha_publicacion", -1).limit(limite)
        docs = list(cursor)

        compounds = [d.get("compound", 0.0) for d in docs]
        compound_promedio = round(sum(compounds) / len(compounds), 4) if compounds else None
        etiqueta = None
        if compound_promedio is not None:
            if compound_promedio > 0.05:
                etiqueta = "BULLISH"
            elif compound_promedio < -0.05:
                etiqueta = "BEARISH"
            else:
                etiqueta = "NEUTRAL"

        return {
            "total": len(docs),
            "filtro_aplicado": {"fuente": fuente, "sentimiento": sentimiento, "ticker": ticker},
            "compound_promedio": compound_promedio,
            "sentimiento_consolidado": etiqueta,
            "mensaje": None if docs else "No hay noticias que coincidan con el filtro seleccionado.",
            "noticias": docs,
        }

    except (PyMongoError, ValueError, Exception):
        return resultado_vacio


# ────────────────────────────────────────────────────────────────
# 6) Estrategias — Notebook 9 (Optimizador Markowitz)
# ────────────────────────────────────────────────────────────────
def get_estrategias(
    perfil_riesgo: str = "moderado",
    horizonte: str = "1y",
    capital: float = 100000.0,
) -> Optional[Dict[str, Any]]:
    """
    Retorna la asignación óptima de portafolio (Markowitz) para una
    combinación de perfil de riesgo y horizonte temporal, con los montos
    en USD escalados al `capital` solicitado.

    Retorna `None` si la combinación no existe o hay error.
    """
    try:
        p = (perfil_riesgo or "").strip().lower()
        h = (horizonte or "").strip().lower()
        if p not in PERFILES_VALIDOS:
            raise ValueError(f"Perfil de riesgo '{perfil_riesgo}' no reconocido.")
        if h not in HORIZONTES_VALIDOS:
            raise ValueError(f"Horizonte '{horizonte}' no reconocido.")

        col = _coleccion(COL_ESTRATEGIAS)
        if col is None:
            return None

        doc = col.find_one({"perfil_riesgo": p, "horizonte": h}, _PROYECCION_BASE)
        if not doc:
            return None

        activos_escalados = []
        for activo in doc.get("activos", []):
            activo_escalado = dict(activo)
            activo_escalado["monto_asignado_usd"] = round(
                capital * activo.get("asignacion_pct", 0.0) / 100.0, 2
            )
            activos_escalados.append(activo_escalado)

        return {
            "perfil_riesgo": doc.get("perfil_riesgo"),
            "horizonte": doc.get("horizonte"),
            "capital_inicial_usd": capital,
            "retorno_esperado_anual": doc.get("retorno_esperado_anual"),
            "volatilidad_anual": doc.get("volatilidad_anual"),
            "sharpe_ratio": doc.get("sharpe_ratio"),
            "optimizacion_exitosa": doc.get("optimizacion_exitosa"),
            "activos": activos_escalados,
            "frontera_eficiente": doc.get("frontera_eficiente", []),
            "fecha_calculo": doc.get("fecha_calculo"),
        }

    except (PyMongoError, ValueError, Exception):
        return None


# ────────────────────────────────────────────────────────────────
# 7) Backtesting / Portafolio — Notebook 10
# ────────────────────────────────────────────────────────────────
def get_backtest_report(
    modelo: str = "SVC",
    perfil_riesgo: str = "moderado",
    capital: float = 100000.0,
) -> Optional[Dict[str, Any]]:
    """
    Retorna el backtest de portafolio diversificado (curva de equity,
    métricas técnicas, trades y posiciones finales) para una combinación
    de modelo de señales y perfil de riesgo, reescalando los montos en USD
    proporcionalmente al `capital` solicitado.

    Alimenta tanto la vista técnica de Backtesting como la vista resumida
    de Portafolio. Retorna `None` si la combinación no existe o hay error.
    """
    try:
        m = (modelo or "").strip()
        p = (perfil_riesgo or "").strip().lower()
        if m not in MODELOS_BACKTEST_VALIDOS:
            raise ValueError(f"Modelo '{modelo}' no reconocido.")
        if p not in PERFILES_VALIDOS:
            raise ValueError(f"Perfil de riesgo '{perfil_riesgo}' no reconocido.")

        col = _coleccion(COL_BACKTESTS)
        if col is None:
            return None

        doc = col.find_one({"modelo": m, "perfil_riesgo": p}, {"_id": 0})
        if not doc:
            return None

        capital_base = doc.get("capital_base", 100000.0) or 100000.0
        escala = capital / capital_base

        equity_curve_escalada = [
            {"fecha": punto["fecha"], "valor": round(punto["valor"] * escala, 2)}
            for punto in doc.get("equity_curve", [])
        ]

        posiciones_escaladas = []
        for pos in doc.get("posiciones_finales", []):
            pos_escalada = dict(pos)
            for campo in ("capital_inicial_sleeve", "valor_actual", "pnl_usd"):
                if pos_escalada.get(campo) is not None:
                    pos_escalada[campo] = round(pos_escalada[campo] * escala, 2)
            posiciones_escaladas.append(pos_escalada)

        return {
            "modelo": doc.get("modelo"),
            "perfil_riesgo": doc.get("perfil_riesgo"),
            "horizonte_pesos": doc.get("horizonte_pesos"),
            "capital_inicial_usd": capital,
            "comision_pct": doc.get("comision_pct"),
            "slippage_pct": doc.get("slippage_pct"),
            "fecha_inicio": doc.get("fecha_inicio"),
            "fecha_fin": doc.get("fecha_fin"),
            "metricas": doc.get("metricas", {}),
            "equity_curve": equity_curve_escalada,
            "trades": doc.get("trades", []),  # en % de retorno, no dependen del capital
            "posiciones_finales": posiciones_escaladas,
        }

    except (PyMongoError, ValueError, Exception):
        return None


# ────────────────────────────────────────────────────────────────
# 8) Broker — libro de órdenes manual (escritura)
# ────────────────────────────────────────────────────────────────
def registrar_orden_manual(
    ticker: str,
    direccion: str,
    cantidad: float,
    tipo_orden: str = "MARKET",
) -> Optional[Dict[str, Any]]:
    """
    Registra una orden manual del usuario en `ordenes_simuladas`.

    El precio de ejecución se toma SIEMPRE del último precio de cierre real
    disponible en `precios_ohlcv` (nunca se confía en un precio enviado por
    el cliente). Calcula subtotal, comisión (0.10%) y total.

    Retorna el documento insertado (con `_id` como string), o `None` si
    ocurre un error de validación o de conexión.
    """
    try:
        t = _validar_ticker(ticker)

        d = (direccion or "").strip().upper()
        if d not in {"BUY", "SELL"}:
            raise ValueError("direccion debe ser 'BUY' o 'SELL'.")

        if cantidad is None or cantidad <= 0:
            raise ValueError("cantidad debe ser mayor que 0.")

        precios_col = _coleccion(COL_PRECIOS)
        ordenes_col = _coleccion(COL_ORDENES)
        if precios_col is None or ordenes_col is None:
            return None

        ultimo_doc = precios_col.find_one({"ticker": t}, sort=[("fecha", -1)])
        if not ultimo_doc:
            raise ValueError(f"No hay precios disponibles para '{t}'.")

        precio_ejecucion = float(ultimo_doc["close"])
        subtotal = precio_ejecucion * cantidad
        comision = round(subtotal * COMISION_ORDEN_PCT, 2)
        total = round(subtotal + comision, 2) if d == "BUY" else round(subtotal - comision, 2)

        documento = {
            "ticker": t,
            "direccion": d,
            "cantidad": cantidad,
            "tipo_orden": (tipo_orden or "MARKET").strip().upper(),
            "precio_ejecucion": round(precio_ejecucion, 4),
            "subtotal": round(subtotal, 2),
            "comision": comision,
            "total": total,
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "created_at": datetime.now(),
        }

        resultado = ordenes_col.insert_one(documento)
        documento["_id"] = str(resultado.inserted_id)
        return documento

    except (PyMongoError, ValueError, Exception):
        return None


def get_historial_ordenes(ticker: Optional[str] = None, limite: int = 50) -> List[Dict[str, Any]]:
    """
    Lista las órdenes manuales registradas en `ordenes_simuladas`,
    ordenadas por fecha de creación descendente (más recientes primero).

    Retorna una lista vacía si no hay órdenes o si ocurre un error.
    """
    try:
        col = _coleccion(COL_ORDENES)
        if col is None:
            return []

        filtro: Dict[str, Any] = {}
        if ticker:
            filtro["ticker"] = ticker.strip().upper()

        cursor = col.find(filtro, {"created_at": 0}).sort("created_at", -1).limit(limite)
        ordenes = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            ordenes.append(doc)
        return ordenes

    except (PyMongoError, Exception):
        return []
