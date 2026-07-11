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

import numpy as np
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

# ────────────────────────────────────────────────────────────────
# Parámetros de entrenamiento "en vivo" (disparado desde la Consola de
# Modelos). Reducidos respecto a los Notebooks originales para que el
# pipeline termine en segundos en Streamlit Community Cloud, sin saturar
# CPU/memoria del contenedor. La arquitectura de cada modelo (tamaño de
# capas) se mantiene idéntica a la de su Notebook — solo se recorta
# cuánto entrena, no qué tan grande es.
# ────────────────────────────────────────────────────────────────
EPOCHS_CLOUD = 5           # Notebooks originales: 80 épocas (con EarlyStopping patience=10)
CV_FOLDS_CLOUD = 2         # Notebook 2 original: cv=5
N_STEPS_RNN = 20           # igual que Notebooks 3-6 (ventana temporal, sin cambios)

FEATURES_PRECIO = ["close", "sma_20", "sma_50", "ema_12", "ema_26", "rsi_14", "retorno"]
FEATURES_VOLUMEN = ["vol_change_pct", "vol_relative", "vol_rsi", "obv_slope", "money_flow_rel"]

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


# ══════════════════════════════════════════════════════════════════════
# 9) Pipeline de ingesta y entrenamiento EN VIVO (disparado desde la
#    Consola de Modelos). Todo lo pesado (yfinance, scikit-learn,
#    tensorflow) se importa de forma perezosa (dentro de cada función)
#    para no penalizar el arranque de las páginas que solo leen datos.
# ══════════════════════════════════════════════════════════════════════

def _calcular_sma(serie: pd.Series, periodo: int) -> pd.Series:
    """Media móvil simple."""
    return serie.rolling(window=periodo).mean()


def _calcular_ema(serie: pd.Series, periodo: int) -> pd.Series:
    """Media móvil exponencial."""
    return serie.ewm(span=periodo, adjust=False).mean()


def _calcular_rsi(serie: pd.Series, periodo: int = 14) -> pd.Series:
    """Índice de Fuerza Relativa (RSI)."""
    delta = serie.diff()
    ganancia = delta.where(delta > 0, 0).rolling(window=periodo).mean()
    perdida = -delta.where(delta < 0, 0).rolling(window=periodo).mean()
    rs = ganancia / perdida
    return 100 - (100 / (1 + rs))


def _limpiar_valor(v) -> Optional[float]:
    """Convierte NaN en None para que el documento sea BSON válido."""
    return None if pd.isna(v) else round(float(v), 4)


def ejecutar_ingesta_y_tecnico(ticker: str, periodo: str = "1y") -> Dict[str, Any]:
    """
    Descarga OHLCV real desde Yahoo Finance (yfinance), calcula los
    indicadores técnicos (SMA20, SMA50, EMA12, EMA26, RSI14 y Bandas de
    Bollinger) y reemplaza los documentos de `precios_ohlcv` para ese
    ticker. Equivalente en vivo al Notebook 1.

    Retorna un dict {ok, ticker, n_registros, mensaje} — nunca lanza
    una excepción hacia la página que la llama.
    """
    try:
        import yfinance as yf

        t = _validar_ticker(ticker)
        col = _coleccion(COL_PRECIOS)
        if col is None:
            return {"ok": False, "ticker": t, "n_registros": 0, "mensaje": "Sin conexión a MongoDB Atlas."}

        df = yf.download(t, period=periodo, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return {"ok": False, "ticker": t, "n_registros": 0, "mensaje": f"Yahoo Finance no devolvió datos para {t}."}

        # yfinance devuelve columnas con MultiIndex incluso para un solo ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["sma_20"] = _calcular_sma(df["Close"], 20)
        df["sma_50"] = _calcular_sma(df["Close"], 50)
        df["ema_12"] = _calcular_ema(df["Close"], 12)
        df["ema_26"] = _calcular_ema(df["Close"], 26)
        df["rsi_14"] = _calcular_rsi(df["Close"], 14)

        # Bandas de Bollinger (20 periodos, 2 desviaciones estándar)
        bb_media = df["Close"].rolling(window=20).mean()
        bb_std = df["Close"].rolling(window=20).std()
        df["bb_upper"] = bb_media + 2 * bb_std
        df["bb_middle"] = bb_media
        df["bb_lower"] = bb_media - 2 * bb_std

        registros = []
        for fecha, fila in df.iterrows():
            registros.append({
                "ticker": t,
                "fecha": fecha.strftime("%Y-%m-%d"),
                "open": _limpiar_valor(fila["Open"]),
                "high": _limpiar_valor(fila["High"]),
                "low": _limpiar_valor(fila["Low"]),
                "close": _limpiar_valor(fila["Close"]),
                "volume": int(fila["Volume"]) if not pd.isna(fila["Volume"]) else 0,
                "sma_20": _limpiar_valor(fila["sma_20"]),
                "sma_50": _limpiar_valor(fila["sma_50"]),
                "ema_12": _limpiar_valor(fila["ema_12"]),
                "ema_26": _limpiar_valor(fila["ema_26"]),
                "rsi_14": _limpiar_valor(fila["rsi_14"]),
                "bb_upper": _limpiar_valor(fila["bb_upper"]),
                "bb_middle": _limpiar_valor(fila["bb_middle"]),
                "bb_lower": _limpiar_valor(fila["bb_lower"]),
                "created_at": datetime.now(),
            })

        col.delete_many({"ticker": t})
        if registros:
            col.insert_many(registros)

        return {
            "ok": True, "ticker": t, "n_registros": len(registros),
            "mensaje": f"{len(registros)} días guardados en precios_ohlcv.",
        }

    except Exception as e:
        return {"ok": False, "ticker": ticker, "n_registros": 0, "mensaje": f"Error en ingesta: {e}"}


def _preparar_features_precio(df: pd.DataFrame):
    """Feature engineering estándar (precio + indicadores). Target: ¿sube mañana?"""
    df = df.copy()
    df["retorno"] = df["close"].pct_change()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna(subset=FEATURES_PRECIO + ["target"])

    X = df[FEATURES_PRECIO].values
    y = df["target"].values
    fechas = df["fecha"].values
    precios = df["close"].values
    return X, y, fechas, precios


def _preparar_features_volumen(df: pd.DataFrame):
    """Feature engineering basado en volumen, usado para VOLCABC1.LM (precio casi constante)."""
    df = df.copy()
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

    df["vol_change_pct"] = df["volume"].pct_change().fillna(0).clip(-5, 5)
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_relative"] = (df["volume"] / df["vol_sma20"].replace(0, np.nan)).fillna(1).clip(0, 10)

    direction = np.sign(df["close"].diff().fillna(0))
    df["obv"] = (direction * df["volume"]).cumsum()
    df["obv_sma20"] = df["obv"].rolling(20).mean()
    df["obv_slope"] = df["obv"].diff(5).fillna(0)

    delta_vol = df["volume"].diff()
    gain_vol = delta_vol.clip(lower=0).rolling(14).mean()
    loss_vol = (-delta_vol.clip(upper=0)).rolling(14).mean()
    rs_vol = gain_vol / loss_vol.replace(0, np.nan)
    df["vol_rsi"] = (100 - 100 / (1 + rs_vol)).fillna(50)

    df["money_flow"] = df["close"] * df["volume"]
    df["mf_sma10"] = df["money_flow"].rolling(10).mean()
    df["money_flow_rel"] = (df["money_flow"] / df["mf_sma10"].replace(0, np.nan)).fillna(1)

    df["target"] = (df["obv"] > df["obv_sma20"]).astype(int)
    df = df.dropna(subset=FEATURES_VOLUMEN)

    X = df[FEATURES_VOLUMEN].values
    y = df["target"].values
    fechas = df["fecha"].values
    precios = df["close"].values
    return X, y, fechas, precios


def _crear_secuencias(feats, target, fechas, precios, n_steps: int):
    """Convierte features 2D en secuencias 3D (ventana deslizante) para las RNN."""
    X_seq, y_seq, fechas_seq, precios_seq = [], [], [], []
    for i in range(n_steps, len(feats)):
        X_seq.append(feats[i - n_steps:i])
        y_seq.append(target[i])
        fechas_seq.append(fechas[i])
        precios_seq.append(precios[i])
    return (np.array(X_seq), np.array(y_seq), np.array(fechas_seq), np.array(precios_seq))


def entrenar_modelo_svc(ticker: str, cv_folds: int = CV_FOLDS_CLOUD) -> Dict[str, Any]:
    """
    Entrena un SVC (GridSearchCV + Pipeline StandardScaler+SVC) sobre los
    datos ya presentes en `precios_ohlcv` para `ticker`, y guarda la
    predicción actual + métricas + histórico de señales en MongoDB.
    Equivalente en vivo al Notebook 2, con `cv_folds` reducido para
    Streamlit Community Cloud (por defecto 2 en vez de 5).

    Retorna un dict {ok, ticker, modelo, accuracy, f1, senal, mensaje}.
    """
    try:
        from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                      precision_score, recall_score)
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        t = _validar_ticker(ticker)
        df = get_mercado_data(t)
        if df.empty or len(df) < 100:
            return {"ok": False, "ticker": t, "modelo": "SVC",
                    "mensaje": f"Datos insuficientes ({len(df)} registros). Ejecuta primero la ingesta."}

        if t == "VOLCABC1.LM":
            X, y, fechas, precios = _preparar_features_volumen(df)
            tipo_features = "volumen"
        else:
            X, y, fechas, precios = _preparar_features_precio(df)
            tipo_features = "precio"

        if len(X) < 50:
            return {"ok": False, "ticker": t, "modelo": "SVC", "mensaje": "Muestras insuficientes tras limpieza."}

        n = len(X)
        corte = int(n * 0.80)
        X_train, X_test = X[:corte], X[corte:]
        y_train, y_test = y[:corte], y[corte:]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(probability=True, class_weight="balanced")),
        ])
        param_grid = {
            "svc__kernel": ["linear", "rbf"],
            "svc__C": [0.1, 1, 10],
            "svc__gamma": ["scale", "auto"],
        }
        grid = GridSearchCV(
            pipeline, param_grid,
            cv=max(2, cv_folds), scoring="f1_weighted",
            n_jobs=-1, refit=True,
        )
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)
        metricas = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "mejor_kernel": grid.best_params_["svc__kernel"],
            "mejor_C": grid.best_params_["svc__C"],
            "mejor_gamma": str(grid.best_params_["svc__gamma"]),
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        }
        cm = confusion_matrix(y_test, y_pred).tolist()

        ultima_X = X[-1:].reshape(1, -1)
        pred_actual = int(grid.predict(ultima_X)[0])
        prob_actual = float(grid.predict_proba(ultima_X)[0].max())
        senal_actual = "BUY" if pred_actual == 1 else "SELL"

        preds_all = grid.predict(X)
        probas_all = grid.predict_proba(X)
        buy_idx = list(grid.classes_).index(1) if 1 in grid.classes_ else 0

        historico_senales = [
            {
                "fecha": str(fechas[i])[:10],
                "precio": round(float(precios[i]), 4),
                "prediccion": "BUY" if preds_all[i] == 1 else "SELL",
                "probabilidad": round(float(probas_all[i][buy_idx]), 4),
            }
            for i in range(len(preds_all))
        ]

        pred_col = _coleccion(COL_PREDICCIONES)
        met_col = _coleccion(COL_METRICAS)
        if pred_col is None or met_col is None:
            return {"ok": False, "ticker": t, "modelo": "SVC", "mensaje": "Sin conexión a MongoDB Atlas."}

        pred_col.delete_many({"ticker": t, "modelo": "SVC"})
        pred_col.insert_one({
            "ticker": t, "modelo": "SVC",
            "senal": senal_actual, "confianza": round(prob_actual, 4),
            "fecha_prediccion": datetime.now().strftime("%Y-%m-%d"),
            "historico_senales": historico_senales,
            "tipo_features": tipo_features,
            "created_at": datetime.now(),
        })

        met_col.delete_many({"ticker": t, "modelo": "SVC"})
        met_col.insert_one({
            "ticker": t, "modelo": "SVC",
            **metricas,
            "matriz_confusion": cm,
            "tipo_features": tipo_features,
            "fecha_entrenamiento": datetime.now(),
        })

        return {
            "ok": True, "ticker": t, "modelo": "SVC",
            "accuracy": metricas["accuracy"], "f1": metricas["f1"], "senal": senal_actual,
            "mensaje": f"SVC entrenado: acc={metricas['accuracy']:.0%} | f1={metricas['f1']:.0%}",
        }

    except Exception as e:
        return {"ok": False, "ticker": ticker, "modelo": "SVC", "mensaje": f"Error entrenando SVC: {e}"}


def entrenar_modelo_rnn(ticker: str, arquitectura: str, epochs: int = EPOCHS_CLOUD) -> Dict[str, Any]:
    """
    Entrena una RNN (LSTM/BiLSTM/GRU/SimpleRNN — misma arquitectura que su
    Notebook correspondiente) sobre secuencias construidas a partir de
    `precios_ohlcv`, y guarda la predicción actual + métricas + histórico
    de señales en MongoDB. `epochs` se reduce por defecto a 5 (Notebooks
    originales: 80 épocas con EarlyStopping) para Streamlit Community Cloud.

    Retorna un dict {ok, ticker, modelo, accuracy, f1, senal, mensaje}.
    """
    try:
        from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                      precision_score, recall_score)
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import (GRU, LSTM, Bidirectional, Dense, Dropout,
                                              Input, SimpleRNN)
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam

        t = _validar_ticker(ticker)
        arq = (arquitectura or "").strip()
        if arq not in MODELOS_RNN_VALIDOS:
            return {"ok": False, "ticker": t, "modelo": arq,
                    "mensaje": f"Arquitectura '{arq}' no reconocida. Válidas: {sorted(MODELOS_RNN_VALIDOS)}"}

        df = get_mercado_data(t)
        if df.empty or len(df) < 100:
            return {"ok": False, "ticker": t, "modelo": arq,
                    "mensaje": f"Datos insuficientes ({len(df)} registros). Ejecuta primero la ingesta."}

        if t == "VOLCABC1.LM":
            feats, target, fechas, precios = _preparar_features_volumen(df)
            tipo_features = "volumen"
        else:
            feats, target, fechas, precios = _preparar_features_precio(df)
            tipo_features = "precio"

        X, y, fechas_seq, precios_seq = _crear_secuencias(feats, target, fechas, precios, N_STEPS_RNN)
        if len(X) < 50:
            return {"ok": False, "ticker": t, "modelo": arq, "mensaje": "Secuencias insuficientes tras limpieza."}

        n = len(X)
        corte = int(n * 0.80)
        n_steps, n_features = X.shape[1], X.shape[2]

        scaler = MinMaxScaler()
        scaler.fit(X[:corte].reshape(-1, n_features))

        def _escalar(X_):
            forma = X_.shape
            return scaler.transform(X_.reshape(-1, n_features)).reshape(forma)

        X_train, X_test = _escalar(X[:corte]), _escalar(X[corte:])
        y_train, y_test = y[:corte], y[corte:]

        clases, conteos = np.unique(y_train, return_counts=True)
        total = len(y_train)
        pesos_clase = {int(c): total / (len(clases) * cnt) for c, cnt in zip(clases, conteos)}

        if arq == "LSTM":
            capas = [Input(shape=(n_steps, n_features)), LSTM(260, return_sequences=True),
                     Dropout(0.2), LSTM(130)]
        elif arq == "BiLSTM":
            capas = [Input(shape=(n_steps, n_features)), Bidirectional(LSTM(200, return_sequences=True)),
                     Dropout(0.3), Bidirectional(LSTM(100))]
        elif arq == "GRU":
            capas = [Input(shape=(n_steps, n_features)), GRU(280, return_sequences=True),
                     GRU(140, return_sequences=False)]
        else:  # SimpleRNN
            capas = [Input(shape=(n_steps, n_features)), SimpleRNN(180, return_sequences=True),
                     SimpleRNN(90)]
        capas.append(Dense(1, activation="sigmoid"))

        modelo = Sequential(capas)
        modelo.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

        early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0)
        historial = modelo.fit(
            X_train, y_train,
            epochs=epochs, batch_size=64,
            validation_split=0.15, class_weight=pesos_clase,
            callbacks=[early_stop], verbose=0,
        )

        y_prob = modelo.predict(X_test, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        metricas = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "epocas_entrenadas": int(len(historial.history["loss"])),
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        }
        cm = confusion_matrix(y_test, y_pred).tolist()
        hist_epocas = {
            "loss": [round(float(v), 4) for v in historial.history["loss"]],
            "accuracy": [round(float(v), 4) for v in historial.history.get("accuracy", [])],
            "val_loss": [round(float(v), 4) for v in historial.history.get("val_loss", [])],
            "val_accuracy": [round(float(v), 4) for v in historial.history.get("val_accuracy", [])],
        }

        X_esc_all = scaler.transform(X.reshape(-1, n_features)).reshape(X.shape)
        ultima_X = X_esc_all[-1:]
        prob_actual = float(modelo.predict(ultima_X, verbose=0)[0][0])
        pred_actual = int(prob_actual >= 0.5)
        senal_actual = "BUY" if pred_actual == 1 else "SELL"

        probas_all = modelo.predict(X_esc_all, verbose=0).flatten()
        preds_all = (probas_all >= 0.5).astype(int)
        historico_senales = [
            {
                "fecha": str(fechas_seq[i])[:10],
                "precio": round(float(precios_seq[i]), 4),
                "prediccion": "BUY" if preds_all[i] == 1 else "SELL",
                "probabilidad": round(float(probas_all[i]), 4),
            }
            for i in range(len(preds_all))
        ]

        pred_col = _coleccion(COL_PREDICCIONES)
        met_col = _coleccion(COL_METRICAS)
        if pred_col is None or met_col is None:
            return {"ok": False, "ticker": t, "modelo": arq, "mensaje": "Sin conexión a MongoDB Atlas."}

        pred_col.delete_many({"ticker": t, "modelo": arq})
        pred_col.insert_one({
            "ticker": t, "modelo": arq,
            "senal": senal_actual,
            "confianza": round(prob_actual if pred_actual == 1 else 1 - prob_actual, 4),
            "fecha_prediccion": datetime.now().strftime("%Y-%m-%d"),
            "historico_senales": historico_senales,
            "tipo_features": tipo_features, "n_steps": N_STEPS_RNN,
            "created_at": datetime.now(),
        })

        met_col.delete_many({"ticker": t, "modelo": arq})
        met_col.insert_one({
            "ticker": t, "modelo": arq,
            **metricas,
            "matriz_confusion": cm,
            "historial_epocas": hist_epocas,
            "tipo_features": tipo_features, "n_steps": N_STEPS_RNN,
            "fecha_entrenamiento": datetime.now(),
        })

        return {
            "ok": True, "ticker": t, "modelo": arq,
            "accuracy": metricas["accuracy"], "f1": metricas["f1"], "senal": senal_actual,
            "mensaje": (
                f"{arq} entrenado: acc={metricas['accuracy']:.0%} | f1={metricas['f1']:.0%} "
                f"| épocas={metricas['epocas_entrenadas']}"
            ),
        }

    except Exception as e:
        return {"ok": False, "ticker": ticker, "modelo": arquitectura, "mensaje": f"Error entrenando {arquitectura}: {e}"}
