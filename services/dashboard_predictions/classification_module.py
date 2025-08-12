# classification_module.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Import interni al progetto (assumi che la root del repo sia già nel PYTHONPATH)
import config
from utils.feature_engineering import (
    ensure_min_columns_actuator_classification,
    add_features_actuator_classification,
)
from utils.functions import create_features_for_actuator_model
from colab_models.common import load_hysteresis_map_en  # se sta qui nel tuo repo

# --------------------------------------------------------------------------------------
# Caricamento artefatti
# --------------------------------------------------------------------------------------
print("[CLASSIFICATION] Loading classification model artifacts...")

MODEL_DIR: Path = config.ACTUATOR_MODEL_DIR
MODEL_PATH: Path = MODEL_DIR / "model.keras"
SCALER_PATH: Path = MODEL_DIR / "scaler.joblib"
FEATURES_JSON: Path = MODEL_DIR / "features.json"

try:
    actuator_model = tf.keras.models.load_model(MODEL_PATH)
    actuator_scaler = joblib.load(SCALER_PATH)
    print(f"[CLASSIFICATION] Model loaded: {MODEL_PATH}")
    print(f"[CLASSIFICATION] Scaler loaded: {SCALER_PATH}")
except Exception as e:
    raise SystemExit(f"[FATAL] Could not load model/scaler: {e}")

# Carica la lista delle feature usate in training (se disponibile)
if FEATURES_JSON.exists():
    try:
        with open(FEATURES_JSON, "r") as f:
            TRAIN_FEATURES = list(json.load(f))
        print(f"[CLASSIFICATION] Loaded {len(TRAIN_FEATURES)} features from features.json")
    except Exception as e:
        print(f"[WARN] Could not read features.json: {e}. Falling back to config.")
        TRAIN_FEATURES = list(config.CLASSIFICATION_FEATURES)
else:
    TRAIN_FEATURES = list(config.CLASSIFICATION_FEATURES)
    print(f"[CLASSIFICATION] Using {len(TRAIN_FEATURES)} features from config.")

# Mapping e attuatori
ALL_ACTUATORS_IT = list(config.ALL_ACTUATORS_IT)
ACTUATOR_MAP_IT_TO_EN = dict(config.ACTUATOR_MAP_IT_TO_EN)


# --------------------------------------------------------------------------------------
# Funzioni di supporto
# --------------------------------------------------------------------------------------
def _build_feature_frame(
    history_df: pd.DataFrame,
    weather_df: Optional[pd.DataFrame],
    available_actuators_it: Dict[str, int],
    required_features: list[str],
) -> pd.DataFrame:
    """
    Costruisce un DataFrame di una riga con le feature nell'ordine atteso dal modello.
    Usa le tue funzioni condivise di feature engineering.
    """
    # (1) Verifiche minime e calcolo feature “causali”
    ensure_min_columns_actuator_classification(history_df)
    fe_df = add_features_actuator_classification(history_df.copy())

    # (2) Crea il vettore finale come nel training (usa util condivisa)
    feat_row: pd.DataFrame = create_features_for_actuator_model(
        df_hist=fe_df,
        weather_df=weather_df,
        available_actuators_it=available_actuators_it,
        actuator_states=None,  # in classificazione corrente non simuli scenari
        required_features_list=required_features,
    )

    # (3) Allineamento colonne e gestione NaN
    #    - reindicizza per assicurare l'ordine esatto richiesto dal modello/scaler
    #    - riempi eventuali NaN (es. meteo mancante)
    feat_row = feat_row.reindex(columns=required_features)
    feat_row = feat_row.fillna(0.0)

    # (4) dtype coerente
    return feat_row.astype(np.float32)


def _apply_hysteresis(
    probs: np.ndarray,
    last_known_states: Dict[str, int],
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Applica l'isteresi attuatore-specifica usando soglie globali (EN) se presenti
    oppure i fallback in config (IT). Ritorna (stati_IT, probs_EN).
    """
    # Soglie dall'archivio (EN), es.: {'AC': {'on':0.7,'off':0.4}, ...}
    try:
        hys_en = load_hysteresis_map_en()
    except Exception as e:
        print(f"[WARN] Could not load hysteresis from DB. Using config defaults. Err: {e}")
        hys_en = {}

    current_states_it: Dict[str, int] = {}
    probs_en: Dict[str, float] = {}

    for i, actuator_it in enumerate(ALL_ACTUATORS_IT):
        prob = float(probs[i])
        actuator_en = ACTUATOR_MAP_IT_TO_EN.get(actuator_it)

        if actuator_en and actuator_en in hys_en:
            th_on = float(hys_en[actuator_en].get("on", 0.7))
            th_off = float(hys_en[actuator_en].get("off", 0.4))
        else:
            ths = config.HYSTERESIS_THRESHOLDS.get(actuator_it, {"on": 0.7, "off": 0.4})
            th_on = float(ths["on"])
            th_off = float(ths["off"])

        last_state = int(last_known_states.get(f"state_{actuator_it}", 0))

        # Regola isteresi: se era ON, resta ON finché prob>=off; se era OFF, va ON quando prob>=on
        if last_state == 1:
            new_state = 1 if prob >= th_off else 0
        else:
            new_state = 1 if prob >= th_on else 0

        current_states_it[f"state_{actuator_it}"] = new_state
        if actuator_en:
            probs_en[actuator_en] = prob

        print(
            f"  > {actuator_it:<15} | Prob: {prob:6.3f} | "
            f"thr(on={th_on:.2f}, off={th_off:.2f}) | State: {last_state} -> {new_state}"
        )

    return current_states_it, probs_en


# --------------------------------------------------------------------------------------
# API principale
# --------------------------------------------------------------------------------------
def run_classification(
    history_df: pd.DataFrame,
    weather_df: Optional[pd.DataFrame],
    available_actuators_it: Dict[str, int],
    last_known_states: Dict[str, int],
) -> Tuple[Optional[Dict[str, int]], str, Optional[Dict[str, float]]]:
    """
    Esegue l'inferenza degli stati attuatori correnti.

    Parametri
    ---------
    history_df : pd.DataFrame
        Storico locale dei sensori (ordinato, con timestamp).
    weather_df : Optional[pd.DataFrame]
        Dati meteo allineati/aggregati (può essere None).
    available_actuators_it : Dict[str, int]
        Mappa IT -> 0/1 di disponibilità (Humidifier=1 se presente, ecc.).
    last_known_states : Dict[str, int]
        Stato precedente (isteresi), chiavi 'state_<IT>'.

    Ritorna
    -------
    current_states_it : dict | None
        Stato attuale per IT, es. {'state_Finestra': 0, ...}
    status : str
        "OK" oppure messaggio d'errore.
    probs_en : dict | None
        Probabilità per EN (per UI), es. {'AC': 0.73, ...}
    """
    print("\n--- [STAGE 1: Current State Inference] ---")

    # 1) Costruisci il vettore feature
    try:
        feature_row = _build_feature_frame(
            history_df=history_df,
            weather_df=weather_df,
            available_actuators_it=available_actuators_it,
            required_features=TRAIN_FEATURES,
        )
    except Exception as e:
        print(f"[ERROR] Feature calculation failed: {e}")
        return None, f"Feature calculation failed: {e}", None

    # 2) Scala e predici
    try:
        X_scaled = actuator_scaler.transform(feature_row)
        probs = actuator_model.predict(X_scaled, verbose=0)[0]  # shape: (n_outputs,)
        # Paracadute contro NaN/inf (robustezza)
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return None, f"Inference failed: {e}", None

    # 3) Isteresi
    try:
        current_states_it, probs_en = _apply_hysteresis(probs, last_known_states)
    except Exception as e:
        print(f"[ERROR] Hysteresis application failed: {e}")
        return None, f"Hysteresis failed: {e}", None

    # 4) Log finale
    log_states = [f"{k.replace('state_','')}={v}" for k, v in current_states_it.items()]
    print(f"  [CLASSIFICATION] Final states: {', '.join(log_states)}")

    return current_states_it, "OK", probs_en
