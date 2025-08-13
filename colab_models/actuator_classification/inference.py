# inference_logic.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.feature_engineering import ensure_min_columns_actuator_classification, add_features_actuator_classification
from colab_models.common import get_actuator_names

def run_inference(history_df: pd.DataFrame) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, float]], str]:
    try:
        script_dir = Path(__file__).resolve().parent
        model_dir = script_dir / "output"

        model_path = model_dir / "model.keras"
        scaler_path = model_dir / "scaler.joblib"
        features_path = model_dir / "features.json"
        thresholds_path = model_dir / "thresholds.json"

        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

        with open(features_path, 'r', encoding='utf-8') as f:
            feature_names = json.load(f)

        with open(thresholds_path, 'r', encoding='utf-8') as f:
            thresholds = json.load(f)

        print(f"[SETUP] Modello caricato con successo da '{model_dir}'")

    except Exception as e:
        return None, None, f"Impossibile caricare il modello: {e}"

    try:
        ensure_min_columns_actuator_classification(history_df)
        fe_df = add_features_actuator_classification(history_df.copy())
        feature_row = fe_df.reindex(columns=feature_names).fillna(0.0)
        feature_row = feature_row.astype(np.float32)
    except Exception as e:
        return None, None, f"Calcolo feature fallito: {e}"

    try:
        X_scaled = scaler.transform(feature_row)
        raw_probs = model.predict(X_scaled, verbose=0)[0]
        probs = np.nan_to_num(raw_probs, nan=0.0, posinf=1.0, neginf=0.0)
    except Exception as e:
        return None, None, f"Inferenza del modello fallita: {e}"

    current_states: Dict[str, int] = {}
    probabilities: Dict[str, float] = {}

    actuator_names_it = get_actuator_names()
    for i, actuator_name in enumerate(actuator_names_it):
        prob = float(probs[i])
        threshold = thresholds.get(actuator_name, 0.5)
        state = 1 if prob >= threshold else 0
        current_states[f"{actuator_name}"] = state
        probabilities[actuator_name] = prob

    return current_states, probabilities, "OK"