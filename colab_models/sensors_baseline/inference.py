# inference_prediction.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = (CURRENT_DIR / "../..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.feature_engineering import ensure_min_columns_baseline_prediction, add_features_baseline_prediction, add_targets_baseline_prediction, final_features_baseline_prediction

MODEL_DIR = PROJECT_ROOT / "colab_models" / "sensors_baseline" / "output"

def run_prediction_inference(history_df: pd.DataFrame) -> Tuple[Optional[Dict[str, Any]], str]:

    try:
        model = tf.keras.models.load_model(MODEL_DIR / "prediction_model.keras")
        x_scaler = joblib.load(MODEL_DIR / "prediction_x_scaler.joblib")
        y_scaler = joblib.load(MODEL_DIR / "prediction_y_scaler.joblib")

        with open(MODEL_DIR / "prediction_features.json", 'r', encoding='utf-8') as f:
            feature_names = json.load(f)
        with open(MODEL_DIR / "prediction_targets.json", 'r', encoding='utf-8') as f:
            target_names = json.load(f)

        print(f"[SETUP] Modello di previsione caricato con successo da '{MODEL_DIR}'")

    except Exception as e:
        return None, f"Impossibile caricare il modello di previsione: {e}"


    try:
        fe_full = add_features_baseline_prediction(history_df.copy())
        fe_last = fe_full.tail(1)
        feature_row = fe_last.reindex(columns=feature_names).fillna(0.0)

    except Exception as e:
        return None, f"Calcolo feature per la previsione fallito: {e}"

    try:
        X_scaled = x_scaler.transform(feature_row)
        y_pred_scaled = model.predict(X_scaled, verbose=0)
        y_pred_deltas = y_scaler.inverse_transform(y_pred_scaled)[0]

    except Exception as e:
        return None, f"Inferenza del modello di previsione fallita: {e}"

    predictions: Dict[str, Dict[str, float]] = {}

    for i, target_name in enumerate(target_names):
        predicted_delta = y_pred_deltas[i]

        match = re.match(r'(.+)_pred_(\d+)m', target_name)
        if not match:
            continue

        base_col, horizon = match.groups()
        horizon_key = f"{horizon}min"

        current_value = fe_last[base_col].iloc[0]
        predicted_absolute_value = current_value + predicted_delta

        if horizon_key not in predictions:
            predictions[horizon_key] = {}
        predictions[horizon_key][base_col] = round(predicted_absolute_value, 3)

    return predictions, "OK"