# prediction_module.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import config
import utils

print("[PREDICTION] Loading prediction model artifacts...")
try:
    prediction_model = tf.keras.models.load_model(config.PREDICTION_MODEL_DIR / "prediction_model.keras")
    prediction_x_scaler = joblib.load(config.PREDICTION_MODEL_DIR / "prediction_x_scaler.joblib")
    prediction_y_scaler = joblib.load(config.PREDICTION_MODEL_DIR / "prediction_y_scaler.joblib")
    with open(config.PREDICTION_MODEL_DIR / "prediction_targets.json", 'r') as f:
        PREDICTION_MODEL_TARGETS = json.load(f)
    PREDICTION_HORIZONS = sorted(list(set([int(t.split('_')[-1].replace('m','')) for t in PREDICTION_MODEL_TARGETS])))
    print(f"[PREDICTION] Artifacts loaded. Prediction horizons: {PREDICTION_HORIZONS} minutes.")
except Exception as e:
    exit(f"[FATAL ERROR] Could not load prediction model artifacts: {e}")


def run_prediction(device_id, history_df, weather_df):
    """
    Esegue la previsione futura dello stato ambientale.
    Il modello non è più condizionato dalle azioni degli attuatori, quindi esegue
    una singola previsione basata sullo storico dei sensori e sulle previsioni meteo.
    """
    print("\n--- [STAGE 2: Prediction Inference] ---")

    # Ottieni i valori attuali come base per i calcoli con i delta
    latest_sensor_data = history_df.iloc[-1]
    current_temp = latest_sensor_data['temperature_sensor']
    current_hum_rel = latest_sensor_data['humidity_sensor']
    current_co2 = latest_sensor_data['co2']
    current_voc = latest_sensor_data['voc']
    # Calcola l'umidità assoluta attuale, poiché il modello predice il suo delta
    current_hum_abs = utils.calculate_absolute_humidity(current_temp, current_hum_rel)
    
    print("\n[PREDICTION] Running prediction...")
    try:
        # Il vettore di feature non dipende più dallo stato o dalla disponibilità degli attuatori.
        # Si assume che 'hod_sin' e 'hod_cos' siano ora generate da create_feature_vector
        # in base all'indice temporale di history_df, come da nuovo config.PREDICTION_FEATURES.
        feature_vector = utils.create_features_for_prediction_model(
            df_hist=history_df,
            weather_df=weather_df, 
            required_features_list=config.PREDICTION_FEATURES
        )
    except Exception as e:
        print(f"[ERROR] Feature calculation for prediction failed: {e}. Aborting.")
        return None, None

    # Scala le feature, esegui la predizione e de-scala i risultati
    X_scaled_pred = prediction_x_scaler.transform(feature_vector)
    y_pred_scaled = prediction_model.predict(X_scaled_pred, verbose=0)
    y_pred_unscaled = prediction_y_scaler.inverse_transform(y_pred_scaled)
    delta_results_df = pd.DataFrame(y_pred_unscaled, columns=PREDICTION_MODEL_TARGETS)
    
    predictions = []
    for h in PREDICTION_HORIZONS:
        try:
            # 1. Prendi i DELTA predetti dal modello
            delta_temp = float(delta_results_df[f'temperature_sensor_pred_{h}m'].iloc[0])
            delta_abs_hum = float(delta_results_df[f'absolute_humidity_sensor_pred_{h}m'].iloc[0])
            delta_co2 = float(delta_results_df[f'co2_pred_{h}m'].iloc[0])
            delta_voc = float(delta_results_df[f'voc_pred_{h}m'].iloc[0])

            # 2. Calcola i VALORI ASSOLUTI FUTURI sommando i delta ai valori attuali
            future_temp = current_temp + delta_temp
            future_abs_hum = current_hum_abs + delta_abs_hum
            future_co2 = current_co2 + delta_co2
            future_voc = current_voc + delta_voc

            # 3. Usa i valori futuri assoluti per calcolare le metriche di comfort
            future_rel_hum = utils.calculate_relative_humidity(future_temp, future_abs_hum)
            pred_heat_index = utils.calculate_heat_index(future_temp, future_rel_hum)
            iaq_score = utils.calculate_iaq_index(future_co2, future_voc)
            global_score = utils.calculate_global_comfort(pred_heat_index, iaq_score)

            # LOG COMPATTO PER LA PREVISIONE
            log_msg = (
                f"  [Pred @ {h}m] "
                f"T:{future_temp:.1f}°C, "
                f"RH:{future_rel_hum:.0f}%, "
                f"CO2:{future_co2:.0f}, "
                f"VOC:{future_voc:.0f} "
                f"-> Comfort:{global_score:.1f}%"
            )
            print(log_msg)

            predictions.append({
                "device_id": device_id, "horizon": f"{h}m", "is_suggestion_for": None,
                "temperature": round(future_temp, 2), "humidity": round(future_rel_hum, 2),
                "co2": round(future_co2, 0), "voc": round(future_voc, 0),
                "heatIndex": round(pred_heat_index, 2), "iaqIndex": round(iaq_score, 1),
                "globalComfort": round(global_score, 1) if global_score is not None else 0
            })
        except (KeyError, IndexError) as e:
            print(f"[ERROR] Could not process horizon {h}m: {e}")
    
    # Non essendoci più scenari, non viene eseguita l'analisi dei suggerimenti.
    # Restituisce le predizioni e None per mantenere la firma del return.
    print("\n[ANALYSIS] Prediction complete. No suggestion analysis performed.")
    return predictions, None