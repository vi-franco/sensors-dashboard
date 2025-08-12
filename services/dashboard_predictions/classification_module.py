# classification_module.py
# -*- coding: utf-8 -*-
import tensorflow as tf
import joblib
import config
import utils

print("[CLASSIFICATION] Loading classification model artifacts...")
try:
    # Percorsi identici al tuo originale
    actuator_model = tf.keras.models.load_model(config.ACTUATOR_MODEL_DIR / "model.keras")
    actuator_scaler = joblib.load(config.ACTUATOR_MODEL_DIR / "scaler.joblib")
    print("[CLASSIFICATION] Artifacts loaded successfully.")
except Exception as e:
    exit(f"[FATAL ERROR] Could not load classification model or scaler: {e}")

def run_classification(history_df, weather_df, available_actuators_it, last_known_states):
    """
    Esegue l'inferenza per determinare lo stato corrente degli attuatori.
    Usa lo storico sensori + meteo per creare le feature.
    - Isteresi: legge soglie globali dal DB (hysteresis_global); fallback a config.
    - Ritorna:
        current_states_it: dict con chiavi 'state_<IT>'
        status: "OK" o messaggio errore
        probs_en: dict con chiavi <EN> e valori [0..1] (per salvare prob_<EN> nella UI)
    """
    print("\n--- [STAGE 1: Current State Inference] ---")
    try:
        # Feature vector per classificazione (come nel tuo originale)
        feature_vector = utils.create_features_for_actuator_model(
            df_hist=history_df,
            weather_df=weather_df,
            available_actuators_it=available_actuators_it,
            actuator_states=None,  # non stiamo simulando scenari
            required_features_list=config.CLASSIFICATION_FEATURES
        )
    except Exception as e:
        print(f"[ERROR] Feature calculation for actuator model failed: {e}")
        return None, f"Feature calculation failed: {e}", None

    # Scala le feature e ottieni le probabilità dal modello
    X_scaled = actuator_scaler.transform(feature_vector)
    probabilities = actuator_model.predict(X_scaled, verbose=0)[0]  # shape: (n_actuators,)

    # Carica soglie isteresi GLOBALI dal DB (EN); fallback ai default in config (IT)
    hys_en = utils.load_hysteresis_map_en()  # es: {'AC': {'on':0.7,'off':0.4}, ...}

    current_states_it = {}
    probs_en = {}  # per salvare prob_<EN> nella latest_status
    log_states = []

    # Applica isteresi per ciascun attuatore (ordine IT come nel tuo modello)
    for i, actuator_it in enumerate(config.ALL_ACTUATORS_IT):
        prob = float(probabilities[i])

        # mapping IT -> EN per recuperare soglie dal DB
        actuator_en = config.ACTUATOR_MAP_IT_TO_EN.get(actuator_it, None)
        if actuator_en and actuator_en in hys_en:
            th_on = float(hys_en[actuator_en]["on"])
            th_off = float(hys_en[actuator_en]["off"])
        else:
            # fallback ai default originali in config (IT)
            ths = config.HYSTERESIS_THRESHOLDS.get(actuator_it, {'on': 0.7, 'off': 0.4})
            th_on = float(ths['on'])
            th_off = float(ths['off'])

        last_state = int(last_known_states.get(f"state_{actuator_it}", 0))

        # Isteresi: ON resta ON finché prob >= th_off; OFF va ON se prob >= th_on
        if last_state == 1:
            new_state = 1 if prob >= th_off else 0
        else:
            new_state = 1 if prob >= th_on else 0

        current_states_it[f'state_{actuator_it}'] = new_state
        if actuator_en:  # salva probability in EN per la UI
            probs_en[actuator_en] = prob

        print(f"  > {actuator_it:<15} | Prob: {prob*100:5.1f}% | "
              f"thr(on={th_on:.2f}, off={th_off:.2f}) | State: {last_state} -> {new_state}")
        log_states.append(f"{actuator_it}={new_state}")

    # Log riassuntivo degli stati finali decisi
    print(f"  [CLASSIFICATION] Final states: {', '.join(log_states)}")

    # Ritorna anche probs_en per poter salvare prob_<EN> in latest_status
    return current_states_it, "OK", probs_en
