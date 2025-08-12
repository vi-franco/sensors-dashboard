# action_suggestion_module.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import config
import utils

# ===== Caricamento artefatti del regressore condizionato dall'azione =====
print("[ACTION] Loading action-conditioned regressor artifacts...")
try:
    ACTION_MODEL_DIR = getattr(config, "ACTION_REGRESSOR_DIR", config.PREDICTION_MODEL_DIR)
    model = tf.keras.models.load_model(ACTION_MODEL_DIR / "action_conditioned_regressor.keras")
    x_scaler = joblib.load(ACTION_MODEL_DIR / "action_regressor_x_scaler.joblib")
    y_scaler = joblib.load(ACTION_MODEL_DIR / "action_regressor_y_scaler.joblib")

    with open(ACTION_MODEL_DIR / "action_regressor_features.json", "r") as f:
        ACTION_FEATURES = json.load(f)
    with open(ACTION_MODEL_DIR / "action_regressor_targets.json", "r") as f:
        ACTION_TARGETS = json.load(f)
    with open(ACTION_MODEL_DIR / "action_labels_onoff.json", "r") as f:
        ACTION_ONEHOT_COLS = json.load(f)

    ACTION_HORIZONS = sorted(list(set([int(t.split('_')[-1].replace('m','')) for t in ACTION_TARGETS])))
    print(f"[ACTION] Artifacts loaded. Horizons: {ACTION_HORIZONS} minutes.")
except Exception as e:
    raise SystemExit(f"[FATAL ERROR] Could not load action regressor artifacts: {e}")


def _prepare_action_features(history_df, weather_df, base_required_feats):
    """
    Prepara il vettore per il regressore azione:
    - calcola le stesse feature del training
    - aggiunge alias per colonne con suffissi (_x/_y) se il modello le richiede
    - NON inserisce le one-hot action_* (le settiamo negli scenari)
    """
    # 1) Calcola tutte le feature "fisiche"
    full_vec = utils.create_features_for_action_model(
        df_hist=history_df,
        weather_df=weather_df,
        required_features_list=[]
    )

    # 2) Alias per suffissi: se il modello richiede 'co2_x' ma noi abbiamo 'co2'
    aliases = {}
    for col in ACTION_FEATURES:
        if col in full_vec.columns:
            continue
        # mappa *_x / *_y => base
        if col.endswith("_x") or col.endswith("_y"):
            base = col[:-2]
            if base in full_vec.columns:
                aliases[col] = full_vec[base].values
        # mappa hour_* se il training salvò hod_*
        if col in ("hour_sin","hour_cos"):
            if col not in full_vec.columns:
                # già calcolate con hour_*, non serve alias
                pass

    if aliases:
        for k, v in aliases.items():
            full_vec[k] = v

    # 3) Colonne *_eval_* (se, per errore, sono finite tra le feature salvate): mettile a 0 e WARNING
    eval_like = [c for c in ACTION_FEATURES if "_eval_" in c]
    if eval_like:
        for c in eval_like:
            if c not in full_vec.columns:
                full_vec[c] = 0.0
        print(f"[ACTION][WARN] Il modello richiede colonne *_eval_* come feature: {eval_like[:6]}{'...' if len(eval_like)>6 else ''} (settate a 0)")

    # 4) is_augmented = 0 in inferenza se richiesto
    if "is_augmented" in ACTION_FEATURES and "is_augmented" not in full_vec.columns:
        full_vec["is_augmented"] = 0

    # 5) Rimuovi le action_* (le metteremo negli scenari); se il modello le richiede, saranno 0 qui
    action_like = [c for c in ACTION_FEATURES if c.startswith("action_")]
    for c in action_like:
        if c not in full_vec.columns:
            full_vec[c] = 0

    # 6) Ora reindex esattamente a ACTION_FEATURES
    missing = [c for c in ACTION_FEATURES if c not in full_vec.columns]
    if missing:
        raise ValueError(f"[ACTION] Mancano ancora colonne richieste dal modello: {missing[:10]}{'...' if len(missing)>10 else ''}")

    out = full_vec.reindex(columns=ACTION_FEATURES).copy()
    return out
    

def _predict_action_scenario(feature_row, current_state):
    """
    Esegue la predizione (deltæ) per un singolo scenario (feature_row = 1 riga),
    riconverte in valori assoluti usando lo stato corrente.
    """
    Xs = x_scaler.transform(feature_row[ACTION_FEATURES])
    y_pred_scaled = model.predict(Xs, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)[0]  # shape: (n_targets,)

    # Ricostruisci assoluti per ogni variabile/orizzonte
    results = {}
    for i, tgt in enumerate(ACTION_TARGETS):
        base = tgt.split("_pred_")[0]
        delta = float(y_pred[i])
        results.setdefault(base, {})
        results[base][tgt] = current_state.get(base, 0.0) + delta
    return results  # es: {'temperature_sensor': {'temperature_sensor_pred_15m': 24.2, ...}, ...}


def run_action_suggestions(device_id, history_df, weather_df, baseline_predictions, available_it, last_known_states):
    """
    Genera suggerimenti provando azioni ON/OFF (una alla volta).
    Mantiene le azioni che migliorano il comfort globale di almeno la soglia.
    Restituisce la lista 'winning_suggestions' per utils.save_predictions_and_suggestions_to_db().
    """
    if history_df is None or history_df.empty:
        print("[ACTION] No history provided. Skipping.")
        return None

    # Stato attuale per ricostruire assoluti (temp, AH, CO2, VOC)
    latest = history_df.iloc[-1]
    current_temp = float(latest.get('temperature_sensor', np.nan))
    current_rh   = float(latest.get('humidity_sensor', np.nan))
    current_co2  = float(latest.get('co2', np.nan))
    current_voc  = float(latest.get('voc', np.nan))
    current_ah   = utils.calculate_absolute_humidity(current_temp, current_rh)

    current_state = {
        "temperature_sensor": current_temp,
        "absolute_humidity_sensor": current_ah,
        "co2": current_co2,
        "voc": current_voc,
    }

    # Prepara il vettore base (senza azione)
    base_vec = _prepare_action_features(
        history_df=history_df,
        weather_df=weather_df,
        base_required_feats=getattr(config, "PREDICTION_FEATURES", [])
    )

    feat_mismatch = [c for c in ACTION_FEATURES if c not in base_vec.columns]
    if feat_mismatch:
        raise SystemExit(f"[ACTION][FATAL] Feature mismatch vs scaler/features.json: {feat_mismatch[:10]}...")

    try:
        noact_abs = _predict_action_scenario(base_vec.iloc[[0]], current_state)
        dbg = []
        for h in ACTION_HORIZONS:
            # temp
            t_noact = noact_abs["temperature_sensor"].get(f"temperature_sensor_pred_{h}m", current_temp)
            t_base  = next((p["temperature"] for p in (baseline_predictions or []) if p["horizon"]==f"{h}m"), None)
            if t_base is not None and abs(t_noact - t_base) > 1.5:  # soglia debug 1.5°C
                dbg.append(f"T@{h}m Δ={t_noact - t_base:.2f}")
        if dbg:
            print("[ACTION][WARN] No-action != baseline:", ", ".join(dbg))
    except Exception as e:
        print(f"[ACTION][WARN] Sanity no-action check failed: {e}")

    # Determina quali colonne one-hot per le azioni sono realmente nel modello
    action_cols_present = [c for c in ACTION_ONEHOT_COLS if c in base_vec.columns]
    if not action_cols_present:
        print("[ACTION] No action one-hot columns found in features. Aborting action suggestions.")
        return None

    # Baseline comfort per confronto (arriva dal modulo prediction_module)
    baseline_by_h = {int(rec["horizon"].replace("m","")): float(rec["globalComfort"]) for rec in (baseline_predictions or [])}

    # Candidati: per ciascun attuatore disponibile (IT), prova ON e OFF se ha senso rispetto allo stato noto
    candidates = []
    for act_it in available_it:
        cur = int(last_known_states.get(f"state_{act_it}", 0))
        for direction in ("on", "off"):
            # Evita suggerire lo stesso stato
            if (direction == "on" and cur == 1) or (direction == "off" and cur == 0):
                continue
            col_name = f"action_{act_it}_{direction}"
            if col_name not in action_cols_present:
                continue
            candidates.append((act_it, direction))

    if not candidates:
        print("[ACTION] No candidate actions (all states already desired or not supported).")
        return None

    # Prova ogni scenario (UNA azione alla volta)
    suggestions = []
    for act_it, direction in candidates:
        feats = base_vec.copy()

        action_col = f"action_{act_it}_{direction}"

        feats.loc[:, action_cols_present] = 0
        feats.loc[:, [action_col]] = 1

        # Predici deltas -> assoluti
        abs_future = _predict_action_scenario(feats.iloc[[0]], current_state)

        # Costruisci predictions + calcola comfort per ogni orizzonte
        scenario_predictions = []
        best_improvement = -1e9
        for h in ACTION_HORIZONS:
            f_temp = abs_future["temperature_sensor"].get(f"temperature_sensor_pred_{h}m", current_temp)
            f_ah   = abs_future["absolute_humidity_sensor"].get(f"absolute_humidity_sensor_pred_{h}m", current_ah)
            f_co2  = abs_future["co2"].get(f"co2_pred_{h}m", current_co2)
            f_voc  = abs_future["voc"].get(f"voc_pred_{h}m", current_voc)

            f_rh = utils.calculate_relative_humidity(f_temp, f_ah)
            heat_index = utils.calculate_heat_index(f_temp, f_rh)
            iaq = utils.calculate_iaq_index(f_co2, f_voc)
            gcomfort = utils.calculate_global_comfort(heat_index, iaq)

            # baseline comfort per lo stesso orizzonte
            base_gc = baseline_by_h.get(h, None)
            improvement = (gcomfort - base_gc) if (gcomfort is not None and base_gc is not None) else -1e9
            best_improvement = max(best_improvement, improvement)

            scenario_predictions.append({
                "device_id": device_id,
                "horizon": f"{h}m",
                "is_suggestion_for": None,  # sarà valorizzato in save_...
                "temperature": round(f_temp, 2),
                "humidity": round(f_rh if f_rh is not None else 0, 2),
                "co2": round(f_co2, 0),
                "voc": round(f_voc, 0),
                "heatIndex": round(heat_index if heat_index is not None else 0, 2),
                "iaqIndex": round(iaq if iaq is not None else 0, 1),
                "globalComfort": round(gcomfort, 1) if gcomfort is not None else 0.0
            })

        suggestions.append({
            "actuator_it": act_it,
            "direction": direction,
            "best_improvement": best_improvement,
            "predictions": scenario_predictions
        })

    # Filtra per soglia e ordina per miglioramento
    thresh = float(getattr(config, "SUGGESTION_COMFORT_IMPROVEMENT_THRESHOLD", 5.0))
    filtered = [s for s in suggestions if s["best_improvement"] >= thresh]
    filtered.sort(key=lambda x: x["best_improvement"], reverse=True)

    if not filtered:
        print(f"[ACTION] No suggestions exceed threshold (+{thresh} comfort).")
        return None

    # Converte nel formato atteso da utils.save_predictions_and_suggestions_to_db
    winning_suggestions = []
    for s in filtered[:5]:  # cap a 5 suggerimenti
        # stati suggeriti: togglia solo l’attuatore corrente, lascia gli altri come sono
        states = {f"state_{it}": int(last_known_states.get(f"state_{it}", 0)) for it in config.ALL_ACTUATORS_IT}
        states[f"state_{s['actuator_it']}"] = 1 if s["direction"] == "on" else 0

        winning_suggestions.append({
            "states": states,                 # IT keys
            "predictions": s["predictions"]   # lista di record predictions
        })

    print(f"[ACTION] Selected {len(winning_suggestions)} suggestion(s).")
    return winning_suggestions
