# inference.py
# -*- coding: utf-8 -*-
import sqlite3
import os
from datetime import datetime, timezone
import config
import utils
import classification_module
import prediction_module
import action_suggestion_module

if __name__ == "__main__":
    print(f"\n--- Starting Unified Inference Script: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    if not os.path.exists(config.DB_PATH):
        exit(f"[FATAL ERROR] Database file not found at '{config.DB_PATH}'. Please run the setup script first.")

    # Leggi device
    with sqlite3.connect(config.DB_PATH) as con:
        con.row_factory = sqlite3.Row
        devices = con.execute(
            "SELECT device_id, room_name, location_name, owm_lat as latitude, owm_lon as longitude FROM devices"
        ).fetchall()
        print(f"[DB] Found {len(devices)} devices to process.")

    for device in devices:
        device_id = device['device_id']
        lat = device['latitude']; lon = device['longitude']
        print(f"\n{'='*20} Processing Device: {device_id} at ({lat}, {lon}) {'='*20}")

        # 1) Meteo storico
        weather_history_df = utils.manage_weather_history(device_id=device_id, lat=lat, lon=lon)
        if weather_history_df is None or weather_history_df.empty:
            print(f"[ERROR] No weather data for {device_id}. Skipping.")
            continue

        # 2) History + attuatori disponibili
        status = {'level': 0, 'message': 'OK'}
        with sqlite3.connect(config.DB_PATH) as con:
            con.row_factory = sqlite3.Row
            act_rows = con.execute(
                "SELECT actuator_name FROM device_actuators WHERE device_id = ? AND is_enabled = 1",
                (device_id,)
            ).fetchall()
            available_en = [row['actuator_name'] for row in act_rows]
        en_to_it = config.ACTUATOR_MAP_EN_TO_IT
        available_it = [en_to_it[a] for a in available_en if a in en_to_it]

        last_known_states = utils.get_last_known_states(device_id)
        history_df, status = utils.get_sensor_history(device_id, status)
        if history_df is None or history_df.empty:
            print(f"[ERROR] {status.get('message','No history')} for {device_id}. Skipping.")
            continue

        # 3) Classificazione (stati IT + probabilità EN)
        states_it, class_status, probs_en = classification_module.run_classification(
            history_df, weather_history_df, available_it, last_known_states
        )
        if states_it is None:
            print(f"[ERROR] Classification failed for {device_id}: {class_status}. Skipping.")
            continue

        # 4) Salva stato corrente (sensori + state_<EN> + prob_<EN>)
        latest = history_df.iloc[-1]
        heat_index = utils.calculate_heat_index(latest.get('temperature_sensor'), latest.get('humidity_sensor'))
        iaq = utils.calculate_iaq_index(latest.get('co2'), latest.get('voc'))
        record = {
            'device_id': device_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'temperature': latest.get('temperature_sensor'),
            'humidity':    latest.get('humidity_sensor'),
            'co2':         max(float(latest.get('co2', 0) or 0), float(getattr(config, "MIN_CO2", 400.0))),
            'voc':         max(float(latest.get('voc', 0) or 0), float(getattr(config, "MIN_VOC", 0.0))),
            'heatIndex':   heat_index,
            'iaqIndex':    iaq,
            'globalComfort': utils.calculate_global_comfort(heat_index, iaq),
            'status_level': status['level'],
            'status_message': status['message'],
        }
        # IT -> EN per colonne state_<EN>
        for it, en in config.ACTUATOR_MAP_IT_TO_EN.items():
            record[f'state_{en}'] = int(states_it.get(f'state_{it}', 0))
        # prob_<EN> per UI
        if probs_en:
            for en, p in probs_en.items():
                record[f'prob_{en}'] = float(p)

        utils.save_results_to_db(record)

        # 5) Predizioni e suggerimenti
        predictions, _ = prediction_module.run_prediction(
            device_id, history_df, weather_history_df
        )

        winning_suggestions = action_suggestion_module.run_action_suggestions(
            device_id=device_id,
            history_df=history_df,
            weather_df=weather_history_df,
            baseline_predictions=predictions,
            available_it=available_it,
            last_known_states=last_known_states
        )
        
        if predictions:
            utils.save_predictions_and_suggestions_to_db(device_id, predictions, winning_suggestions)
            utils.save_prediction_timestamp(device_id)
        else:
            print(f"[ANALYSIS] No predictions for {device_id}.")



    print(f"\n--- Unified Inference run complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
