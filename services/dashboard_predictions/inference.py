# inference.py
# -*- coding: utf-8 -*-
from pathlib import Path

import sqlite3
import sys
import os
from datetime import datetime, timezone
import config
import helpers

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = (CURRENT_DIR / "../..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from colab_models.actuator_classification.inference import run_inference as run_classification_inference

if __name__ == "__main__":
    print(f"\n--- Starting inference: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

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
        status = {'level': 0, 'message': 'OK'}

        history_df, status = helpers.get_sensor_history(device_id, status)
        if history_df is None or history_df.empty:
            print(f"[ERROR] {status.get('message','No history')} for {device_id}. Skipping.")
            continue

        weather_history_df = helpers.manage_weather_history(device_id=device_id, lat=lat, lon=lon)
        if weather_history_df is None or weather_history_df.empty:
            print(f"[ERROR] No weather data for {device_id}. Skipping.")
            continue

        weather_history_df.index.name = 'utc_datetime'
        merged_df = history_df.join(weather_history_df, how='left')
        merged_df = merged_df.ffill().bfill()

        print(f"[DEBUG] Merged DataFrame for {device_id}:\n{merged_df.head()}")

        states, probs, class_status = run_classification_inference(merged_df)
        if states is None:
            print(f"[ERROR] Classification failed for {device_id}: {class_status}. Skipping.")
            continue

        last_known_states = helpers.get_last_known_states(device_id)

        with sqlite3.connect(config.DB_PATH) as con:
            con.row_factory = sqlite3.Row
            act_rows = con.execute(
                "SELECT actuator_name FROM device_actuators WHERE device_id = ? AND is_enabled = 1",
                (device_id,)
            ).fetchall()
            actuators = [row['actuator_name'] for row in act_rows]

        latest = history_df.iloc[-1]
        heat_index = helpers.calculate_heat_index(latest.get('temperature_sensor'), latest.get('humidity_sensor'))
        iaq = helpers.calculate_iaq_index(latest.get('co2'), latest.get('voc'))
        record = {
            'device_id': device_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'temperature': latest.get('temperature_sensor'),
            'humidity':    latest.get('humidity_sensor'),
            'co2':         max(float(latest.get('co2', 0) or 0), float(getattr(config, "MIN_CO2", 400.0))),
            'voc':         max(float(latest.get('voc', 0) or 0), float(getattr(config, "MIN_VOC", 0.0))),
            'heatIndex':   heat_index,
            'iaqIndex':    iaq,
            'globalComfort': helpers.calculate_global_comfort(heat_index, iaq),
            'status_level': status['level'],
            'status_message': status['message'],
        }

        for actuator in get_actuator_names():
            record[f'state_{actuator}'] = int(states.get(f'state_{actuator}', 0))
        if probs:
            for actuator, p in probs.items():
                record[f'prob_{actuator}'] = float(p)

        helpers.save_results_to_db(record)

        # 5) Predizioni e suggerimenti
        #predictions, _ = prediction_module.run_prediction(
        #    device_id, history_df, weather_history_df
        #)

        #winning_suggestions = action_suggestion_module.run_action_suggestions(
        #    device_id=device_id,
        #    history_df=history_df,
        #    weather_df=weather_history_df,
        #    baseline_predictions=predictions,
        #    available_it=available_it,
        #    last_known_states=last_known_states
        #)
        
        #if predictions:
        #    helpers.save_predictions_and_suggestions_to_db(device_id, predictions, winning_suggestions)
        #    helpers.save_prediction_timestamp(device_id)
        #else:
        #    print(f"[ANALYSIS] No predictions for {device_id}.")



    print(f"\n--- Unified Inference run complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
