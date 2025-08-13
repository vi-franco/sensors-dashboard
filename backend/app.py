# app.py
import sqlite3
import json
import time
import requests
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS

from colab_models.common import get_actuator_names

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = (CURRENT_DIR / "../..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "backend" / 'database.db'
ACTUATORS = get_actuator_names()

OWM_API_KEY = "d71435a2c59c063aaddc1332c9f226be"
OWM_CACHE = {}
OWM_CACHE_SECONDS = 15 * 60

app = Flask(__name__)
CORS(app)

# --- DATABASE & API HELPERS ---
def get_db_connection():
    """Establishes a connection to the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def table_exists(conn, name):
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,))
    return cur.fetchone() is not None

def col_exists(conn, table, col):
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        cols = [r["name"] for r in cur.fetchall()]
        return col in cols
    except Exception:
        return False

def load_hysteresis_map_en(conn):
    """
    Ritorna dict EN -> {on, off} dalle soglie globali (se esiste la tabella).
    Se assente, ritorna {} (la UI mostrerà nulla o default lato FE).
    """
    if not table_exists(conn, "hysteresis_global"):
        return {}
    rows = conn.execute("SELECT actuator_name, th_on, th_off FROM hysteresis_global").fetchall()
    return {r["actuator_name"]: {"on": float(r["th_on"]), "off": float(r["th_off"])} for r in rows} if rows else {}

def upsert_hysteresis_en(conn, act_en, th_on, th_off):
    """
    Aggiorna/crea le soglie globali per un attuatore EN.
    Richiede tabella hysteresis_global. Valida i range: 0 <= off < on <= 1
    """
    th_on = float(th_on); th_off = float(th_off)
    if not (0.0 <= th_off < th_on <= 1.0):
        raise ValueError("Soglie non valide: deve valere 0 <= OFF < ON <= 1")
    if not table_exists(conn, "hysteresis_global"):
        raise RuntimeError("La tabella 'hysteresis_global' non esiste. Esegui lo script di init DB.")
    conn.execute("""
        INSERT INTO hysteresis_global (actuator_name, th_on, th_off, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(actuator_name) DO UPDATE SET
            th_on=excluded.th_on, th_off=excluded.th_off, updated_at=excluded.updated_at
    """, (act_en, th_on, th_off, datetime.utcnow().isoformat()))
    conn.commit()

def get_weather_for_device(lat, lon):
    """Fetches weather data from OpenWeatherMap, using a cache."""
    if not lat or not lon: return None
    cache_key = f"{lat},{lon}"
    if cache_key in OWM_CACHE and (time.time() - OWM_CACHE[cache_key]['timestamp']) < OWM_CACHE_SECONDS:
        return OWM_CACHE[cache_key]['data']
    
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        weather_data = {
            "location": data.get('name', 'Unknown'),
            "temperature": data['main']['temp'],
            "humidity": data['main']['humidity'],
            "wind": data['wind']['speed'],
            "icon": data['weather'][0]['main'].lower()
        }
        OWM_CACHE[cache_key] = {'timestamp': time.time(), 'data': weather_data}
        return weather_data
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Weather API call failed: {e}")
        return None

def generate_suggestion_text(current_states, suggested_states):
    """Generates a human-readable action text from actuator state changes."""
    actions = []
    for act_en in ACTUATORS:
        state_key_db = f'state_{act_en}'
        act_it = ACTUATOR_MAP_EN_TO_IT.get(act_en, act_en)
        
        # 'current_states' è un dict di booleans/oggetti (UI): possiamo considerare 'state' se presente
        cur_val = current_states.get(act_en)
        if isinstance(cur_val, dict) and "state" in cur_val:
            current = bool(cur_val["state"])
        else:
            current = bool(cur_val)

        suggested = suggested_states.get(state_key_db, 0)

        if bool(current) != bool(suggested):
            action = "Accendi" if suggested == 1 else "Spegni"
            actions.append(f"{action} {act_it}")
    return ", ".join(actions) if actions else "Mantieni impostazioni correnti"

# --- MAIN DATA GATHERING FUNCTION ---
def get_full_data():
    """Queries the database and assembles the complete JSON payload for the UI."""
    with get_db_connection() as conn:
        devices_rows = conn.execute("SELECT * FROM devices").fetchall()
        
        # Precarica hysteresis globale (se presente)
        hysteresis_map = load_hysteresis_map_en(conn)

        devices_payload = []
        for device in devices_rows:
            device_id = device['device_id']
            device_data = dict(device)
            device_data.update({"current": {}, "predictions": {}, "suggestions": [], "history": [], "actuators": {}})

            # available_actuators (dalla tabella device_actuators)
            act_rows = conn.execute(
                "SELECT actuator_name, is_enabled FROM device_actuators WHERE device_id = ?",
                (device_id,)
            ).fetchall()
            device_data['available_actuators'] = {row['actuator_name']: bool(row['is_enabled']) for row in act_rows}

            # latest_status (stato, sensori, prob_* se presenti)
            status = conn.execute("SELECT * FROM latest_status WHERE device_id = ?", (device_id,)).fetchone()
            if status:
                # ✅ converto una volta sola e riuso
                status_dict = dict(status)

                device_data['current'] = status_dict

                # costruiamo 'actuators' come oggetti {state, prob, thresholds}
                actuators_payload = {}
                for act in ACTUATORS:
                    state_key = f'state_{act}'
                    prob_key  = f'prob_{act}'

                    state_val = bool(status_dict.get(state_key, 0))
                    # prob_* potrebbe non esserci come colonna: dict.get() copre tutto
                    prob_val = status_dict.get(prob_key, None)

                    thresholds = hysteresis_map.get(act) if hysteresis_map else None
                    actuators_payload[act] = {"state": state_val, "prob": prob_val, "thresholds": thresholds}

                device_data['actuators'] = actuators_payload

            # predictions baseline
            if table_exists(conn, "predictions"):
                pred_rows = conn.execute(
                    "SELECT * FROM predictions WHERE device_id = ? AND is_suggestion_for IS NULL",
                    (device_id,)
                ).fetchall()
                device_data['predictions'] = {row['horizon']: dict(row) for row in pred_rows}

            # suggestions + relative predictions
            if table_exists(conn, "suggestions"):
                sugg_rows = conn.execute(
                    "SELECT * FROM suggestions WHERE device_id = ?",
                    (device_id,)
                ).fetchall()
                for sugg in sugg_rows:
                    sugg_id = sugg['suggestion_id']
                    action_text = generate_suggestion_text(device_data['actuators'], dict(sugg))
                    sugg_pred_rows = conn.execute(
                        "SELECT * FROM predictions WHERE is_suggestion_for = ?",
                        (sugg_id,)
                    ).fetchall()
                    device_data['suggestions'].append({
                        "action": action_text,
                        "predictions": {row['horizon']: dict(row) for row in sugg_pred_rows}
                    })

            # history (ultima ora)
            if table_exists(conn, "history"):
                one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
                history_rows = conn.execute(
                    "SELECT * FROM history WHERE device_id = ? AND timestamp >= ? ORDER BY timestamp ASC",
                    (device_id, one_hour_ago)
                ).fetchall()
                device_data['history'] = [{
                    "time": datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00')).strftime('%H:%M'),
                    **dict(row)
                } for row in history_rows]

            # weather
            device_data['externalWeather'] = get_weather_for_device(device['owm_lat'], device['owm_lon'])

            # predictions_meta (timestamp generazione predizioni), compatibile se tabella esiste
            if table_exists(conn, "predictions_meta"):
                meta = conn.execute(
                    "SELECT generated_at FROM predictions_meta WHERE device_id = ?",
                    (device_id,)
                ).fetchone()
                if meta:
                    device_data['predictions_meta'] = {"generatedAt": meta['generated_at']}

            devices_payload.append(device_data)
            
    return {"devices": devices_payload, "all_actuators": ACTUATORS}

# --- API ENDPOINTS ---
@app.route('/api/data', methods=['GET'])
def api_get_data():
    return jsonify(get_full_data())

@app.route('/api/devices/add', methods=['POST'])
def api_add_device():
    data = request.json
    device_id = data.get('deviceId')
    location_name = data.get('locationName', 'Default')
    lat = data.get('lat', 0.0)
    lon = data.get('lon', 0.0)

    if not device_id: return jsonify({"error": "deviceId is required"}), 400
    with get_db_connection() as conn:
        try:
            conn.execute(
                "INSERT INTO devices (device_id, room_name, location_name, owm_lat, owm_lon) VALUES (?, NULL, ?, ?, ?)",
                (device_id, location_name, lat, lon)
            )
            for act in ACTUATORS:
                conn.execute(
                    "INSERT INTO device_actuators (device_id, actuator_name, is_enabled) VALUES (?, ?, ?)",
                    (device_id, act, 1)
                )
            conn.commit()
        except sqlite3.IntegrityError:
            return jsonify({"error": "Device ID already exists"}), 409
    return jsonify(get_full_data())

@app.route('/api/devices/update', methods=['POST'])
def api_update_device():
    data = request.json
    device_id = data.get('deviceId')
    if not device_id: return jsonify({"error": "deviceId is required"}), 400
    
    with get_db_connection() as conn:
        fields_to_update = {}
        if 'roomName' in data: fields_to_update['room_name'] = data['roomName']
        if 'locationName' in data: fields_to_update['location_name'] = data['locationName']
        if 'lat' in data: fields_to_update['owm_lat'] = data['lat']
        if 'lon' in data: fields_to_update['owm_lon'] = data['lon']
        
        if not fields_to_update: return jsonify({"error": "No valid fields to update"}), 400

        set_clause = ", ".join([f"{key} = ?" for key in fields_to_update.keys()])
        values = list(fields_to_update.values()) + [device_id]
        
        conn.execute(f"UPDATE devices SET {set_clause} WHERE device_id = ?", tuple(values))
        conn.commit()
        
    return jsonify(get_full_data())

@app.route('/api/disassociate', methods=['POST'])
def api_disassociate_device():
    device_id = request.json.get('deviceId')
    if not device_id: return jsonify({"error": "Missing deviceId"}), 400
    with get_db_connection() as conn:
        conn.execute("UPDATE devices SET room_name = NULL WHERE device_id = ?", (device_id,))
        conn.commit()
    return jsonify(get_full_data())

@app.route('/api/settings/actuator', methods=['POST'])
def api_toggle_actuator():
    data = request.json
    device_id, actuator_name, is_enabled = data.get('deviceId'), data.get('actuatorName'), data.get('isEnabled')
    if not all([device_id, actuator_name, is_enabled is not None]):
        return jsonify({"error": "Missing parameters"}), 400
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE device_actuators SET is_enabled = ? WHERE device_id = ? AND actuator_name = ?",
            (1 if is_enabled else 0, device_id, actuator_name)
        )
        conn.commit()
    return jsonify(get_full_data())

# --- NEW: Hysteresis settings (global per attuatore EN) ---
@app.route('/api/settings/hysteresis', methods=['GET'])
def api_get_hysteresis():
    with get_db_connection() as conn:
        data = load_hysteresis_map_en(conn)
    return jsonify(data)

@app.route('/api/settings/hysteresis', methods=['POST'])
def api_set_hysteresis():
    """
    Accetta sia:
      { "actuatorName": "AC", "on": 0.7, "off": 0.3 }
      { "actuator_name": "AC", "th_on": 0.7, "th_off": 0.3 }
    ...oppure una LISTA di oggetti come sopra.
    Mappa automaticamente nomi IT -> EN.
    """
    from datetime import datetime, timezone

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON"}), 400

    # Normalizza a lista
    items = payload if isinstance(payload, list) else [payload]

    updates, errors = [], []
    try:
        with get_db_connection() as conn:
            # assicura tabella
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hysteresis_global (
                    actuator_name TEXT PRIMARY KEY,
                    th_on REAL NOT NULL,
                    th_off REAL NOT NULL,
                    updated_at TEXT
                )
            """)

            for item in items:
                if not isinstance(item, dict):
                    errors.append({"item": item, "error": "Item must be an object"})
                    continue

                # campi accettati (camelCase o snake_case)
                raw_name = item.get("actuator_name") or item.get("actuatorName") or item.get("name")
                th_on = item.get("th_on", item.get("on"))
                th_off = item.get("th_off", item.get("off"))

                if raw_name is None or th_on is None or th_off is None:
                    errors.append({"item": item, "error": "Missing actuator name or thresholds"})
                    continue

                # mappa IT -> EN se necessario
                act_en = raw_name
                try:
                    from config import ACTUATOR_MAP_IT_TO_EN
                    act_en = ACTUATOR_MAP_IT_TO_EN.get(raw_name, raw_name)
                except Exception:
                    pass  # usa raw_name

                # cast + validazione
                try:
                    th_on = float(th_on)
                    th_off = float(th_off)
                except (TypeError, ValueError):
                    errors.append({"item": item, "error": "Thresholds must be numbers"})
                    continue

                if not (0.0 <= th_off < th_on <= 1.0):
                    errors.append({"item": item, "error": "Invalid thresholds: need 0 <= off < on <= 1"})
                    continue

                # upsert
                conn.execute("""
                    INSERT INTO hysteresis_global (actuator_name, th_on, th_off, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(actuator_name) DO UPDATE SET
                        th_on = excluded.th_on,
                        th_off = excluded.th_off,
                        updated_at = excluded.updated_at
                """, (act_en, th_on, th_off, datetime.now(timezone.utc).isoformat()))
                updates.append({"actuator": act_en, "on": th_on, "off": th_off})

            conn.commit()

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    resp = {"updated": updates}
    if errors:
        resp["errors"] = errors
        # 207-ish: parzialmente ok
        return jsonify(resp), (200 if updates else 400)

    return jsonify(resp), 200


# --- SERVER START ---
if __name__ == '__main__':
    if not os.path.exists(DB_PATH):
        print("[FATAL ERROR] Database file not found!")
        print(f"Please run your DB init script first.")
        exit(1)
    
    print("[INFO] Starting Flask server...")
    app.run(host='0.0.0.0', port=5500, debug=True)
