# init_db.py
import sqlite3
import os
import sys
from datetime import datetime

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = (CURRENT_DIR / "../..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from colab_models.common import get_actuator_names

# --- CONFIGURATION (come nel tuo originale) ---
DB_PATH = 'database.db'
ALL_ACTUATORS = get_actuator_names()

def initialize_database():
    """
    Inizializza il database:
    - (come nel tuo originale) DROPA e ricrea tutte le tabelle principali
    - AGGIUNGE:
        * colonne prob_<Act> in latest_status
        * tabella hysteresis_global (isteresi globale per attuatore EN)
        * tabella predictions_meta (timestamp generazione predizioni)
    ATTENZIONE: droppa le tabelle esistenti. Usalo solo per (re)inizializzare.
    """
    if os.path.exists(DB_PATH):
        print(f"[INFO] Using DB at: {DB_PATH}")
    else:
        print(f"[INFO] Creating new DB at: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()

        # --- DROPS (come nel tuo originale: pulizia completa) ---
        print("Dropping old tables if they exist...")
        cur.execute("DROP TABLE IF EXISTS history")
        cur.execute("DROP TABLE IF EXISTS predictions")
        cur.execute("DROP TABLE IF EXISTS suggestions")
        cur.execute("DROP TABLE IF EXISTS latest_status")
        cur.execute("DROP TABLE IF EXISTS device_actuators")
        cur.execute("DROP TABLE IF EXISTS devices")
        # Nuove tabelle aggiunte da noi:
        cur.execute("DROP TABLE IF EXISTS hysteresis_global")
        cur.execute("DROP TABLE IF EXISTS predictions_meta")

        # 1) DEVICES
        print("Creating 'devices' table...")
        cur.execute("""
            CREATE TABLE devices (
                device_id       TEXT PRIMARY KEY,
                room_name       TEXT, -- Nome stanza da mostrare in UI
                location_name   TEXT, -- Es. "Senise"
                owm_lat         REAL, -- Latitudine per meteo esterno
                owm_lon         REAL  -- Longitudine per meteo esterno
            )
        """)

        # 2) DEVICE_ACTUATORS (quali attuatori sono abilitati/visibili per device)
        print("Creating 'device_actuators' table...")
        cur.execute("""
            CREATE TABLE device_actuators (
                device_id       TEXT,
                actuator_name   TEXT,
                is_enabled      INTEGER, -- 1 attivo/visibile, 0 no
                PRIMARY KEY (device_id, actuator_name),
                FOREIGN KEY (device_id) REFERENCES devices(device_id) ON DELETE CASCADE
            )
        """)

        # 3) LATEST_STATUS (ultimo stato sensori + stati attuatori)
        print("Creating 'latest_status' table...")
        status_cols = [
            "device_id TEXT PRIMARY KEY",
            "timestamp TEXT",
            "temperature REAL",
            "humidity REAL",
            "co2 REAL",
            "voc REAL",
            "heatIndex REAL",
            "iaqIndex REAL",
            "globalComfort REAL",
            "status_level INTEGER",
            "status_message TEXT"
        ]
        # Stati attuatori EN come nel tuo originale
        status_cols.extend([f"state_{act} INTEGER" for act in ALL_ACTUATORS])
        # NOVITÀ: probabilità attuatori (una colonna per ciascun attuatore EN)
        status_cols.extend([f"prob_{act} REAL" for act in ALL_ACTUATORS])

        cur.execute(f"CREATE TABLE latest_status ({', '.join(status_cols)})")

        # 4) SUGGESTIONS (come nel tuo originale: combinazioni suggerite di stati)
        print("Creating 'suggestions' table...")
        sugg_cols = ["suggestion_id INTEGER PRIMARY KEY AUTOINCREMENT", "device_id TEXT"]
        sugg_cols.extend([f"state_{act} INTEGER" for act in ALL_ACTUATORS])
        cur.execute(f"CREATE TABLE suggestions ({', '.join(sugg_cols)})")

        # 5) PREDICTIONS (come nel tuo originale)
        print("Creating 'predictions' table...")
        cur.execute("""
            CREATE TABLE predictions (
                prediction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id           TEXT,
                horizon             TEXT,
                is_suggestion_for   INTEGER, -- FK a suggestions.suggestion_id (o NULL per baseline)
                temperature         REAL,
                humidity            REAL,
                co2                 REAL,
                voc                 REAL,
                heatIndex           REAL,
                iaqIndex            REAL,
                globalComfort       REAL
            )
        """)

        # 6) HISTORY (come nel tuo originale)
        print("Creating 'history' table...")
        cur.execute("""
            CREATE TABLE history (
                history_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id       TEXT,
                timestamp       TEXT,
                temperature     REAL,
                humidity        REAL,
                co2             REAL,
                voc             REAL,
                heatIndex       REAL,
                iaqIndex        REAL,
                globalComfort   REAL
            )
        """)

        # 7) NOVITÀ: HYSTERESIS GLOBALE per attuatore EN
        print("Creating 'hysteresis_global' table...")
        cur.execute("""
            CREATE TABLE hysteresis_global (
                actuator_name TEXT PRIMARY KEY,  -- 'AC', 'Window', ...
                th_on  REAL NOT NULL,
                th_off REAL NOT NULL,
                updated_at TEXT
            )
        """)

        # Seed di default (soglie ragionevoli). Modificale se vuoi.
        print("Seeding default hysteresis thresholds...")
        default_hys = {
            'Finestra':       {'on': 0.70, 'off': 0.30},
            'Umidificatore':   {'on': 0.70, 'off': 0.30},
            'Deumidificatore': {'on': 0.70, 'off': 0.30},
            'Riscaldamento':      {'on': 0.70, 'off': 0.30},
            'Clima':           {'on': 0.70, 'off': 0.30},
        }
        for act, th in default_hys.items():
            cur.execute("""
                INSERT INTO hysteresis_global (actuator_name, th_on, th_off, updated_at)
                VALUES (?, ?, ?, ?)
            """, (act, th['on'], th['off'], datetime.utcnow().isoformat()))

        # 8) NOVITÀ: PREDICTIONS META (timestamp "predicted at" per device)
        print("Creating 'predictions_meta' table...")
        cur.execute("""
            CREATE TABLE predictions_meta (
                device_id    TEXT PRIMARY KEY,
                generated_at TEXT
            )
        """)

        conn.commit()
        print("\n[SUCCESS] Database initialization complete.")

if __name__ == "__main__":
    initialize_database()
