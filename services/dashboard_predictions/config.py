# config.py
# -*- coding: utf-8 -*-
from pathlib import Path

# ==============================================================================
# --- GLOBAL CONFIGURATION ---
# ==============================================================================
# Root del progetto (due livelli su dal file config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Percorsi base ---
BASE_PATH = PROJECT_ROOT  # se serve per altre cose

# --- Percorso database (se serve tenerlo qui) ---
DB_PATH = PROJECT_ROOT / "database.db"

# --- Model Directories ---
ACTUATOR_MODEL_DIR   = PROJECT_ROOT / "colab_models" / "actuator_classification" / "output"
PREDICTION_MODEL_DIR = PROJECT_ROOT / "colab_models" / "prediction_model" / "output"
ACTION_REGRESSOR_DIR = PROJECT_ROOT / "colab_models" / "action_regressor" / "output"

# Alias di compatibilità per altri moduli
MODEL_DIR_CLASSIFICATION = ACTUATOR_MODEL_DIR
MODEL_DIR_PREDICTION     = PREDICTION_MODEL_DIR

# --- Data sources ---
INFLUXDB_HOST = 'localhost'
INFLUXDB_PORT = 8086
INFLUXDB_DATABASE = 'sensori'
INFLUXDB_MEASUREMENT = 'dati_sensori'

# --- Weather API ---
OWM_API_KEY = "d71435a2c59c063aaddc1332c9f226be"
OWM_CACHE_DIR = BASE_PATH / "owm_cache"
WEATHER_HISTORY_DIR = BASE_PATH / "weather_history"
OWM_CACHE_MINUTES = 15
WEATHER_HISTORY_MINUTES = 90

# --- Inference Parameters ---
HYSTERESIS_THRESHOLDS = {
    'Umidificatore':     {'on': 0.50, 'off': 0.40},
    'Finestra':          {'on': 0.50, 'off': 0.40},
    'Deumidificatore':   {'on': 0.30, 'off': 0.25},
    'Riscaldamento':     {'on': 0.35, 'off': 0.30},
    'Clima':             {'on': 0.70, 'off': 0.60},
}

HISTORY_MINUTES = 1440
MIN_RECORDS_REQUIRED = 10
SUGGESTION_COMFORT_IMPROVEMENT_THRESHOLD = 5.0

MIN_CO2 = 400.0
MIN_VOC = 0.0