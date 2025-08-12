# config.py
# -*- coding: utf-8 -*-
from pathlib import Path

# ==============================================================================
# --- GLOBAL CONFIGURATION ---
# ==============================================================================
BASE_PATH = Path(__file__).resolve().parent
DB_PATH = BASE_PATH / "database.db"

# --- Model Directories ---
ACTUATOR_MODEL_DIR = BASE_PATH / "modelli_salvati_adv"
PREDICTION_MODEL_DIR = BASE_PATH / "modelli_previsione"
ACTION_REGRESSOR_DIR = PREDICTION_MODEL_DIR
# Alias di compatibilità per altri moduli
MODEL_DIR_CLASSIFICATION = ACTUATOR_MODEL_DIR
MODEL_DIR_PREDICTION     = PREDICTION_MODEL_DIR
ACTION_REGRESSOR_DIR = PREDICTION_MODEL_DIR

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

# --- Actuator Configuration ---
ALL_ACTUATORS_IT = ['Umidificatore', 'Finestra', 'Deumidificatore', 'Riscaldamento', 'Clima']
ALL_ACTUATORS_EN = ['Humidifier', 'Window', 'Dehumidifier', 'Heating', 'AC']
ACTUATOR_MAP_IT_TO_EN = dict(zip(ALL_ACTUATORS_IT, ALL_ACTUATORS_EN))
ACTUATOR_MAP_EN_TO_IT = dict(zip(ALL_ACTUATORS_EN, ALL_ACTUATORS_IT))

# --- Inference Parameters ---
HYSTERESIS_THRESHOLDS = {
    'Umidificatore':     {'on': 0.50, 'off': 0.40},
    'Finestra':          {'on': 0.50, 'off': 0.40},
    'Deumidificatore':   {'on': 0.30, 'off': 0.25},
    'Riscaldamento':     {'on': 0.35, 'off': 0.30},
    'Clima':             {'on': 0.70, 'off': 0.60},
}
HISTORY_MINUTES = 90
MIN_RECORDS_REQUIRED = 10
SUGGESTION_COMFORT_IMPROVEMENT_THRESHOLD = 5.0

# Limiti fisici
MIN_CO2 = 400.0
MIN_VOC = 0.0

# ==============================================================================
# --- FINAL FEATURE LISTS ---
# ==============================================================================

# --- FEATURES FOR CLASSIFICATION MODEL ---
CLASSIFICATION_FEATURES = [
    # --- Base istantanee
    "temperature_sensor", "absolute_humidity_sensor", "co2", "voc",
    "temperature_external", "absolute_humidity_external",
    "ground_level_pressure", "wind_speed", "clouds_percentage", "rain_1h",
    "presence",

    # --- Gradienti / VPD / Interazioni
    "temp_diff_in_out", "ah_diff_in_out", "dewpoint_diff_in_out",
    "vpd_in", "vpd_out", "vpd_diff",
    "temp_diff_x_wind",

    # --- Tempo locale (ciclico + luce)
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "minutes_from_sunrise", "minutes_to_sunset", "is_daylight",

    # --- Rolling schema 5m/30m (interni)
    # temperature_sensor
    "temperature_sensor_accel_1m",
    "temperature_sensor_trend_5m", "temperature_sensor_trend_30m",
    "temperature_sensor_mean_5m",  "temperature_sensor_mean_30m",
    "temperature_sensor_std_5m",   "temperature_sensor_std_30m",
    # absolute_humidity_sensor
    "absolute_humidity_sensor_accel_1m",
    "absolute_humidity_sensor_trend_5m", "absolute_humidity_sensor_trend_30m",
    "absolute_humidity_sensor_mean_5m",  "absolute_humidity_sensor_mean_30m",
    "absolute_humidity_sensor_std_5m",   "absolute_humidity_sensor_std_30m",
    # co2
    "co2_accel_1m",
    "co2_trend_5m", "co2_trend_30m",
    "co2_mean_5m",  "co2_mean_30m",
    "co2_std_5m",   "co2_std_30m",
    # voc
    "voc_accel_1m",
    "voc_trend_5m", "voc_trend_30m",
    "voc_mean_5m",  "voc_mean_30m",
    "voc_std_5m",   "voc_std_30m",

    # --- Trend esterni 5m/30m
    "temperature_external_trend_5m", "temperature_external_trend_30m",
    "absolute_humidity_external_trend_5m", "absolute_humidity_external_trend_30m",

    # --- Baseline delta (mediana ~24h per device)
    "co2_baseline_delta", "voc_baseline_delta",
    "temperature_sensor_baseline_delta", "absolute_humidity_sensor_baseline_delta",

    # --- Termodinamica
    "dew_point_sensor", "dew_point_external",
]

# --- FEATURES FOR PREDICTION MODEL ---
PREDICTION_FEATURES = [
    'absolute_humidity_external',
    'absolute_humidity_external_trend_15m',
    'absolute_humidity_external_trend_5m',
    'absolute_humidity_sensor',
    'absolute_humidity_sensor_accel_1m',
    'absolute_humidity_sensor_mean_10m',
    'absolute_humidity_sensor_mean_15m',
    'absolute_humidity_sensor_mean_30m',
    'absolute_humidity_sensor_mean_5m',
    'absolute_humidity_sensor_std_10m',
    'absolute_humidity_sensor_std_15m',
    'absolute_humidity_sensor_std_30m',
    'absolute_humidity_sensor_std_5m',
    'absolute_humidity_sensor_trend_10m',
    'absolute_humidity_sensor_trend_15m',
    'absolute_humidity_sensor_trend_1m',
    'absolute_humidity_sensor_trend_5m',
    'clouds_percentage',
    'co2',
    'co2_accel_1m',
    'co2_mean_10m',
    'co2_mean_15m',
    'co2_mean_30m',
    'co2_mean_5m',
    'co2_std_10m',
    'co2_std_15m',
    'co2_std_30m',
    'co2_std_5m',
    'co2_trend_10m',
    'co2_trend_15m',
    'co2_trend_1m',
    'co2_trend_5m',
    'dew_point_external',
    'dew_point_sensor',
    'ground_level_pressure',
    'hod_cos',
    'hod_sin',
    'humidity_delta',
    'rain_1h',
    'temperature_delta',
    'temperature_external',
    'temperature_external_trend_15m',
    'temperature_external_trend_5m',
    'temperature_sensor',
    'temperature_sensor_accel_1m',
    'temperature_sensor_mean_10m',
    'temperature_sensor_mean_15m',
    'temperature_sensor_mean_30m',
    'temperature_sensor_mean_5m',
    'temperature_sensor_std_10m',
    'temperature_sensor_std_15m',
    'temperature_sensor_std_30m',
    'temperature_sensor_std_5m',
    'temperature_sensor_trend_10m',
    'temperature_sensor_trend_15m',
    'temperature_sensor_trend_1m',
    'temperature_sensor_trend_5m',
    'voc',
    'voc_accel_1m',
    'voc_mean_10m',
    'voc_mean_15m',
    'voc_mean_30m',
    'voc_mean_5m',
    'voc_std_10m',
    'voc_std_15m',
    'voc_std_30m',
    'voc_std_5m',
    'voc_trend_10m',
    'voc_trend_15m',
    'voc_trend_1m',
    'voc_trend_5m',
    'wind_speed'
]
