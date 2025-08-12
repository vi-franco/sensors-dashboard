# ==============================================================================
# SEZIONE 0: SETUP E CONFIGURAZIONE (INVARIATA)
# ==============================================================================
print("--- [SEZIONE 0] Inizio Setup e Configurazione ---")

import json
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=True)
    BASE_PATH = Path("/content/drive/MyDrive/Tesi")
except (ImportError, Exception):
    BASE_PATH = Path("./dati")

DATASET_COMPLETO_PATH = BASE_PATH / 'DatasetSegmentato'
TRAINING_PERIODS_FILE = BASE_PATH / "training_periods.csv"
TEST_PERIODS_FILE = BASE_PATH / "test_periods.csv"
DEBUG_DATA_PATH = BASE_PATH / "debug_data"
SAVED_MODEL_PATH = BASE_PATH / "modelli_previsione"
DEBUG_DATA_PATH.mkdir(parents=True, exist_ok=True)
SAVED_MODEL_PATH.mkdir(parents=True, exist_ok=True)
logging.info(f"Percorsi impostati. Dati letti da: {DATASET_COMPLETO_PATH}")

ALL_ACTUATORS = ['Umidificatore', 'Finestra', 'Deumidificatore', 'Riscaldamento', 'Clima']
STATE_COLS = [f"state_{act}" for act in ALL_ACTUATORS]
PREDICTION_HORIZONS = [15, 30, 60]
SENSOR_TARGETS = ["temperature_sensor", "absolute_humidity_sensor", "co2", "voc"]
logging.info("✅ [SEZIONE 0] Setup completato.")


# ==============================================================================
# SEZIONE 1: FUNZIONI DI UTILITÀ (INVARIATA)
# ==============================================================================
logging.info("\n--- [SEZIONE 1] Definizione Funzioni di Utilità ---")
def load_unified_dataset(folder_path):
    all_files_data = []
    if not folder_path.is_dir():
        logging.error(f"La cartella {folder_path} non esiste.")
        return pd.DataFrame()
    for file_path in folder_path.glob('*.csv'):
        try:
            df = pd.read_csv(file_path, on_bad_lines="skip")
            all_files_data.append(df)
        except Exception as e:
            logging.warning(f"Errore lettura {file_path.name}: {e}")
    if not all_files_data:
        logging.error(f"Nessun file CSV trovato in {folder_path}.")
        return pd.DataFrame()
    return pd.concat(all_files_data, ignore_index=True)

def calculate_relative_humidity(temp_c, abs_hum_g_m3):
    if pd.isna(temp_c) or pd.isna(abs_hum_g_m3) or abs_hum_g_m3 < 0: return np.nan
    temp_k = temp_c + 273.15
    e_s = 611.2 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    if e_s == 0: return np.nan
    e = (abs_hum_g_m3 * 461.5 * temp_k) / 1000
    rh = (e / e_s) * 100
    return max(0, min(100, rh))
logging.info("✅ [SEZIONE 1] Funzioni di utilità definite.")


# ==============================================================================
# SEZIONE 2: CARICAMENTO DATI (MODIFICATA: parsing local time)
# ==============================================================================
logging.info("\n--- [SEZIONE 2] Inizio Caricamento Dati Pre-elaborati ---")
final_df = load_unified_dataset(DATASET_COMPLETO_PATH)
if not final_df.empty:
    # UTC
    final_df['utc_datetime'] = pd.to_datetime(final_df['utc_datetime'], errors='coerce', utc=True)

    # Local time (usa il primo disponibile tra queste colonne)
    local_time_col = None
    for cand in ['local_datetime', 'locale_time', 'local_time']:
        if cand in final_df.columns:
            local_time_col = cand
            final_df[cand] = pd.to_datetime(final_df[cand], errors='coerce')
            break
    if local_time_col is None:
        logging.warning("⛔ Nessuna colonna di tempo locale trovata (local_datetime/locale_time/local_time). "
                        "Le feature cicliche sull'ora useranno UTC come fallback.")
        local_time_col = 'utc_datetime'  # fallback

    final_df.dropna(subset=['utc_datetime', 'device'], inplace=True)
    final_df.sort_values(['device', 'utc_datetime'], inplace=True)

    for col in STATE_COLS:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0).astype(int)

    logging.info(f"Dataset unificato caricato. Shape: {final_df.shape}")
else:
    logging.error("Il DataFrame è vuoto. Lo script non può continuare.")

logging.info(f"✅ [SEZIONE 2] Pipeline dati completata.")


# ==============================================================================
# SEZIONE 3: SELEZIONE E FEATURE ENGINEERING (MODIFICATA: ora del giorno ciclica)
# ==============================================================================
logging.info("\n--- [SEZIONE 3] Inizio Feature Engineering ---")
df = final_df.copy()

# Delta interno-esterno
logging.info("Calcolo delle feature 'delta' (differenza interno/esterno)...")
if 'temperature_external' in df.columns and 'temperature_sensor' in df.columns:
    df['temperature_delta'] = df['temperature_sensor'] - df['temperature_external']
if 'absolute_humidity_external' in df.columns and 'absolute_humidity_sensor' in df.columns:
    df['humidity_delta'] = df['absolute_humidity_sensor'] - df['absolute_humidity_external']

base_features_for_eng = ["temperature_sensor", "absolute_humidity_sensor", "co2", "voc"]
df_selected = df.copy()

# Trend esterni
logging.info("Calcolo trend per dati esterni (temperatura e umidità)...")
for w in [5, 15]:
    if 'temperature_external' in df_selected.columns:
        df_selected[f'temperature_external_trend_{w}m'] = df_selected.groupby('device')['temperature_external'].diff(periods=w)
    if 'absolute_humidity_external' in df_selected.columns:
        df_selected[f'absolute_humidity_external_trend_{w}m'] = df_selected.groupby('device')['absolute_humidity_external'].diff(periods=w)

# Rolling/trend interni
ROLLING_WINDOWS = [5, 10, 15, 30]
DIFF_WINDOWS = [1, 5, 10, 15]
logging.info(f"Calcolo feature di rolling/trend per sensori interni...")
for feat in base_features_for_eng:
    if feat in df_selected.columns:
        for w in ROLLING_WINDOWS:
            df_selected[f"{feat}_mean_{w}m"] = (
                df_selected.groupby("device")[feat]
                .rolling(window=w, min_periods=2)
                .mean()
                .reset_index(level=0, drop=True)
            )
            df_selected[f"{feat}_std_{w}m"] = (
                df_selected.groupby("device")[feat]
                .rolling(window=w, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
            )
        for w in DIFF_WINDOWS:
            df_selected[f"{feat}_trend_{w}m"] = df_selected.groupby("device")[feat].diff(periods=w)
        df_selected[f"{feat}_accel_1m"] = df_selected.groupby("device")[f"{feat}_trend_1m"].diff()

# ===== NEW: ora del giorno ciclica da tempo locale =====
logging.info("Aggiunta feature ora del giorno ciclica (sin/cos) da tempo locale...")
_local_col = None
for cand in ['local_datetime', 'locale_time', 'local_time', 'utc_datetime']:  # include UTC come fallback
    if cand in df_selected.columns:
        _local_col = cand
        break

if _local_col is None:
    raise ValueError("Nessuna colonna temporale disponibile per calcolare l'ora ciclica.")

# Frazione di ora nel giorno [0,24)
_h = df_selected[_local_col].dt.hour.fillna(0)
_m = df_selected[_local_col].dt.minute.fillna(0)
_s = df_selected[_local_col].dt.second.fillna(0)
_hod = _h + _m/60.0 + _s/3600.0

df_selected['hod_sin'] = np.sin(2 * np.pi * (_hod / 24.0))
df_selected['hod_cos'] = np.cos(2 * np.pi * (_hod / 24.0))

# Ordinamento/riempimento
df_selected.sort_values(["device", "utc_datetime"], inplace=True)
cols_to_fill = [col for col in df_selected.columns if col not in df.columns]
df_selected[cols_to_fill] = df_selected.groupby('device')[cols_to_fill].transform(lambda x: x.bfill().ffill())
df_selected.dropna(inplace=True)

logging.info(f"✅ [SEZIONE 3] Feature engineering completato. Shape finale: {df_selected.shape}")


# ==============================================================================
# SEZIONE 4: PREPARAZIONE DATI PER LA PREVISIONE
# (MODIFICATA: usa tutti i dati tranne i periodi di test; esclude feature attuatori;
#               include ora del giorno ciclica)
# ==============================================================================
logging.info("\n--- [SEZIONE 4] Preparazione Dati per la Previsione ---")
df_for_prediction = df_selected.copy()

TARGET_COLS, EVAL_COLS = [], []
logging.info("Creazione dei target (delta) e delle colonne di valutazione (valori assoluti)...")
for col in SENSOR_TARGETS:
    if col in df_for_prediction.columns:
        for h in PREDICTION_HORIZONS:
            tgt = f"{col}_pred_{h}m"
            evl = f"{col}_eval_{h}m"
            TARGET_COLS.append(tgt)
            EVAL_COLS.append(evl)
            df_for_prediction[tgt] = df_for_prediction.groupby('device')[col].shift(-h) - df_for_prediction[col]
            df_for_prediction[evl] = df_for_prediction.groupby('device')[col].shift(-h)
    else:
        logging.warning(f"La colonna target '{col}' non è stata trovata in df_for_prediction e sarà saltata.")

# (Opzionale) umidità relativa per valutazione, se presente
for h in PREDICTION_HORIZONS:
    if 'humidity_sensor' in df_for_prediction.columns:
        col_eval_rh = f'humidity_sensor_eval_{h}m'
        EVAL_COLS.append(col_eval_rh)
        df_for_prediction[col_eval_rh] = df_for_prediction.groupby('device')['humidity_sensor'].shift(-h)

# Drop righe con target/eval NaN
df_for_prediction.dropna(subset=TARGET_COLS + EVAL_COLS, inplace=True)
logging.info(f"DataFrame per previsione pulito. Shape: {df_for_prediction.shape}")

# ------------------ COSTRUZIONE FEATURE ------------------
import re
engineered = []
for feat in base_features_for_eng:
    pattern = re.escape(feat) + r"_(mean|std|trend|accel)_.*"
    engineered += [c for c in df_for_prediction.columns if re.match(pattern, c)]

THERMO_FEATURES = ['dew_point_sensor', 'dew_point_external']
DELTA_FEATURES = ['temperature_delta', 'humidity_delta']
EXTERNAL_STATIC_FEATURES = [
    'rain_1h', 'ground_level_pressure', 'clouds_percentage', 'wind_speed',
    'temperature_external', 'absolute_humidity_external'
]
EXTERNAL_TREND_FEATURES = [
    'temperature_external_trend_5m', 'temperature_external_trend_15m',
    'absolute_humidity_external_trend_5m', 'absolute_humidity_external_trend_15m'
]
CYCLIC_TIME_FEATURES = ['hod_sin', 'hod_cos']  # già create in Sezione 3

# >>> IMPORTANTE: niente feature degli attuatori (né state_*, né available_*)
MODEL_FEATURES = (
    base_features_for_eng
    + engineered
    + THERMO_FEATURES
    + DELTA_FEATURES
    + EXTERNAL_STATIC_FEATURES
    + EXTERNAL_TREND_FEATURES
    + CYCLIC_TIME_FEATURES
)
MODEL_FEATURES = sorted(set(
    f for f in MODEL_FEATURES
    if f in df_for_prediction.columns and not (f.startswith('state_') or f.startswith('available_'))
))

logging.info(f"Pronte {len(MODEL_FEATURES)} feature (senza attuatori) e {len(TARGET_COLS)} target.")

# ------------------ SPLIT: TEST PERIODS & TRAIN = TUTTO IL RESTO ------------------
def get_data_from_periods(df, periods_file):
    periods = pd.read_csv(periods_file) if Path(periods_file).exists() else pd.DataFrame()
    if periods.empty:
        return pd.DataFrame()
    idx = []
    for _, r in periods.iterrows():
        start = pd.to_datetime(r["start_time"], utc=True)
        end = pd.to_datetime(r["end_time"], utc=True)
        mask = (df["device"] == r["device"]) & (df["utc_datetime"] >= start) & (df["utc_datetime"] <= end)
        idx.extend(df[mask].index)
    return df.loc[idx].copy()

# 1) Test: i periodi definiti in TEST_PERIODS_FILE
test_df = get_data_from_periods(df_for_prediction, TEST_PERIODS_FILE)

# 2) Training: TUTTO il resto (escludo esplicitamente gli indici del test)
if not test_df.empty:
    train_idx = df_for_prediction.index.difference(test_df.index)
    data_for_training = df_for_prediction.loc[train_idx].copy()
else:
    data_for_training = df_for_prediction.copy()
    logging.warning("TEST_PERIODS_FILE assente/vuoto: userò tutto il dataset per il training; la valutazione sarà saltata.")

# Rimuovo le colonne *_eval_* dal training per evitare leakage
eval_cols_to_drop = [c for c in data_for_training.columns if "_eval_" in c]
if eval_cols_to_drop:
    data_for_training.drop(columns=eval_cols_to_drop, inplace=True)

logging.info(f"Dati finali -> training: {len(data_for_training)} righe; test: {len(test_df)} righe.")
logging.info("✅ [SEZIONE 4] Pipeline di preparazione per la previsione completata.")


# ==============================================================================
# SEZIONE 5: ADDESTRAMENTO MODELLO DI REGRESSIONE
# ==============================================================================
logging.info("\n--- [SEZIONE 5] Inizio Addestramento Modello di Regressione ---")
if data_for_training.empty:
    raise ValueError("DataFrame 'data_for_training' vuoto. Impossibile addestrare.")

df_train = data_for_training.sample(frac=1, random_state=42).reset_index(drop=True)

X_df = df_train[MODEL_FEATURES]
y_df = df_train[TARGET_COLS]

# --- split train/val ---
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

# --- scaling: fit su TRAIN ---
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)

X_train_s = x_scaler.transform(X_train).astype("float32")
X_val_s   = x_scaler.transform(X_val).astype("float32")
y_train_s = y_scaler.transform(y_train).astype("float32")
y_val_s   = y_scaler.transform(y_val).astype("float32")

# --- modello ---
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.losses import Huber

def create_regression_model(input_dim, output_dim, width=128, depth=3, dropout=0.1, lr=3e-4):
    inp = layers.Input(shape=(input_dim,))
    x = inp
    for _ in range(depth):
        x = layers.Dense(width, activation="relu")(x)
        if dropout:
            x = layers.Dropout(dropout)(x)
    out = layers.Dense(output_dim, activation="linear")(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=Huber(delta=1.0), metrics=["mae"])
    return model

model = create_regression_model(input_dim=X_train_s.shape[1], output_dim=y_train_s.shape[1])

# --- training ---
history = model.fit(
    X_train_s, y_train_s,
    validation_data=(X_val_s, y_val_s),
    epochs=100,
    batch_size=2048,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=3, min_lr=1e-5),
    ],
    verbose=1
)

# --- salvataggi ---
try:
    model.save(str(SAVED_MODEL_PATH / "prediction_model.keras"))
    joblib.dump(x_scaler, str(SAVED_MODEL_PATH / "prediction_x_scaler.joblib"))
    joblib.dump(y_scaler, str(SAVED_MODEL_PATH / "prediction_y_scaler.joblib"))
    with open(str(SAVED_MODEL_PATH / "prediction_features.json"), "w") as f: json.dump(MODEL_FEATURES, f)
    with open(str(SAVED_MODEL_PATH / "prediction_targets.json"), "w") as f: json.dump(TARGET_COLS, f)
    logging.info(f"✅ Modello e artefatti salvati in: {SAVED_MODEL_PATH}")
except Exception as e:
    logging.error(f"❌ Errore durante il salvataggio dei file: {e}")

logging.info("✅ [SEZIONE 5] Addestramento completato.")



# ==============================================================================
# SEZIONE 6: ANALISI DETTAGLIATA E VALUTAZIONE
# ==============================================================================
logging.info("\n--- [SEZIONE 6] Inizio Analisi Dettagliata e Valutazione ---")
if 'test_df' not in locals() or test_df.empty:
    logging.warning("Test set non disponibile, salto la valutazione dettagliata.")
else:
    X_test_df = test_df[MODEL_FEATURES]
    X_test_scaled = x_scaler_final.transform(X_test_df)

    y_pred_scaled = model_final.predict(X_test_scaled, verbose=0)
    y_pred_delta = y_scaler_final.inverse_transform(y_pred_scaled)
    results_df = test_df[['device', 'utc_datetime'] + STATE_COLS].copy()
    for i, name in enumerate(TARGET_COLS):
        base_col = name.split('_pred_')[0]
        results_df[f"pred_absolute_{name}"] = test_df[base_col].values + y_pred_delta[:, i]

    logging.info("\n--- Analisi MAE e Percentili di Errore per tutte le variabili ---")
    for h in PREDICTION_HORIZONS:
        print(f"\nOrizzonte @ {h} min:")
        # --- MODIFICA: Aggiunto calcolo e stampa dei percentili ---
        for var in SENSOR_TARGETS:
            if f"{var}_eval_{h}m" in test_df.columns:
                true_abs = test_df[f"{var}_eval_{h}m"].values
                pred_abs = results_df[f"pred_absolute_{var}_pred_{h}m"].values

                # Calcola l'errore assoluto per ogni punto
                abs_error = np.abs(true_abs - pred_abs)

                # Calcola metriche
                mae = np.mean(abs_error)
                p50 = np.percentile(abs_error, 50) # Mediana
                p90 = np.percentile(abs_error, 90)
                p95 = np.percentile(abs_error, 95)

                unit = {'temperature_sensor':'°C', 'absolute_humidity_sensor':'g/m³', 'co2':'ppm', 'voc':'index'}[var]
                var_name = var.replace('_sensor','').replace('absolute_humidity','umid_abs')

                print(f"  - {var_name:<10}: MAE = {mae:.3f} {unit}")
                print(f"    └─ Percentili Errore (P50/P90/P95): {p50:.3f} / {p90:.3f} / {p95:.3f} {unit}")

        if f"humidity_sensor_eval_{h}m" in test_df.columns:
            true_rh = test_df[f"humidity_sensor_eval_{h}m"].values
            pred_temp = results_df[f"pred_absolute_temperature_sensor_pred_{h}m"].values
            pred_abs_hum = results_df[f"pred_absolute_absolute_humidity_sensor_pred_{h}m"].values
            pred_rh = np.vectorize(calculate_relative_humidity)(pred_temp, pred_abs_hum)
            valid_mask = ~np.isnan(true_rh) & ~np.isnan(pred_rh)

            if np.any(valid_mask):
                abs_error_rh = np.abs(true_rh[valid_mask] - pred_rh[valid_mask])
                mae_rh = np.mean(abs_error_rh)
                p50_rh = np.percentile(abs_error_rh, 50)
                p90_rh = np.percentile(abs_error_rh, 90)
                p95_rh = np.percentile(abs_error_rh, 95)

                print(f"  - {'Umidità Rel':<10}: MAE = {mae_rh:.3f} %")
                print(f"    └─ Percentili Errore (P50/P90/P95): {p50_rh:.3f} / {p90_rh:.3f} / {p95_rh:.3f} %")

    logging.info("\n--- Analisi MAE per Stato Attuatore (Orizzonte @ 15 min) ---")
    scenarios = {"Tutto Spento": results_df[STATE_COLS].sum(axis=1) == 0}
    for act in ALL_ACTUATORS:
        scenarios[f"{act} Acceso"] = results_df[f"state_{act}"] == 1
    print(f"{'Scenario':<20} | {'MAE Temp':<10} | {'MAE Umid Abs':<12} | {'MAE CO₂':<10} | {'MAE VOC':<10}")
    print("-" * 75)
    for name, mask in scenarios.items():
        subset_results = results_df[mask]
        if len(subset_results) > 5:
            subset_true = test_df.loc[subset_results.index]
            maes = {}
            has_all_vars = True
            for var in SENSOR_TARGETS:
                if f"{var}_eval_15m" not in subset_true.columns or f"pred_absolute_{var}_pred_15m" not in subset_results.columns:
                    has_all_vars = False
                    break
                maes[var] = mean_absolute_error(subset_true[f"{var}_eval_15m"], subset_results[f"pred_absolute_{var}_pred_15m"])
            if has_all_vars:
                print(f"{name:<20} | {maes['temperature_sensor']:<10.2f} | {maes['absolute_humidity_sensor']:<12.2f} | {maes['co2']:<10.2f} | {maes['voc']:<10.2f}")
            else:
                print(f"{name:<20} | {'N/A (dati mancanti)':-^53}")
        else:
            print(f"{name:<20} | {'N/A (campioni < 5)':-^53}")

    logging.info("Salvataggio casi estremi per analisi...")

    extreme_cases_all = []

    for h in PREDICTION_HORIZONS:
        for var in SENSOR_TARGETS:
            eval_col = f"{var}_eval_{h}m"
            pred_col = f"pred_absolute_{var}_pred_{h}m"
            if eval_col in test_df.columns and pred_col in results_df.columns:
                true_vals = test_df[eval_col].values
                pred_vals = results_df[pred_col].values
                abs_error = np.abs(true_vals - pred_vals)

                threshold = np.percentile(abs_error, 95)
                extreme_idx = np.where(abs_error >= threshold)[0]

                # Creo dataframe con contesto
                df_ext = results_df.iloc[extreme_idx].copy()
                df_ext["var"] = var
                df_ext["horizon_min"] = h
                df_ext["y_true"] = true_vals[extreme_idx]
                df_ext["y_pred"] = pred_vals[extreme_idx]
                df_ext["abs_error"] = abs_error[extreme_idx]

                extreme_cases_all.append(df_ext)

    if extreme_cases_all:
        df_extreme_all = pd.concat(extreme_cases_all, ignore_index=True)
        out_path = DEBUG_DATA_PATH / "casi_estremi.csv"
        df_extreme_all.to_csv(out_path, index=False)
        logging.info(f"✅ Salvati {len(df_extreme_all)} casi estremi in {out_path}")
    else:
        logging.warning("Nessun caso estremo trovato.")

    logging.info("\n✅ [SEZIONE 6] Analisi dettagliata e completa terminata.")