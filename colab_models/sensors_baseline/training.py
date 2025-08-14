# ==============================================================================
# SEZIONE 0 — SETUP E CONFIGURAZIONE
# ==============================================================================
print("--- [SEZIONE 0] Inizio Setup e Configurazione ---")
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import sys
import os

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = (CURRENT_DIR / "../..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.feature_engineering import ensure_min_columns_baseline_prediction, add_features_baseline_prediction, add_targets_baseline_prediction, final_features_baseline_prediction
from colab_models.common import load_unified_dataset, get_data_from_periods

SAVE_DIR = Path(__file__).parent / "output"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Directory di output pronta: {SAVE_DIR}")

BASE_PATH = Path("/content/drive/MyDrive/Tesi")
DATASET_COMPLETO_PATH = BASE_PATH / "DatasetSegmentato"

print("✅ [SEZIONE 0] Setup completato.")

# ==============================================================================
# SEZIONE 1 — CARICAMENTO DEL DATASET
# ==============================================================================
print("\n--- [SEZIONE 1] Caricamento Dati ---")

final_df = load_unified_dataset(DATASET_COMPLETO_PATH)

if final_df.empty:
    raise SystemExit("DataFrame vuoto. Interrompo.")

final_df["utc_datetime"] = pd.to_datetime(final_df["utc_datetime"], errors="coerce", utc=True)
final_df.dropna(subset=["utc_datetime", "device"], inplace=True)
final_df.sort_values(["device", "period_id", "utc_datetime"], inplace=True)

print("✅ [SEZIONE 1] Dati caricati.")

# ==============================================================================
# SEZIONE 2 — FEATURE ENGINEERING (pulita e causale)
# ==============================================================================

print("\n--- [SEZIONE 2] Feature Engineering ---")

ensure_min_columns_baseline_prediction(final_df)
df = add_features_baseline_prediction(final_df)
df = add_targets_baseline_prediction(df)

print(f"✅ [SEZIONE 2] Completata. Shape: {df.shape}")

# ==============================================================================
# SEZIONE 3 — DEFINIZIONE FEATURE E SPLIT TRAIN/TEST (Automatico 80/20 per Periodo)
# ==============================================================================
print("\n--- [SEZIONE 4] Definizione Feature e Split Automatico per Periodo ---")
import numpy as np

features_for_model = final_features_actuator_classification()
targets = [
    f"{col}_pred_{h}m"
    for col in columns_to_predict
    for h in horizons
]
df.dropna(subset=targets, inplace=True)

all_period_ids = df['period_id'].unique()

rng = np.random.RandomState(42)
rng.shuffle(all_period_ids)

test_size_percentage = 0.20
split_point = int(len(all_period_ids) * test_size_percentage)

test_period_ids = all_period_ids[:split_point]
train_period_ids = all_period_ids[split_point:]

print(f"Split automatico: {len(train_period_ids)} periodi per il training, {len(test_period_ids)} per il test.")

data_for_training = df[df['period_id'].isin(train_period_ids)].copy()
test_df = df[df['period_id'].isin(test_period_ids)].copy()

print(f"Righe Training: {len(data_for_training)} · Righe Test: {len(test_df)}")

print("✅ [SEZIONE 4] OK.")

# ==============================================================================
logging.info("\n--- [SEZIONE 5] Inizio Addestramento Modello di Regressione ---")
if data_for_training.empty:
    raise ValueError("DataFrame 'data_for_training' vuoto. Impossibile addestrare.")

df_train_sorted = data_for_training.sort_values('utc_datetime').reset_index(drop=True)

X_df = df_train_sorted[features_for_model]
y_df = df_train_sorted[targets]

val_split_percentage = 0.2
split_point = int(len(X_df) * (1 - val_split_percentage))

X_train, X_val = X_df.iloc[:split_point], X_df.iloc[split_point:]
y_train, y_val = y_df.iloc[:split_point], y_df.iloc[split_point:]

logging.info(f"Split temporale: {len(X_train)} righe per training, {len(X_val)} per validazione.")

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

try:
    model.save(str(SAVED_MODEL_PATH / "prediction_model.keras"))
    joblib.dump(x_scaler, str(SAVED_MODEL_PATH / "prediction_x_scaler.joblib"))
    joblib.dump(y_scaler, str(SAVED_MODEL_PATH / "prediction_y_scaler.joblib"))
    with open(str(SAVED_MODEL_PATH / "prediction_features.json"), "w") as f: json.dump(features_for_model, f)
    with open(str(SAVED_MODEL_PATH / "prediction_targets.json"), "w") as f: json.dump(targets, f)
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
    X_test_df = test_df[features_for_model]
    X_test_scaled = x_scaler.transform(X_test_df)

    y_pred_scaled = model.predict(X_test_scaled, verbose=1)
    y_pred_delta = y_scaler.inverse_transform(y_pred_scaled)

    results_df = test_df[['device', 'utc_datetime'] + STATE_COLS].copy()
    for i, name in enumerate(targets):
        base_col = name.split('_pred_')[0]
        results_df[f"pred_absolute_{name}"] = test_df[base_col].values + y_pred_delta[:, i]

    logging.info("\n--- Analisi MAE e Percentili di Errore per tutte le variabili ---")
    for h in PREDICTION_HORIZONS:
        print(f"\nOrizzonte @ {h} min:")

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