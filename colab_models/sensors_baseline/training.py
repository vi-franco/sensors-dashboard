# ==============================================================================
# SEZIONE 0 — SETUP E CONFIGURAZIONE
# ==============================================================================
print("--- [SEZIONE 0] Inizio Setup e Configurazione ---")
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
import sys
import os

# Per salvare figure senza display (Colab/CLI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Utils locali / progetto ---
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = (CURRENT_DIR / "../..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.feature_engineering import (
    ensure_min_columns_baseline_prediction,
    add_features_baseline_prediction,
    add_targets_baseline_prediction,
    final_features_baseline_prediction,
)
from colab_models.common import load_unified_dataset, get_data_from_periods, get_actuator_names

# --- Cartelle output & modelli ---
SAVE_DIR = CURRENT_DIR / "output"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODEL_PATH = SAVE_DIR  # uso la stessa cartella output per gli artefatti
DEBUG_DATA_PATH = SAVE_DIR / "debug"
DEBUG_DATA_PATH.mkdir(parents=True, exist_ok=True)

print(f"Directory di output pronta: {SAVE_DIR}")

# --- Path dataset (immutati) ---
BASE_PATH = Path("/content/drive/MyDrive/Tesi")
DATASET_COMPLETO_PATH = BASE_PATH / "DatasetSegmentato"

ALL_ACTUATORS = get_actuator_names()
STATE_COLS = [f"state_{act}" for act in ALL_ACTUATORS]

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
print("\n--- [SEZIONE 3] Definizione Feature e Split Automatico per Periodo ---")

columns_to_predict = ["temperature_sensor", "absolute_humidity_sensor", "co2", "voc"]
horizons = [15, 30, 60]

features_for_model = final_features_baseline_prediction()
targets = [f"{col}_pred_{h}m" for col in columns_to_predict for h in horizons]

# Rimuovo righe con target mancanti
df.dropna(subset=targets, inplace=True)

# Split per period_id (shuffle + 80/20)
all_period_ids = df["period_id"].dropna().unique()
rng = np.random.RandomState(42)
rng.shuffle(all_period_ids)

test_size_percentage = 0.20
split_point = int(len(all_period_ids) * test_size_percentage)

test_period_ids = all_period_ids[:split_point]
train_period_ids = all_period_ids[split_point:]

print(f"Split automatico: {len(train_period_ids)} periodi per il training, {len(test_period_ids)} per il test.")

data_for_training = df[df["period_id"].isin(train_period_ids)].copy()
test_df = df[df["period_id"].isin(test_period_ids)].copy()

print(f"Righe Training: {len(data_for_training)} · Righe Test: {len(test_df)}")
print("✅ [SEZIONE 3] OK.")

# ==============================================================================
# SEZIONE 4 — ADDDESTRAMENTO MODELLO DI REGRESSIONE
# ==============================================================================
print("\n--- [SEZIONE 4] Inizio Addestramento Modello di Regressione ---")
if data_for_training.empty:
    raise ValueError("DataFrame 'data_for_training' vuoto. Impossibile addestrare.")

# Ordine temporale per split train/val causale
df_train_sorted = data_for_training.sort_values("utc_datetime").reset_index(drop=True)
X_df = df_train_sorted[features_for_model]
y_df = df_train_sorted[targets]

val_split_percentage = 0.2
split_point_tv = int(len(X_df) * (1 - val_split_percentage))

X_train, X_val = X_df.iloc[:split_point_tv], X_df.iloc[split_point_tv:]
y_train, y_val = y_df.iloc[:split_point_tv], y_df.iloc[split_point_tv:]

print(f"Split temporale: {len(X_train)} righe training, {len(X_val)} validazione.")

# Scaling solo su TRAIN
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)

X_train_s = x_scaler.transform(X_train).astype("float32")
X_val_s   = x_scaler.transform(X_val).astype("float32")
y_train_s = y_scaler.transform(y_train).astype("float32")
y_val_s   = y_scaler.transform(y_val).astype("float32")

def create_regression_model(input_dim, output_dim, width=128, depth=3, dropout=0.1, lr=3e-4):
    inp = layers.Input(shape=(input_dim,))
    x = inp
    for _ in range(depth):
        x = layers.Dense(width, activation="relu")(x)
        if dropout:
            x = layers.Dropout(dropout)(x)
    out = layers.Dense(output_dim, activation="linear")(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=Huber(delta=1.0),
        metrics=["mae"],
    )
    return model

model = create_regression_model(input_dim=X_train_s.shape[1], output_dim=y_train_s.shape[1])

history = model.fit(
    X_train_s, y_train_s,
    validation_data=(X_val_s, y_val_s),
    epochs=100,
    batch_size=2048,
    callbacks=[
        EarlyStopping(monitor="val_mae", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=3, min_lr=1e-5),
    ],
    verbose=1
)

# --- Salvataggio modello e artefatti ---
try:
    model.save(str(SAVED_MODEL_PATH / "prediction_model.keras"))
    joblib.dump(x_scaler, str(SAVED_MODEL_PATH / "prediction_x_scaler.joblib"))
    joblib.dump(y_scaler, str(SAVED_MODEL_PATH / "prediction_y_scaler.joblib"))
    with open(SAVED_MODEL_PATH / "prediction_features.json", "w") as f:
        json.dump(features_for_model, f, ensure_ascii=False, indent=2)
    with open(SAVED_MODEL_PATH / "prediction_targets.json", "w") as f:
        json.dump(targets, f, ensure_ascii=False, indent=2)
    print(f"✅ Modello e artefatti salvati in: {SAVED_MODEL_PATH}")
except Exception as e:
    print(f"❌ Errore durante il salvataggio dei file: {e}")

print("✅ [SEZIONE 4] Addestramento completato.")

# --- Salvo figure training (loss/MAE) ---
try:
    # Loss
    plt.figure()
    plt.plot(history.history.get("loss", []), label="loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    loss_path = SAVE_DIR / "training_loss.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()
    # MAE
    plt.figure()
    plt.plot(history.history.get("mae", []), label="mae")
    plt.plot(history.history.get("val_mae", []), label="val_mae")
    plt.title("Training MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    mae_path = SAVE_DIR / "training_mae.png"
    plt.savefig(mae_path, dpi=150)
    plt.close()
    print(f"📈 Figure salvate: {loss_path.name}, {mae_path.name}")
except Exception as e:
    print(f"⚠️ Impossibile salvare le figure di training: {e}")

# ==============================================================================
# SEZIONE 5 — ANALISI DETTAGLIATA E VALUTAZIONE
# ==============================================================================

print("\n--- [SEZIONE 5] Inizio Analisi Dettagliata e Valutazione ---")

if test_df.empty:
    print("⚠️ Test set non disponibile, salto la valutazione dettagliata.")
else:
    # Utility per RH da T (°C) e AH (g/m^3)
    def calculate_relative_humidity(temp_c, abs_hum_gm3):
        if pd.isna(temp_c) or pd.isna(abs_hum_gm3):
            return np.nan
        T = float(temp_c)
        AH = float(abs_hum_gm3)
        # Magnus (T in °C, es in hPa)
        es = 6.112 * np.exp((17.62 * T) / (243.12 + T))
        # relazione AH = 216.7 * (e / (T+273.15))  => e = AH * (T+273.15) / 216.7
        e = AH * (T + 273.15) / 216.7
        rh = 100.0 * e / es
        return float(np.clip(rh, 0.0, 100.0))

    # Predizione sul test
    X_test_df = test_df[features_for_model]
    X_test_scaled = x_scaler.transform(X_test_df)
    y_pred_scaled = model.predict(X_test_scaled, verbose=1)
    y_pred_delta = y_scaler.inverse_transform(y_pred_scaled)

    # Ricostruisco predizioni assolute
    results_df = test_df[["device", "utc_datetime"]].copy()
    for i, name in enumerate(targets):
        base_col = name.split("_pred_")[0]
        if base_col in test_df.columns:
            results_df[f"pred_absolute_{name}"] = test_df[base_col].values + y_pred_delta[:, i]

    # MAE e percentili di errore per ogni orizzonte/variabile
    PREDICTION_HORIZONS = horizons
    SENSOR_TARGETS = columns_to_predict

    mae_rows = []
    for h in PREDICTION_HORIZONS:
        print(f"\nOrizzonte @ {h} min:")
        for var in SENSOR_TARGETS:
            eval_col = f"{var}_eval_{h}m"
            pred_col = f"pred_absolute_{var}_pred_{h}m"
            if eval_col in test_df.columns and pred_col in results_df.columns:
                true_abs = test_df[eval_col].values
                pred_abs = results_df[pred_col].values
                abs_error = np.abs(true_abs - pred_abs)
                mae = float(np.mean(abs_error))
                p50 = float(np.percentile(abs_error, 50))
                p90 = float(np.percentile(abs_error, 90))
                p95 = float(np.percentile(abs_error, 95))

                unit_map = {
                    "temperature_sensor": "°C",
                    "absolute_humidity_sensor": "g/m³",
                    "co2": "ppm",
                    "voc": "index",
                }
                unit = unit_map.get(var, "")
                var_name = var.replace("_sensor", "").replace("absolute_humidity", "umid_abs")

                print(f"  - {var_name:<12}: MAE = {mae:.3f} {unit}")
                print(f"    └─ Percentili (P50/P90/P95): {p50:.3f} / {p90:.3f} / {p95:.3f} {unit}")

                mae_rows.append({"h_min": h, "var": var_name, "mae": mae})

        # Umidità relativa derivata (se disponibile)
        rh_eval_col = f"humidity_sensor_eval_{h}m"
        t_pred_col = f"pred_absolute_temperature_sensor_pred_{h}m"
        ah_pred_col = f"pred_absolute_absolute_humidity_sensor_pred_{h}m"
        if rh_eval_col in test_df.columns and t_pred_col in results_df.columns and ah_pred_col in results_df.columns:
            true_rh = test_df[rh_eval_col].values
            pred_temp = results_df[t_pred_col].values
            pred_abs_hum = results_df[ah_pred_col].values
            pred_rh = np.vectorize(calculate_relative_humidity)(pred_temp, pred_abs_hum)
            valid_mask = ~np.isnan(true_rh) & ~np.isnan(pred_rh)
            if np.any(valid_mask):
                abs_error_rh = np.abs(true_rh[valid_mask] - pred_rh[valid_mask])
                mae_rh = float(np.mean(abs_error_rh))
                p50_rh = float(np.percentile(abs_error_rh, 50))
                p90_rh = float(np.percentile(abs_error_rh, 90))
                p95_rh = float(np.percentile(abs_error_rh, 95))
                print(f"  - {'Umidità Rel':<12}: MAE = {mae_rh:.3f} %")
                print(f"    └─ Percentili (P50/P90/P95): {p50_rh:.3f} / {p90_rh:.3f} / {p95_rh:.3f} %")
                mae_rows.append({"h_min": h, "var": "RH", "mae": mae_rh})

    # Salvo figura con MAE per orizzonte/variabile (se ho righe)
    if mae_rows:
        mae_df_plot = pd.DataFrame(mae_rows)
        for h in sorted(mae_df_plot["h_min"].unique()):
            sub = mae_df_plot[mae_df_plot["h_min"] == h]
            plt.figure()
            plt.bar(sub["var"], sub["mae"])
            plt.title(f"MAE per variabile @ {h} min")
            plt.xlabel("Variabile")
            plt.ylabel("MAE")
            plt.tight_layout()
            outp = SAVE_DIR / f"mae_{h}m.png"
            plt.savefig(outp, dpi=150)
            plt.close()
        print("📊 Figure MAE salvate nella cartella output.")

    # Analisi per scenari di stato attuatori (se presenti colonne state_)
    print("\n--- Analisi MAE per Stato Attuatore (@15m) ---")
    STATE_COLS = [c for c in test_df.columns if c.startswith("state_")]
    if STATE_COLS:
        # Ricavo la lista attuatori dal prefisso
        ALL_ACTUATORS = [c.replace("state_", "") for c in STATE_COLS]
        scenarios = {"Tutto Spento": (test_df[STATE_COLS].sum(axis=1) == 0)}
        for act in ALL_ACTUATORS:
            col = f"state_{act}"
            if col in test_df.columns:
                scenarios[f"{act} Acceso"] = test_df[col] == 1

        header = f"{'Scenario':<20} | {'MAE Temp':<10} | {'MAE Umid Abs':<12} | {'MAE CO₂':<10} | {'MAE VOC':<10}"
        print(header)
        print("-" * len(header))
        for name, mask in scenarios.items():
            subset_idx = mask[mask].index
            subset_true = test_df.loc[subset_idx]
            subset_pred = results_df.loc[subset_idx]
            if len(subset_true) > 5:
                has_all = all(
                    f"{v}_eval_15m" in subset_true.columns and f"pred_absolute_{v}_pred_15m" in subset_pred.columns
                    for v in SENSOR_TARGETS
                )
                if has_all:
                    maes = {
                        v: mean_absolute_error(subset_true[f"{v}_eval_15m"], subset_pred[f"pred_absolute_{v}_pred_15m"])
                        for v in SENSOR_TARGETS
                    }
                    print(
                        f"{name:<20} | "
                        f"{maes['temperature_sensor']:<10.2f} | "
                        f"{maes['absolute_humidity_sensor']:<12.2f} | "
                        f"{maes['co2']:<10.2f} | "
                        f"{maes['voc']:<10.2f}"
                    )
                else:
                    print(f"{name:<20} | {'N/A (dati mancanti)':-^53}")
            else:
                print(f"{name:<20} | {'N/A (campioni < 5)':-^53}")
    else:
        print("ℹ️ Nessuna colonna di stato attuatori (state_*) trovata: salto l’analisi per scenari.")

    # Salvataggio casi estremi (95° percentile errore) per ciascuna variabile/orizzonte
    print("\nSalvataggio casi estremi per analisi…")
    extreme_cases_all = []
    for h in PREDICTION_HORIZONS:
        for var in SENSOR_TARGETS:
            eval_col = f"{var}_eval_{h}m"
            pred_col = f"pred_absolute_{var}_pred_{h}m"
            if eval_col in test_df.columns and pred_col in results_df.columns:
                true_vals = test_df[eval_col].values
                pred_vals = results_df[pred_col].values
                abs_error = np.abs(true_vals - pred_vals)
                if len(abs_error) == 0:
                    continue
                threshold = np.percentile(abs_error, 95)
                extreme_idx = np.where(abs_error >= threshold)[0]
                if len(extreme_idx) == 0:
                    continue
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
        print(f"✅ Salvati {len(df_extreme_all)} casi estremi in {out_path}")
    else:
        print("ℹ️ Nessun caso estremo trovato.")

    print("\n✅ [SEZIONE 5] Analisi dettagliata e completa terminata.")

print("\n🏁 Fine script.")
