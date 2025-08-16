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

df = add_features_baseline_prediction(final_df)
df = add_targets_baseline_prediction(df)

print(f"✅ [SEZIONE 2] Completata. Shape: {df.shape}")

# ==============================================================================
# SEZIONE 3 — DEFINIZIONE FEATURE E SPLIT TRAIN/TEST (Automatico 80/20 per Periodo)
# ==============================================================================
print("\n--- [SEZIONE 3] Definizione Feature e Split Automatico per Periodo ---")

columns_to_predict = ["temperature_sensor", "absolute_humidity_sensor", "co2", "voc"]
horizons = [15, 30, 60]

FEATURES = final_features_baseline_prediction()
TARGETS = [f"{col}_pred_{h}m" for col in columns_to_predict for h in horizons]

df = df.replace([np.inf, -np.inf], np.nan).dropna()

print("Esempi di feature e target:")
print(df[FEATURES + TARGETS].head(5).to_string(index=False))

# Split per period_id (shuffle + 80/20)
all_period_ids = pd.Series(df["period_id"]).dropna().unique()
rng = np.random.RandomState(42)
rng.shuffle(all_period_ids)

test_size_percentage = 0.20
val_within_train_percentage = 0.20

n_total = len(all_period_ids)
n_test = int(n_total * test_size_percentage)

test_period_ids = all_period_ids[:n_test]
trainval_period_ids = all_period_ids[n_test:]

n_val = int(len(trainval_period_ids) * val_within_train_percentage)
val_period_ids = trainval_period_ids[:n_val]
train_period_ids = trainval_period_ids[n_val:]

print(f"Split per periodi -> train: {len(train_period_ids)}, val: {len(val_period_ids)}, test: {len(test_period_ids)}")

data_for_training = df[df["period_id"].isin(train_period_ids)].copy()
val_df            = df[df["period_id"].isin(val_period_ids)].copy()
test_df           = df[df["period_id"].isin(test_period_ids)].copy()

print(f"Righe Training: {len(data_for_training)} · Righe Validation: {len(val_df)} · Righe Test: {len(test_df)}")
print("✅ [SEZIONE 3] OK.")

# ==============================================================================
# SEZIONE 4 — ADDESTRAMENTO MODELLO DI REGRESSIONE (Pulita)
# ==============================================================================
print("\n--- [SEZIONE 4] Inizio Addestramento Modello di Regressione ---")

if data_for_training.empty:
    raise ValueError("DataFrame 'data_for_training' vuoto. Impossibile addestrare.")
if val_df.empty:
    raise ValueError("Validation set vuoto. Controlla lo split per periodi.")

df_train_sorted = data_for_training.sort_values("utc_datetime").reset_index(drop=True)
df_val_sorted   = val_df.sort_values("utc_datetime").reset_index(drop=True)

X_train = df_train_sorted[FEATURES].copy()
y_train = df_train_sorted[TARGETS].copy()
X_val   = df_val_sorted[FEATURES].copy()
y_val   = df_val_sorted[TARGETS].copy()

print(f"Train: {len(X_train)} righe · Val: {len(X_val)} righe")

# Scaling (fit solo su TRAIN)
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)

X_train_s = x_scaler.transform(X_train).astype("float32")
X_val_s   = x_scaler.transform(X_val).astype("float32")
y_train_s = y_scaler.transform(y_train).astype("float32")
y_val_s   = y_scaler.transform(y_val).astype("float32")

print("✅ Scaling completato (fit su train, transform su train/val).")

# 4.6) Modello
from tensorflow.keras import layers, models, optimizers, losses, callbacks

inp = layers.Input(shape=(X_train_s.shape[1],))
x = layers.Dense(256, activation="relu")(inp)
x = layers.Dropout(0.2)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)

out = layers.Dense(y_train_s.shape[1], activation="linear")(x)

model = models.Model(inputs=inp, outputs=out)
model.compile(
    optimizer=optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
    loss=losses.Huber(delta=1.0),
    metrics=["mae"]
)

# Callback
cb_early = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)
cb_rlr   = callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=5, min_lr=1e-6)

# Training
history = model.fit(
    X_train_s, y_train_s,
    validation_data=(X_val_s, y_val_s),
    epochs=100,
    batch_size=2048,
    callbacks=[cb_early, cb_rlr],
    verbose=1
)

# 4.8) Salvataggio artefatti
try:
    SAVED_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    model.save(str(SAVED_MODEL_PATH / "prediction_model.keras"))
    joblib.dump(x_scaler, str(SAVED_MODEL_PATH / "prediction_x_scaler.joblib"))
    joblib.dump(y_scaler, str(SAVED_MODEL_PATH / "prediction_y_scaler.joblib"))
    with open(SAVED_MODEL_PATH / "prediction_features.json", "w") as f:
        json.dump(FEATURES, f, ensure_ascii=False, indent=2)
    with open(SAVED_MODEL_PATH / "prediction_targets.json", "w") as f:
        json.dump(TARGETS, f, ensure_ascii=False, indent=2)
    print(f"✅ Modello e artefatti salvati in: {SAVED_MODEL_PATH}")
except Exception as e:
    print(f"❌ Errore durante il salvataggio degli artefatti: {e}")

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

import re, numpy as np
from sklearn.metrics import mean_absolute_error

# 5.1) Preparo test set pulito e allineato
X_test_df = test_df[FEATURES].replace([np.inf, -np.inf], np.nan)
y_test_df = test_df[TARGETS].replace([np.inf, -np.inf], np.nan)
mask_ok = (~X_test_df.isna().any(axis=1)) & (~y_test_df.isna().any(axis=1))
X_test_df = X_test_df.loc[mask_ok].copy()
y_test_df = y_test_df.loc[mask_ok].copy()

# 5.2) Predizione (delta standardizzati -> delta reali) rispettando l'ordine TARGETS
X_test_s = x_scaler.transform(X_test_df).astype("float32")
y_pred_s = model.predict(X_test_s, verbose=0)
y_pred_delta = y_scaler.inverse_transform(pd.DataFrame(y_pred_s, columns=TARGETS)[TARGETS])
y_true_delta = y_test_df[TARGETS].to_numpy(dtype=float)

print("Range co2 true_delta:", np.min(y_true_delta[:, TARGETS.index("co2_pred_15m")]),
                                np.max(y_true_delta[:, TARGETS.index("co2_pred_15m")]))

# 5.3) Ricostruzione valori assoluti e metriche per orizzonte
units = {"temperature_sensor":"°C","absolute_humidity_sensor":"g/m³","co2":"ppm","voc":"index"}
by_h = {}  # { "15": {var: (true_abs, pred_abs)} , ... }

for i, tname in enumerate(TARGETS):
    m = re.match(r'(.+)_pred_(\d+)m', tname)
    if not m:
        continue
    base_col, h = m.groups()
    true_abs = X_test_df[base_col].values + y_true_delta[:, i]
    pred_abs = X_test_df[base_col].values + y_pred_delta[:, i]
    by_h.setdefault(h, {})[base_col] = (true_abs, pred_abs)

for h in sorted(by_h.keys(), key=lambda x: int(x)):
    print(f"\nOrizzonte @ {h} min:")
    for base_col, (y_true_abs, y_pred_abs) in by_h[h].items():
        msk = np.isfinite(y_true_abs) & np.isfinite(y_pred_abs)
        if msk.sum() == 0:
            print(f"  - {base_col}: MAE = N/A (nessun valore valido)")
            continue
        ae = np.abs(y_true_abs[msk] - y_pred_abs[msk])
        mae = mean_absolute_error(y_true_abs[msk], y_pred_abs[msk])
        p50, p90, p95 = np.percentile(ae, [50, 90, 95])
        u = units.get(base_col, "")
        print(f"  - {base_col}: MAE = {mae:.3f} {u}  (validi={msk.sum()})")
        print(f"    └─ Percentili (P50/P90/P95): {p50:.3f} / {p90:.3f} / {p95:.3f} {u}")

print("✅ [SEZIONE 5] Analisi dettagliata completata.")