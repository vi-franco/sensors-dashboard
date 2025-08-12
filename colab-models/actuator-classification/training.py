!pip install scikit-multilearn

# ==============================================================================
# SEZIONE 0 — SETUP E CONFIGURAZIONE
# ==============================================================================
print("--- [SEZIONE 0] Inizio Setup e Configurazione ---")
import ../common
import ../utils/feature-engineering
import ../utils/functions

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

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=True)
    BASE_PATH = Path("/content/drive/MyDrive/Tesi")
except (ImportError, Exception):
    print("Non in Google Colab o errore di mount, esco")
    raise SystemExit("Montaggio Google Drive fallito.")

DATASET_COMPLETO_PATH = BASE_PATH / "DatasetSegmentato"
TRAINING_PERIODS_FILE = BASE_PATH / "training_periods.csv"
TEST_PERIODS_FILE = BASE_PATH / "test_periods.csv"

ALL_ACTUATORS = ["Umidificatore", "Finestra", "Deumidificatore", "Riscaldamento", "Clima"]
STATE_COLS = [f"state_{act}" for act in ALL_ACTUATORS]
AVAILABILITY_COLS = [f"available_{act}" for act in ALL_ACTUATORS]

print("✅ [SEZIONE 0] Setup completato.")


# ==============================================================================
# SEZIONE 1 — FUNZIONI
# ==============================================================================

def log_actuator_stats(df: pd.DataFrame, name: str = "Dataset") -> None:
    if df.empty or not set(STATE_COLS).issubset(df.columns):
        print(f"\n--- Statistiche {name}: dataset vuoto/colonne mancanti ---")
        return
    print(f"\n--- Statistiche Attuatori per {name} ({len(df)} righe) ---")
    all_off = df[STATE_COLS].eq(0).all(axis=1).sum()
    any_on = df[STATE_COLS].eq(1).any(axis=1).sum()
    print(f"  · Tutti OFF: {all_off} ({all_off/len(df):.2%})")
    print(f"  · Almeno uno ON: {any_on} ({any_on/len(df):.2%})")
    for c in STATE_COLS:
        on = df[c].sum()
        print(f"  · {c.replace('state_','')}: {on} ({on/len(df):.2%})")

print("✅ [SEZIONE 1] Funzioni definite.")


# ==============================================================================
# SEZIONE 2 — CARICAMENTO DEL DATASET
# ==============================================================================
print("\n--- [SEZIONE 2] Caricamento Dati ---")

final_df = load_unified_dataset(DATASET_COMPLETO_PATH)

if final_df.empty:
    raise SystemExit("DataFrame vuoto. Interrompo.")

final_df["utc_datetime"] = pd.to_datetime(final_df["utc_datetime"], errors="coerce", utc=True)
final_df.dropna(subset=["utc_datetime", "device"], inplace=True)
final_df.sort_values(["device", "utc_datetime"], inplace=True)

for col in STATE_COLS:
    if col in final_df.columns:
        final_df[col] = final_df[col].fillna(0).astype(int)

print("✅ [SEZIONE 2] Dati caricati.")


# ==============================================================================
# SEZIONE 3 — FEATURE ENGINEERING (pulita e causale)
# ==============================================================================

print("\n--- [SEZIONE 3] Feature Engineering ---")

df = add_features_actuator_classification(final_df)

print(f"✅ [SEZIONE 3] Completata. Shape: {df.shape}")


# ==============================================================================
# SEZIONE 4 — DEFINIZIONE FEATURE E SPLIT TRAIN/TEST
# ==============================================================================
print("\n--- [SEZIONE 4] Definizione Feature e Split ---")

# Base osservabili istantanei
BASE_FEATURES = [
    "temperature_sensor","absolute_humidity_sensor","co2","voc",
    "temperature_external","absolute_humidity_external",
    "ground_level_pressure","wind_speed","clouds_percentage","rain_1h",
    "presence",
]

# Gradienti / VPD / interazioni
GRADIENT_FEATURES = [
    "temp_diff_in_out","ah_diff_in_out","dewpoint_diff_in_out",
    "vpd_in","vpd_out","vpd_diff","temp_diff_x_wind",
]

# Tempo locale (ciclico + luce)
TIME_FEATURES = [
    "hour_sin","hour_cos","dow_sin","dow_cos","doy_sin","doy_cos",
    "minutes_from_sunrise","minutes_to_sunset","is_daylight",
]

# Rolling schema coerente per le serie interne
def roll_feats(prefix):
    return [
        f"{prefix}_accel_1m",
        f"{prefix}_trend_5m", f"{prefix}_trend_30m",
        f"{prefix}_mean_5m",  f"{prefix}_mean_30m",
        f"{prefix}_std_5m",   f"{prefix}_std_30m",
    ]
ROLLING_FEATURES = sum([roll_feats(c) for c in INTERNAL_SERIES], [])

# Trend esterni
EXTERNAL_TREND_FEATURES = [
    "temperature_external_trend_5m","temperature_external_trend_30m",
    "absolute_humidity_external_trend_5m","absolute_humidity_external_trend_30m",
]

# Baseline delta
BASELINE_FEATURES = [f"{c}_baseline_delta" for c in BASELINE_COLS]

# TERMODINAMICA (dew points)
THERMO_FEATURES = ["dew_point_sensor","dew_point_external"]

# MODEL FEATURES (senza available_*)
MODEL_FEATURES = (
    BASE_FEATURES + GRADIENT_FEATURES + TIME_FEATURES +
    ROLLING_FEATURES + EXTERNAL_TREND_FEATURES +
    BASELINE_FEATURES + THERMO_FEATURES
)

# tieni solo quelle realmente presenti
features = [f for f in MODEL_FEATURES if f in df.columns]
targets  = STATE_COLS.copy()

print(f"Features: {len(features)} · Targets: {len(targets)}")

data_for_training = get_data_from_periods(df, TRAINING_PERIODS_FILE)
test_df           = get_data_from_periods(df, TEST_PERIODS_FILE)

print(f"Righe Training: {len(data_for_training)} · Test: {len(test_df)}")

log_actuator_stats(data_for_training, "Training Set")
log_actuator_stats(test_df, "Test Set")

print("✅ [SEZIONE 4] OK.")


# ==============================================================================
# SEZIONE 5 — ADDESTRAMENTO & VALUTAZIONE (K-Fold multilabel)
# ==============================================================================
print("\n--- [SEZIONE 5] Addestramento e Valutazione ---")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
)
from skmultilearn.model_selection import IterativeStratification
import numpy as np

if data_for_training.empty:
    raise SystemExit("❌ Dataset di training vuoto. Impossibile addestrare.")

# Shuffle e selezione X/y
df_train = data_for_training.sample(frac=1, random_state=42).reset_index(drop=True)
X_df = df_train[features]
y_df = df_train[targets].astype(int)
avail_df = df_train[AVAILABILITY_COLS].clip(0,1).astype(int) if set(AVAILABILITY_COLS).issubset(df_train.columns) else None

N_SPLITS = 5
kfold = IterativeStratification(n_splits=N_SPLITS, order=1)

def create_model(input_dim, output_dim):
    inp = Input(shape=(input_dim,))
    x = Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-5))(inp)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-5))(x)
    x = Dropout(0.1)(x)
    x = Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.1)(x)
    out = Dense(output_dim, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["binary_accuracy","precision","recall"]
    )
    return model

histories, all_y_val, all_y_pred_probs = [], [], []

for fold_no, (tr_idx, va_idx) in enumerate(kfold.split(X_df, y_df), 1):
    print(f"---------------- Fold {fold_no}/{N_SPLITS} ----------------")
    X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
    y_tr, y_va = y_df.iloc[tr_idx], y_df.iloc[va_idx]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    model = create_model(X_tr_s.shape[1], y_tr.shape[1])
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)
    history = model.fit(X_tr_s, y_tr, epochs=20, batch_size=64,
                        validation_data=(X_va_s, y_va), callbacks=[es], verbose=0)
    histories.append(history)

    preds = model.predict(X_va_s, verbose=0)

    # Maschera di disponibilità SOLO post-predizione (non come feature)
    if avail_df is not None:
        avail_va = avail_df.iloc[va_idx].values
        preds = preds * avail_va  # forza a 0 dove non disponibile

    all_y_val.append(y_va.values)
    all_y_pred_probs.append(preds)

# Aggrego validazione
y_val_all = np.concatenate(all_y_val, axis=0)
y_pred_probs_all = np.concatenate(all_y_pred_probs, axis=0)

# Curve apprendimento medie
def plot_mean_learning_curve(histories, metric):
    seqs_tr = [h.history.get(metric, []) for h in histories]
    seqs_va = [h.history.get(f"val_{metric}", []) for h in histories]
    min_len = min(map(len, seqs_tr + seqs_va)) if seqs_tr and seqs_va else 0
    if min_len == 0: return
    tr = np.array([s[:min_len] for s in seqs_tr]).mean(axis=0)
    va = np.array([s[:min_len] for s in seqs_va]).mean(axis=0)
    epochs = range(1, min_len+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, tr, "o-", label=f"Train {metric}")
    plt.plot(epochs, va, "o-", label=f"Val {metric}")
    plt.title(f"Curva Apprendimento: {metric}")
    plt.xlabel("Epoca"); plt.legend(); plt.grid(True); plt.show()

plot_mean_learning_curve(histories, "loss")
plot_mean_learning_curve(histories, "precision")

# ROC & PR
plt.figure(figsize=(9,7))
for i, act in enumerate(ALL_ACTUATORS):
    fpr, tpr, _ = roc_curve(y_val_all[:, i], y_pred_probs_all[:, i])
    plt.plot(fpr, tpr, label=f"{act} (AUC={auc(fpr, tpr):.3f})")
plt.plot([0,1],[0,1],"k--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Aggregata")
plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(9,7))
for i, act in enumerate(ALL_ACTUATORS):
    pr, rc, _ = precision_recall_curve(y_val_all[:, i], y_pred_probs_all[:, i])
    ap = average_precision_score(y_val_all[:, i], y_pred_probs_all[:, i])
    plt.plot(rc, pr, label=f"{act} (AP={ap:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Aggregata")
plt.legend(); plt.grid(True); plt.show()

# Soglie ottimali per F1
optimal_thresholds = {}
print("\n--- Soglie Ottimali (F1) ---")
for i, act in enumerate(ALL_ACTUATORS):
    pr, rc, th = precision_recall_curve(y_val_all[:, i], y_pred_probs_all[:, i])
    if len(th) == 0:
        optimal_thresholds[act] = 0.5
    else:
        f1 = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-8)
        optimal_thresholds[act] = th[np.argmax(f1)]
    print(f" · {act}: {optimal_thresholds[act]:.2f}")

# Pred binarie con soglie attuatore-specifiche
thr_vec = np.array([optimal_thresholds[a] for a in ALL_ACTUATORS])[None, :]
y_pred_all_opt = (y_pred_probs_all > thr_vec).astype(int)

print("\n--- Report di Classificazione (validazione aggregata) ---")
from sklearn.metrics import classification_report
for i, act in enumerate(ALL_ACTUATORS):
    print(f"\n[{act}]")
    print(classification_report(y_val_all[:, i], y_pred_all_opt[:, i], digits=4, zero_division=0))

emr = np.all(y_val_all == y_pred_all_opt, axis=1).mean()
print(f"\nExact Match Ratio (EMR): {emr:.4f}")

print("\n✅ [SEZIONE 5] Valutazione K-Fold completata.")

# ==============================================================================
# SEZIONE 6 — ADDESTRAMENTO FINALE & SALVATAGGIO
# ==============================================================================
print("\n--- [SEZIONE 6] Addestramento Finale ---")

scaler_final = StandardScaler().fit(X_df)
X_scaled_final = scaler_final.transform(X_df)

model_final = create_model(X_scaled_final.shape[1], y_df.shape[1])
avg_epochs = int(np.mean([len(h.history.get("loss", [])) for h in histories if h.history.get("loss")]))
avg_epochs = max(avg_epochs, 5)
print(f"Addestramento finale per {avg_epochs} epoche...")
model_final.fit(X_scaled_final, y_df, epochs=avg_epochs, batch_size=64, verbose=0)

SAVED_PATH = Path(BASE_PATH) / "modelli_salvati_adv"
SAVED_PATH.mkdir(parents=True, exist_ok=True)
model_final.save(SAVED_PATH / "model.keras")
joblib.dump(scaler_final, SAVED_PATH / "scaler.joblib")
with open(SAVED_PATH / "features.json", "w") as f:
    json.dump(features, f)

print(f"✅ Modello/scaler/feature salvati in: {SAVED_PATH}")
