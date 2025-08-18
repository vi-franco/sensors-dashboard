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

from utils.feature_engineering import ensure_min_columns_actuator_classification, add_features_actuator_classification, final_features_actuator_classification
from colab_models.common import load_unified_dataset, get_data_from_periods, get_actuator_names, log_actuator_stats, duplicate_groups_with_noise

SAVE_DIR = Path(__file__).parent / "output"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Directory di output pronta: {SAVE_DIR}")

BASE_PATH = Path("/content/drive/MyDrive/Tesi")
DATASET_COMPLETO_PATH = BASE_PATH / "DatasetSegmentato"
TRAINING_PERIODS_FILE = BASE_PATH / "training_periods.csv"
TEST_PERIODS_FILE = BASE_PATH / "test_periods.csv"

ALL_ACTUATORS = get_actuator_names()
STATE_COLS = [f"state_{act}" for act in ALL_ACTUATORS]

print("✅ [SEZIONE 0] Setup completato.")


# ==============================================================================
# SEZIONE 2 — CARICAMENTO DEL DATASET
# ==============================================================================
print("\n--- [SEZIONE 2] Caricamento Dati ---")

final_df = load_unified_dataset(DATASET_COMPLETO_PATH)
if final_df.empty:
    raise SystemExit("DataFrame vuoto. Interrompo.")

print(final_df.head())

data_for_training = get_data_from_periods(final_df, TRAINING_PERIODS_FILE)
data_for_test = get_data_from_periods(final_df, TEST_PERIODS_FILE)

aug_df = duplicate_groups_with_noise(data_for_training, n_duplicates=3)
if not aug_df.empty:
    print(f"[AUG] Aggiunte {len(aug_df)} righe (solo training).")
    data_for_training = pd.concat([data_for_training, aug_df], ignore_index=True)

print("✅ [SEZIONE 2] Dati caricati.")

# ==============================================================================
# SEZIONE 3 — FEATURE ENGINEERING (pulita e causale)
# ==============================================================================

print("\n--- [SEZIONE 3] Feature Engineering ---")

data_for_training = add_features_actuator_classification(data_for_training)
print(data_for_training.head())

print(f"✅ [SEZIONE 3] Completata. Shape: {data_for_training.shape}")


# ==============================================================================
# SEZIONE 4 — DEFINIZIONE FEATURE E SPLIT TRAIN/TEST
# ==============================================================================
print("\n--- [SEZIONE 4] Definizione Feature e Split ---")

features = final_features_actuator_classification()
targets  = STATE_COLS.copy()

print(f"Features: {len(features)} · Targets: {len(targets)}")

print(f"Righe Training: {len(data_for_training)} · Test: {len(data_for_test)}")

log_actuator_stats(data_for_training, STATE_COLS, "Training Set")
log_actuator_stats(data_for_test, STATE_COLS, "Test Set")

print("✅ [SEZIONE 4] OK.")

# ==============================================================================
# SEZIONE 5 — ADDESTRAMENTO & VALUTAZIONE (single split, no AUG in val)
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import keras

def create_model(input_dim, output_dim):
    x_in = Input(shape=(input_dim,))
    x = Dense(64, kernel_regularizer=keras.regularizers.l2(1e-4))(x_in)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(32, kernel_regularizer=keras.regularizers.l2(1e-4))(x_in)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = Dropout(0.3)(x)
    y_out = Dense(output_dim, activation="sigmoid")(x)
    m = Model(inputs=x_in, outputs=y_out)
    m.compile(optimizer=tf.keras.optimizers.Adam(2e-4),
              loss="binary_crossentropy",
              metrics=["binary_accuracy", "precision", "recall"])
    return m

if data_for_training.empty:
    raise SystemExit("❌ Dataset di training vuoto. Impossibile addestrare.")

data_for_training = data_for_training.sort_values("utc_datetime").reset_index(drop=True)
X_df = data_for_training[features]
y_df = data_for_training[targets].astype(int).values

# base periods in ordine temporale (collassando AUG nella base)
grp_start = data_for_training.groupby("period_id")["utc_datetime"].min().sort_values()
groups = grp_start.index.to_series()
base_ids = groups.str.split("__").str[0]
order_df = (
    grp_start.reset_index()
    .assign(base=base_ids.values)
    .groupby("base")["utc_datetime"].min()
    .sort_values()
    .reset_index()
)
base_order = order_df["base"].to_numpy()

# single split: ultimi base_periods per validation
val_ratio = 0.2
n_val = max(1, int(len(base_order) * val_ratio))
val_bases = base_order[-n_val:]
train_bases = base_order[:-n_val]

# escludo gruppi AUG dalla validation
is_original = ~groups.str.contains("__")
train_groups = groups[base_ids.isin(train_bases)].values                  # originali + AUG
val_groups   = groups[is_original & base_ids.isin(val_bases)].values      # solo originali

tr_idx = data_for_training["period_id"].isin(train_groups).to_numpy().nonzero()[0]
va_idx = data_for_training["period_id"].isin(val_groups).to_numpy().nonzero()[0]
print(f"Train rows: {len(tr_idx)} · Val rows: {len(va_idx)}")
if len(tr_idx) == 0 or len(va_idx) == 0:
    raise SystemExit("❌ Split vuoto: controlla i periodi disponibili.")

X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
y_tr, y_va = y_df[tr_idx], y_df[va_idx]

imputer = SimpleImputer(strategy="mean")
X_tr_imp = imputer.fit_transform(X_tr)
X_va_imp = imputer.transform(X_va)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr_imp)
X_va_s = scaler.transform(X_va_imp)

model = create_model(X_tr_s.shape[1], y_tr.shape[1])
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1)

history = model.fit(
    X_tr_s, y_tr,
    validation_data=(X_va_s, y_va),
    epochs=50,
    batch_size=512,
    verbose=1,
    callbacks=[es, rlr],
)

y_pred_probs = model.predict(X_va_s, verbose=0)

def plot_learning_curve(hist, metric, path):
    tr = hist.history.get(metric, [])
    va = hist.history.get(f"val_{metric}", [])
    if not tr or not va:
        return
    epochs = range(1, min(len(tr), len(va)) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, tr[:len(epochs)], "o-", label=f"Train {metric}")
    plt.plot(epochs, va[:len(epochs)], "o-", label=f"Val {metric}")
    plt.title(f"Curva Apprendimento: {metric}")
    plt.xlabel("Epoca"); plt.ylabel(metric.capitalize()); plt.legend(); plt.grid(True)
    plt.savefig(path)
    plt.show()

plot_learning_curve(history, "loss", SAVE_DIR / "learning_curve_loss.png")
plot_learning_curve(history, "precision", SAVE_DIR / "learning_curve_precision.png")

# --- Plot: ROC ---
plt.figure(figsize=(9,7))
for i, act in enumerate(ALL_ACTUATORS):
    if y_va[:, i].max() == 0 or y_va[:, i].min() == 1:
        continue  # label degenerata in val
    fpr, tpr, _ = roc_curve(y_va[:, i], y_pred_probs[:, i])
    plt.plot(fpr, tpr, label=f"{act} (AUC={auc(fpr, tpr):.3f})")
plt.plot([0,1],[0,1],"k--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Curva ROC")
plt.legend(); plt.grid(True)
plt.savefig(SAVE_DIR / "roc_curve_aggregated.png")
plt.show()

# --- Plot: Precision-Recall ---
plt.figure(figsize=(9,7))
for i, act in enumerate(ALL_ACTUATORS):
    if y_va[:, i].sum() == 0:
        continue
    pr, rc, _ = precision_recall_curve(y_va[:, i], y_pred_probs[:, i])
    ap = average_precision_score(y_va[:, i], y_pred_probs[:, i])
    plt.plot(rc, pr, label=f"{act} (AP={ap:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Curva Precision-Recall")
plt.legend(); plt.grid(True)
plt.savefig(SAVE_DIR / "precision_recall_aggregated.png")
plt.show()

# --- Soglie ottimali (F1) e report ---
optimal_thresholds = {}
print("\n--- Soglie Ottimali (F1) ---")
for i, act in enumerate(ALL_ACTUATORS):
    if y_va[:, i].sum() == 0 or (y_va[:, i] == 0).all():
        optimal_thresholds[act] = 0.5
        print(f" · {act}: 0.50 (degenerata in val)")
        continue
    pr, rc, th = precision_recall_curve(y_va[:, i], y_pred_probs[:, i])
    if len(th) == 0:
        optimal_thresholds[act] = 0.5
    else:
        f1 = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-8)
        optimal_thresholds[act] = float(th[np.nanargmax(f1)])
    print(f" · {act}: {optimal_thresholds[act]:.2f}")

thr_vec = np.array([optimal_thresholds[a] for a in ALL_ACTUATORS])[None, :]
y_pred_bin = (y_pred_probs > thr_vec).astype(int)

print("\n--- Report di Classificazione (validation) ---")
for i, act in enumerate(ALL_ACTUATORS):
    print(f"\n[{act}]")
    print(classification_report(y_va[:, i], y_pred_bin[:, i], digits=4, zero_division=0))

emr = np.all(y_va == y_pred_bin, axis=1).mean()
print(f"\nExact Match Ratio (EMR): {emr:.4f}")

print("\n✅ [SEZIONE 5] Addestramento+Validazione completati.")

# ==============================================================================
# SEZIONE 6 — ADDESTRAMENTO FINALE & SALVATAGGIO
# ==============================================================================
print("\n--- [SEZIONE 6] Addestramento Finale ---")

model.save(SAVE_DIR / "model.keras")
joblib.dump(scaler, SAVE_DIR / "scaler.joblib")
with open(SAVE_DIR / "features.json", "w") as f:
    json.dump(features, f, indent=2)

thresholds_to_save = {key: float(value) for key, value in optimal_thresholds.items()}
with open(SAVE_DIR / "thresholds.json", "w") as f:
    json.dump(thresholds_to_save, f, indent=4)

metrics = {
    "emr": float(emr),
}
with open(SAVE_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Modello/scaler/feature salvati in: {SAVE_DIR}")
