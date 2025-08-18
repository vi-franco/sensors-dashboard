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
from colab_models.common import load_unified_dataset, get_data_from_periods, get_actuator_names, log_actuator_stats

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

train_base_raw = get_data_from_periods(final_df, TRAINING_PERIODS_FILE)
N_MIN = 10000  #
aug_df = augment_minority_periods_on_windows(
    train_base=train_base_raw,
    state_cols=STATE_COLS,
    time_col="utc_datetime",
    group_col="period_id",
    min_pos_rows_per_act=N_MIN,
    context_minutes=30,
    noise_pct=0.01,
    id_suffix="AUG",
)
if not aug_df.empty:
    print(f"[AUG] Aggiunte {len(aug_df)} righe (solo training).")
    # Concateniamo al dataset grezzo
    final_df = pd.concat([final_df, aug_df], ignore_index=True)

print("✅ [SEZIONE 2] Dati caricati.")

# ==============================================================================
# SEZIONE 3 — FEATURE ENGINEERING (pulita e causale)
# ==============================================================================

print("\n--- [SEZIONE 3] Feature Engineering ---")

df = add_features_actuator_classification(final_df)
print(df.head())

print(f"✅ [SEZIONE 3] Completata. Shape: {df.shape}")


# ==============================================================================
# SEZIONE 4 — DEFINIZIONE FEATURE E SPLIT TRAIN/TEST
# ==============================================================================
print("\n--- [SEZIONE 4] Definizione Feature e Split ---")

features = final_features_actuator_classification()
targets  = STATE_COLS.copy()

print(f"Features: {len(features)} · Targets: {len(targets)}")

data_for_training = get_data_from_periods(df, TRAINING_PERIODS_FILE)
test_df           = get_data_from_periods(df, TEST_PERIODS_FILE)

print(f"Righe Training: {len(data_for_training)} · Test: {len(test_df)}")

log_actuator_stats(data_for_training, STATE_COLS, "Training Set")
log_actuator_stats(test_df, STATE_COLS, "Test Set")

print("✅ [SEZIONE 4] OK.")

# ==============================================================================
# SEZIONE 5 — ADDESTRAMENTO & VALUTAZIONE (Group Expanding Time-Series CV)
# ==============================================================================

print("\n--- [SEZIONE 5] Addestramento e Valutazione con Group Expanding Split ---")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import keras

def create_model(input_dim, output_dim):
    inp = Input(shape=(input_dim,))
    x = Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(inp)
    x = Dropout(0.1)(x)
    x = Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.1)(x)
    out = Dense(output_dim, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["binary_accuracy", "precision", "recall"],
    )
    return model

def group_expanding_splits(df, group_col="period_id", time_col="utc_datetime",
                           n_splits=3, embargo_groups=1, warmup_blocks=1):
    grp_order = (df.groupby(group_col)[time_col].min().sort_values())
    groups = grp_order.index.to_numpy()
    m = len(groups)
    n_blocks = n_splits + warmup_blocks
    block_sizes = np.full(n_blocks, m // n_blocks, dtype=int)
    block_sizes[: m % n_blocks] += 1
    cuts = np.cumsum(block_sizes)

    for k in range(n_splits):
        val_start = cuts[warmup_blocks - 1 + k - 1] if (warmup_blocks - 1 + k - 1) >= 0 else 0
        val_end   = cuts[warmup_blocks - 1 + k]
        train_groups = groups[:max(0, val_start - embargo_groups)]
        val_groups   = groups[val_start:val_end]
        tr_idx = df[df[group_col].isin(train_groups)].index.values
        va_idx = df[df[group_col].isin(val_groups)].index.values
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        yield tr_idx, va_idx, val_groups

if data_for_training.empty:
    raise SystemExit("❌ Dataset di training vuoto. Impossibile addestrare.")

df_train = data_for_training.sort_values("utc_datetime").reset_index(drop=True)
X_df = df_train[features]
y_df = df_train[targets].astype(int)

histories, all_y_val, all_y_pred_probs = [], [], []

for fold_no, (tr_idx, va_idx, val_groups) in enumerate(
    group_expanding_splits(df_train, "period_id", "utc_datetime",
                           n_splits=3, embargo_groups=1, warmup_blocks=1), 1
):
    print(f"Fold {fold_no} - val groups: {list(val_groups)} (train={len(tr_idx)}, val={len(va_idx)})")
    X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
    y_tr, y_va = y_df.iloc[tr_idx], y_df.iloc[va_idx]

    # Imputazione e scaling: fit SOLO sul train
    imputer = SimpleImputer(strategy="mean")
    X_tr_imp = imputer.fit_transform(X_tr)
    X_va_imp = imputer.transform(X_va)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_imp)
    X_va_s = scaler.transform(X_va_imp)

    model = create_model(X_tr_s.shape[1], y_tr.shape[1])
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)
    history = model.fit(
        X_tr_s, y_tr,
        epochs=20,
        batch_size=64,
        validation_data=(X_va_s, y_va),
        callbacks=[es],
        verbose=1,
    )
    histories.append(history)

    preds = model.predict(X_va_s, verbose=0)
    all_y_val.append(y_va.values)
    all_y_pred_probs.append(preds)

# Aggregazione dei fold di validazione
y_val_all = np.concatenate(all_y_val, axis=0)
y_pred_probs_all = np.concatenate(all_y_pred_probs, axis=0)


# Curve apprendimento medie
def plot_mean_learning_curve(histories, metric, save_path):
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
    plt.title(f"Curva Apprendimento Media: {metric}")
    plt.xlabel("Epoca"); plt.ylabel(metric.capitalize()); plt.legend(); plt.grid(True)
    plt.savefig(save_path) # --- MODIFICA: Salva il grafico
    plt.show()

plot_mean_learning_curve(histories, "loss", save_path=SAVE_DIR / "learning_curve_loss.png")
plot_mean_learning_curve(histories, "precision", save_path=SAVE_DIR / "learning_curve_precision.png")


# ROC & PR
plt.figure(figsize=(9,7))
for i, act in enumerate(ALL_ACTUATORS):
    fpr, tpr, _ = roc_curve(y_val_all[:, i], y_pred_probs_all[:, i])
    plt.plot(fpr, tpr, label=f"{act} (AUC={auc(fpr, tpr):.3f})")
plt.plot([0,1],[0,1],"k--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Curva ROC Aggregata")
plt.legend(); plt.grid(True)
plt.savefig(SAVE_DIR / "roc_curve_aggregated.png") # --- MODIFICA: Salva il grafico
plt.show()


for i, act in enumerate(ALL_ACTUATORS):
    pr, rc, _ = precision_recall_curve(y_val_all[:, i], y_pred_probs_all[:, i])
    ap = average_precision_score(y_val_all[:, i], y_pred_probs_all[:, i])
    plt.plot(rc, pr, label=f"{act} (AP={ap:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Curva Precision-Recall Aggregata")
plt.legend(); plt.grid(True)
plt.savefig(SAVE_DIR / "precision_recall_aggregated.png") # --- MODIFICA: Salva il grafico
plt.show()


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
model_final.fit(X_scaled_final, y_df, epochs=avg_epochs, batch_size=128, verbose=1)

model_final.save(SAVE_DIR / "model.keras")
joblib.dump(scaler_final, SAVE_DIR / "scaler.joblib")
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
