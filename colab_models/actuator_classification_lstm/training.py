# ==============================================================================
# SEZIONE 0 — SETUP E CONFIGURAZIONE
# ==============================================================================
print("--- [SEZIONE 0] Inizio Setup e Configurazione ---")
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers

import sys
import os

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = (CURRENT_DIR / "../..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Utils progetto (come nel tuo script attuale)
from utils.feature_engineering import (
    ensure_min_columns_actuator_classification,
    add_features_actuator_classification,
    final_features_actuator_classification,
)
from colab_models.common import (
    load_unified_dataset,
    get_data_from_periods,
    get_actuator_names,
)

from utils.feature_engineering import (
    build_sequences,
)

# --- Paths & costanti
SAVE_DIR = Path(__file__).parent / "output"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Directory di output pronta: {SAVE_DIR}")

BASE_PATH = Path("/content/drive/MyDrive/Tesi")
DATASET_COMPLETO_PATH = BASE_PATH / "DatasetSegmentato"
TRAINING_PERIODS_FILE = BASE_PATH / "training_periods.csv"
TEST_PERIODS_FILE = BASE_PATH / "test_periods.csv"

ALL_ACTUATORS = get_actuator_names()
STATE_COLS = [f"state_{a}" for a in ALL_ACTUATORS]

# Config sequenze
WINDOW = 180       # 3 ore a 1-min cadence
STRIDE = 2         # velocizza (puoi rimettere 1 se vuoi massimizzare i campioni)
TIME_COL = "utc_datetime"
GROUP_COLS = ("device", "period_id")

BATCH_SIZE = 128
EPOCHS_MAX = 30
LR = 3e-4

print("✅ [SEZIONE 0] Setup completato.")

# ==============================================================================
# SEZIONE 1 — FUNZIONI DI SUPPORTO PER ANALISI
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


def temporal_blocked_folds(times: np.ndarray, n_splits: int = 5):
    """
    Crea fold temporali "bloccati" SENZA shuffle, dividendo per quantili del tempo finale finestra.
    Ritorna lista di (train_idx, val_idx).
    """
    order = np.argsort(times)
    times_sorted = times[order]
    n = len(times_sorted)
    # Indici di taglio per 5 blocchi uguali
    cuts = [int(n * k / n_splits) for k in range(1, n_splits)]
    blocks = np.split(order, cuts)

    folds = []
    # Per ciascun blocco Bk, usa tutti i blocchi < k per train, Bk per val
    for k in range(1, len(blocks)):
        train_idx = np.concatenate(blocks[:k])
        val_idx = blocks[k]
        folds.append((train_idx, val_idx))
    # Se vuoi avere esattamente n_splits fold, prendi anche l'ultimo:
    if len(blocks) >= 2:
        folds.append((np.concatenate(blocks[:-1]), blocks[-1]))
    return folds[:n_splits]


def plot_mean_learning_curve(histories, metric, save_path):
    seqs_tr = [h.history.get(metric, []) for h in histories]
    seqs_va = [h.history.get(f"val_{metric}", []) for h in histories]
    min_len = min([len(s) for s in (seqs_tr + seqs_va) if len(s) > 0]) if histories else 0
    if min_len == 0:
        return
    tr = np.array([s[:min_len] for s in seqs_tr]).mean(axis=0)
    va = np.array([s[:min_len] for s in seqs_va]).mean(axis=0)
    epochs = range(1, min_len + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, tr, "o-", label=f"Train {metric}")
    plt.plot(epochs, va, "o-", label=f"Val {metric}")
    plt.title(f"Curva Apprendimento Media: {metric}")
    plt.xlabel("Epoca"); plt.ylabel(metric.capitalize()); plt.legend(); plt.grid(True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def compute_optimal_thresholds(y_true, y_prob, actuator_names):
    from sklearn.metrics import precision_recall_curve
    thresholds = {}
    for i, act in enumerate(actuator_names):
        pr, rc, th = precision_recall_curve(y_true[:, i], y_prob[:, i])
        if len(th) == 0:
            thresholds[act] = 0.5
        else:
            f1 = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-8)
            thresholds[act] = float(th[np.argmax(f1)])
    return thresholds


def fit_scaler(train_df, feature_cols):
    """Fit dello StandardScaler solo sul training set."""
    X = train_df[feature_cols].astype(float).fillna(0)
    scaler = StandardScaler()
    scaler.fit(X.values)
    return scaler

def transform_with_scaler(df, feature_cols, scaler):
    """Applica lo scaler a un DataFrame e restituisce una copia scalata."""
    X = df[feature_cols].astype(float).fillna(0)
    Xs = scaler.transform(X.values)
    out = df.copy()
    out.loc[:, feature_cols] = Xs
    return out


def build_lstm_model(input_shape, output_dim, units=64, dropout=0.2, lr=3e-4):
    """Crea un modello LSTM semplice per classificazione multi-label."""
    inp = keras.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, strides=2, padding="same")(inp)
    x = layers.ReLU()(x)
    x = layers.GRU(units, dropout=dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(output_dim, activation="sigmoid")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["binary_accuracy", "precision", "recall"]
    )
    return model

print("✅ [SEZIONE 1] Funzioni analisi pronte.")

# ==============================================================================
# SEZIONE 2 — CARICAMENTO DEL DATASET
# ==============================================================================
print("\n--- [SEZIONE 2] Caricamento Dati ---")

final_df = load_unified_dataset(DATASET_COMPLETO_PATH)
if final_df.empty:
    raise SystemExit("DataFrame vuoto. Interrompo.")

final_df["utc_datetime"] = pd.to_datetime(final_df["utc_datetime"], errors="coerce", utc=True)
final_df.dropna(subset=["utc_datetime", "device"], inplace=True)
final_df.sort_values(["device", "period_id", "utc_datetime"], inplace=True)

for col in STATE_COLS:
    if col in final_df.columns:
        final_df[col] = final_df[col].fillna(0).astype(int)

print("✅ [SEZIONE 2] Dati caricati.")



# ==============================================================================
# SEZIONE 3 — FEATURE ENGINEERING (causale)
# ==============================================================================
print("\n--- [SEZIONE 3] Feature Engineering ---")

ensure_min_columns_actuator_classification(final_df)
df = add_features_actuator_classification(final_df)

print(f"✅ [SEZIONE 3] Completata. Shape: {df.shape}")


# ==============================================================================
# SEZIONE 4 — DEFINIZIONE FEATURE, SPLIT PERIODI, SCALING
# ==============================================================================
print("\n--- [SEZIONE 4] Definizione Feature e Split ---")

features = final_features_actuator_classification()
targets = STATE_COLS.copy()
print(f"Features: {len(features)} · Targets: {len(targets)}")

train_df_rows = get_data_from_periods(df, TRAINING_PERIODS_FILE)
test_df_rows  = get_data_from_periods(df, TEST_PERIODS_FILE)

print(f"Righe Training (raw): {len(train_df_rows)} · Test (raw): {len(test_df_rows)}")
log_actuator_stats(train_df_rows, "Training Set (raw)")
log_actuator_stats(test_df_rows, "Test Set (raw)")

# Scaler fit SOLO sul training raw (per evitare leakage), poi applicato a train/test
scaler = fit_scaler(train_df_rows, features)
train_df = transform_with_scaler(train_df_rows, features, scaler)
test_df  = transform_with_scaler(test_df_rows, features, scaler)

print("✅ [SEZIONE 4] Scaling applicato (fit su train only).")

# ==============================================================================
# SEZIONE 5 — WINDOWING (costruzione sequenze)
# ==============================================================================
print("\n--- [SEZIONE 5] Windowing Sequenze ---")

X_train, y_train, t_train = build_sequences(
    train_df, features, targets, window=WINDOW, stride=STRIDE,
    time_col=TIME_COL, group_cols=GROUP_COLS
)
X_test, y_test, t_test = build_sequences(
    test_df, features, targets, window=WINDOW, stride=1,
    time_col=TIME_COL, group_cols=GROUP_COLS
)

print(f"Train sequences: {X_train.shape}  ·  Test sequences: {X_test.shape}")
input_shape = (X_train.shape[1], X_train.shape[2])
output_dim = y_train.shape[1]
print(f"Input shape (W,F): {input_shape}  ·  Output dim: {output_dim}")

print("✅ [SEZIONE 5] Sequenze pronte.")


# ==============================================================================
# SEZIONE 6 — ADD. & VALUTAZIONE (Temporal CV)
# ==============================================================================
print("\n--- [SEZIONE 6] Addestramento e Valutazione con Temporal CV ---")
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report

folds = temporal_blocked_folds(t_train, n_splits=5)
histories = []
val_probs_all = []
val_true_all = []

for i, (tr_idx, va_idx) in enumerate(folds, 1):
    print(f"---------------- Fold {i}/{len(folds)} ----------------")
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    model = build_lstm_model(input_shape, output_dim, units=64, dropout=0.2, lr=LR)
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
    history = model.fit(
        X_tr, y_tr,
        epochs=EPOCHS_MAX,
        batch_size=BATCH_SIZE,
        validation_data=(X_va, y_va),
        callbacks=[es],
        verbose=0
    )
    histories.append(history)

    preds = model.predict(X_va, verbose=0)
    val_probs_all.append(preds)
    val_true_all.append(y_va)

# Aggrego i risultati di validazione
y_val_all = np.concatenate(val_true_all, axis=0)
y_pred_probs_all = np.concatenate(val_probs_all, axis=0)

# Curve apprendimento medie
plot_mean_learning_curve(histories, "loss", save_path=SAVE_DIR / "learning_curve_loss.png")
plot_mean_learning_curve(histories, "precision", save_path=SAVE_DIR / "learning_curve_precision.png")

# ROC aggregata
plt.figure(figsize=(9, 7))
for j, act in enumerate(ALL_ACTUATORS):
    fpr, tpr, _ = roc_curve(y_val_all[:, j], y_pred_probs_all[:, j])
    plt.plot(fpr, tpr, label=f"{act} (AUC={auc(fpr, tpr):.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Curva ROC Aggregata (Validazione)")
plt.legend(); plt.grid(True)
plt.savefig(SAVE_DIR / "roc_curve_aggregated.png", bbox_inches="tight")
plt.close()

# PR aggregata
plt.figure(figsize=(9, 7))
for j, act in enumerate(ALL_ACTUATORS):
    pr, rc, _ = precision_recall_curve(y_val_all[:, j], y_pred_probs_all[:, j])
    ap = average_precision_score(y_val_all[:, j], y_pred_probs_all[:, j])
    plt.plot(rc, pr, label=f"{act} (AP={ap:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Curva Precision-Recall Aggregata (Validazione)")
plt.legend(); plt.grid(True)
plt.savefig(SAVE_DIR / "precision_recall_aggregated.png", bbox_inches="tight")
plt.close()

# Soglie attuatore-specifiche per F1
optimal_thresholds = compute_optimal_thresholds(y_val_all, y_pred_probs_all, ALL_ACTUATORS)
thr_vec = np.array([optimal_thresholds[a] for a in ALL_ACTUATORS])[None, :]
y_pred_all_opt = (y_pred_probs_all > thr_vec).astype(int)

# Report per attuatore
print("\n--- Report di Classificazione (validazione aggregata) ---")
for j, act in enumerate(ALL_ACTUATORS):
    print(f"\n[{act}]")
    print(classification_report(y_val_all[:, j], y_pred_all_opt[:, j], digits=4, zero_division=0))

emr = np.all(y_val_all == y_pred_all_opt, axis=1).mean()
print(f"\nExact Match Ratio (EMR - validazione): {emr:.4f}")

print("\n✅ [SEZIONE 6] Valutazione CV completata.")


# ==============================================================================
# SEZIONE 7 — ADD. FINALE SU TUTTO IL TRAIN & TEST REPORT
# ==============================================================================
print("\n--- [SEZIONE 7] Addestramento Finale e Test ---")

# Epoche medie dalle curve (fallback >=5)
avg_epochs = int(np.mean([len(h.history.get("loss", [])) for h in histories if h.history.get("loss")]))
avg_epochs = max(avg_epochs, 5)
print(f"Addestramento finale per {avg_epochs} epoche...")

final_model = build_lstm_model(input_shape, output_dim, units=128, dropout=0.2, lr=LR)
final_model.fit(
    X_train, y_train,
    epochs=avg_epochs,
    batch_size=BATCH_SIZE,
    verbose=0
)

# Inference sul test
test_probs = final_model.predict(X_test, verbose=0)

# Applica soglie F1 trovate in validazione
thr_vec_test = np.array([optimal_thresholds[a] for a in ALL_ACTUATORS])[None, :]
test_pred = (test_probs > thr_vec_test).astype(int)

# Report test
from sklearn.metrics import classification_report
print("\n--- Report di Classificazione (TEST) ---")
for j, act in enumerate(ALL_ACTUATORS):
    print(f"\n[{act}]")
    print(classification_report(y_test[:, j], test_pred[:, j], digits=4, zero_division=0))

emr_test = np.all(y_test == test_pred, axis=1).mean()
print(f"\nExact Match Ratio (EMR - test): {emr_test:.4f}")

# ==============================================================================
# SEZIONE 8 — SALVATAGGI
# ==============================================================================
print("\n--- [SEZIONE 8] Salvataggi ---")

# Modello/scaler/feature/thresholds
final_model.save(SAVE_DIR / "model_lstm.keras")
joblib.dump(scaler, SAVE_DIR / "scaler.joblib")

with open(SAVE_DIR / "features.json", "w") as f:
    json.dump(features, f, indent=2)

with open(SAVE_DIR / "thresholds.json", "w") as f:
    json.dump({k: float(v) for k, v in optimal_thresholds.items()}, f, indent=2)

metrics = {
    "cv_emr": float(emr),
    "test_emr": float(emr_test),
    "window": int(WINDOW),
    "stride_train": int(STRIDE),
    "batch_size": int(BATCH_SIZE),
    "lr": float(LR),
    "epochs_final": int(avg_epochs),
}
with open(SAVE_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Artefatti salvati in: {SAVE_DIR}")