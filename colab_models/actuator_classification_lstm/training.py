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
    final_features_actuator_classification_lstm,
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
WINDOW = 30
STRIDE = 1
TIME_COL = "utc_datetime"
GROUP_COLS = ("device", "period_id")

BATCH_SIZE = 256
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
    X = df[feature_cols].astype(np.float32).fillna(0)   # <- float32
    Xs = scaler.transform(X.values).astype(np.float32)  # <- float32
    out = df.copy()
    out.loc[:, feature_cols] = Xs
    return out

def rebalance_folds_by_class(folds, y_all, min_pos=1):
    good = []
    for tr, va in folds:
        ok = True
        for i in range(y_all.shape[1]):
            if y_all[va, i].sum() < min_pos:
                ok = False; break
        if ok: good.append((tr, va))
    return good

def build_lstm_model(input_shape, output_dim):
    """Crea un modello LSTM semplice per classificazione multi-label."""
    inp = keras.Input(shape=input_shape)
    x = layers.Conv1D(64, 5, strides=2, padding="same")(inp)
    x = layers.ReLU()(x)
    x = layers.GRU(64, dropout=0.2)(x)  # torna a 64/0.2: più stabile
    out = layers.Dense(output_dim, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4, clipnorm=1.0),  # <— clipnorm
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

features = final_features_actuator_classification_lstm()
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
# SEZIONE 6 — ADD. & VALUTAZIONE (Holdout temporale robusto)
# ==============================================================================
print("\n--- [SEZIONE 6] Addestramento e Valutazione con Holdout ---")
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report

order = np.argsort(t_train)
n = len(order)

# Holdout iniziale 85/15
cut = int(n * 0.85)

def has_pos_neg(idx, y, min_pos=20):
    # almeno min_pos positivi e almeno 1 negativo per ogni classe
    yv = y[idx]
    pos = yv.sum(axis=0)
    neg = (len(yv) - pos)
    return (pos >= min_pos).all() and (neg >= 1).all()

# allarga finché tutte le classi hanno abbastanza esempi
step = int(n * 0.05)
while cut > 0 and not has_pos_neg(order[cut:], y_train, min_pos=20):
    cut = max(0, cut - step)

tr_idx, va_idx = order[:cut], order[cut:]
print(f"Holdout: train={len(tr_idx)}  val={len(va_idx)}")
print("Positivi VAL per classe:", y_train[va_idx].sum(0).astype(int).tolist())

X_tr, X_va = X_train[tr_idx], X_train[va_idx]
y_tr, y_va = y_train[tr_idx], y_train[va_idx]

pos_rate = y_tr.mean(axis=0)
pos_w = ((1.0 - pos_rate) / (pos_rate + 1e-6)).clip(1.0, 20.0).astype(np.float32)

# Collassa a pesi PER-CAMPIONE 1D (evita broadcast error)
# Idea: media dei pesi delle classi positive del campione; se nessun positivo -> 1.0
num_pos = y_tr.sum(axis=1, keepdims=False)                  # [N]
sum_w   = (y_tr * pos_w).sum(axis=1)                        # [N]
sample_w_tr = np.where(num_pos > 0, sum_w / num_pos, 1.0)   # [N]
sample_w_tr = sample_w_tr.astype(np.float32)

print("Train pos rate per classe:", np.round(pos_rate, 4).tolist())
print("Peso medio (train):", float(sample_w_tr.mean()))

model = build_lstm_model(input_shape, output_dim)
es1 = keras.callbacks.EarlyStopping(monitor="val_recall", mode="max", patience=3, restore_best_weights=True)
es2 = keras.callbacks.EarlyStopping(monitor="val_loss",  mode="min", patience=3, restore_best_weights=True)
rlr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  mode="min", factor=0.5, patience=2, min_lr=1e-5)

history = model.fit(
    X_tr, y_tr,
    epochs=EPOCHS_MAX,
    batch_size=BATCH_SIZE,
    validation_data=(X_va, y_va),
    sample_weight=sample_w_tr,   # <— aggiungi questo
    callbacks=[es1, es2, rlr],
    verbose=1
)

# per riusare il resto invariato
histories = [history]
y_val_all = y_va
y_pred_probs_all = model.predict(X_va, verbose=0)

plot_mean_learning_curve(histories, "loss", save_path=SAVE_DIR / "learning_curve_loss.png")
plot_mean_learning_curve(histories, "precision", save_path=SAVE_DIR / "learning_curve_precision.png")

# --- grafici invariati ma con guard per classi costanti ---
plt.figure(figsize=(9, 7))
for j, act in enumerate(ALL_ACTUATORS):
    yj = y_val_all[:, j]; pj = y_pred_probs_all[:, j]
    if yj.max() == yj.min():
        print(f"ROC: salto {act} (classe costante in validazione).")
        continue
    fpr, tpr, _ = roc_curve(yj, pj)
    plt.plot(fpr, tpr, label=f"{act} (AUC={auc(fpr, tpr):.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Curva ROC Aggregata (Validazione)")
plt.legend(); plt.grid(True)
plt.savefig(SAVE_DIR / "roc_curve_aggregated.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(9, 7))
for j, act in enumerate(ALL_ACTUATORS):
    yj = y_val_all[:, j]; pj = y_pred_probs_all[:, j]
    if yj.max() == yj.min():
        print(f"PR: salto {act} (classe costante in validazione).")
        continue
    pr, rc, _ = precision_recall_curve(yj, pj)
    ap = average_precision_score(yj, pj)
    plt.plot(rc, pr, label=f"{act} (AP={ap:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Curva Precision-Recall Aggregata (Validazione)")
plt.legend(); plt.grid(True)
plt.savefig(SAVE_DIR / "precision_recall_aggregated.png", bbox_inches="tight")
plt.close()

from sklearn.metrics import precision_recall_curve

# target di precisione per classe (puoi tarare questi numeri)
# ordine: [Umidificatore, Finestra, Deumidificatore, Riscaldamento, Clima]
prec_target = [0.60, 0.70, 0.60, 0.60, 0.75]

thresholds = {}
for j, act in enumerate(ALL_ACTUATORS):
    yj = y_va[:, j]
    pj = model.predict(X_va, verbose=0)[:, j]  # oppure usa y_pred_probs_all[:, j] se lo hai già
    pr, rc, th = precision_recall_curve(yj, pj)
    if len(th) == 0:
        thresholds[act] = 0.5
        continue
    # trova la prima soglia che raggiunge la precisione target (o la massima precisione disponibile)
    idx = np.where(pr[:-1] >= prec_target[j])[0]
    best = th[idx[0]] if len(idx) else th[np.argmax(pr[:-1])]
    thresholds[act] = float(best)

thr_vec = np.array([thresholds[a] for a in ALL_ACTUATORS])[None, :]
y_pred_probs_all = model.predict(X_va, verbose=0)
y_pred_all_opt = (y_pred_probs_all > thr_vec).astype(int)

print("Soglie per precisione target:", thresholds)

print("\n--- Report di Classificazione (validazione holdout) ---")
for j, act in enumerate(ALL_ACTUATORS):
    print(f"\n[{act}]")
    print(classification_report(y_val_all[:, j], y_pred_all_opt[:, j], digits=4, zero_division=0))

emr = np.all(y_val_all == y_pred_all_opt, axis=1).mean()
print(f"\nExact Match Ratio (EMR - validazione): {emr:.4f}")

print("\n✅ [SEZIONE 6] Valutazione holdout completata.")

# ==============================================================================
# SEZIONE 7 — TEST CON MODELLO EARLY-STOPPED (NO RETRAIN)
# ==============================================================================
print("\n--- [SEZIONE 7] Test con modello early-stopped (niente retrain) ---")

# Inference sul test
test_probs = model.predict(X_test, verbose=0)

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
model.save(SAVE_DIR / "model_lstm.keras")
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
}
with open(SAVE_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Artefatti salvati in: {SAVE_DIR}")