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
from colab_models.common import load_unified_dataset, get_data_from_periods, get_actuator_names, log_actuator_stats, augment_specific_groups_with_noise, focal_loss

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

print("✅ [SEZIONE 2] Dati caricati.")

# ==============================================================================
# SEZIONE 3 — FEATURE ENGINEERING (pulita e causale)
# ==============================================================================

print("\n--- [SEZIONE 3] Feature Engineering ---")

final_df = add_features_actuator_classification(final_df)
print(final_df.head())

print(f"✅ [SEZIONE 3] Completata. Shape: {final_df.shape}")


# ==============================================================================
# SEZIONE 4 — DEFINIZIONE FEATURE E SPLIT TRAIN/TEST
# ==============================================================================
print("\n--- [SEZIONE 4] Definizione Feature e Split ---")

data_for_training = get_data_from_periods(final_df, TRAINING_PERIODS_FILE)
data_for_test = get_data_from_periods(final_df, TEST_PERIODS_FILE)

features = final_features_actuator_classification()
targets  = STATE_COLS.copy()

print(f"Features: {len(features)} · Targets: {len(targets)}")

print(f"Righe Training: {len(data_for_training)} · Test: {len(data_for_test)}")

log_actuator_stats(data_for_training, STATE_COLS, "Training Set")
log_actuator_stats(data_for_test, STATE_COLS, "Test Set")

print("✅ [SEZIONE 4] OK.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras

def create_model(input_dim, output_dim):
    """Crea e compila il modello Keras."""
    x_in = Input(shape=(input_dim,))
    x = Dense(64, kernel_regularizer=keras.regularizers.l2(1e-3))(x_in)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(32, kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)
    y_out = Dense(output_dim, activation="sigmoid")(x)
    m = Model(inputs=x_in, outputs=y_out)
    m.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
              loss=focal_loss(alpha=0.25, gamma=2.0),
              metrics=["binary_accuracy", "precision", "recall"])
    return m

def plot_avg_learning_curve(histories, metric, path):
    """Plotta la media e la deviazione standard di una metrica su più fold."""
    train_metrics = [h.history.get(metric, []) for h in histories]
    val_metrics = [h.history.get(f"val_{metric}", []) for h in histories]
    if not any(train_metrics) or not any(val_metrics): return
    max_epochs = max(len(m) for m in train_metrics + val_metrics if m)
    def pad_metric(metrics, max_len):
        padded = np.full((len(metrics), max_len), np.nan)
        for i, m in enumerate(metrics): padded[i, :len(m)] = m
        return padded
    train_padded = pad_metric(train_metrics, max_epochs)
    val_padded = pad_metric(val_metrics, max_epochs)
    mean_train, std_train = np.nanmean(train_padded, axis=0), np.nanstd(train_padded, axis=0)
    mean_val, std_val = np.nanmean(val_padded, axis=0), np.nanstd(val_padded, axis=0)
    epochs = range(1, max_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_train, "o-", color="blue", label=f"Train {metric} (Media)")
    plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, color="blue", alpha=0.1)
    plt.plot(epochs, mean_val, "o-", color="orange", label=f"Val {metric} (Media)")
    plt.fill_between(epochs, mean_val - std_val, mean_val + std_val, color="orange", alpha=0.1)
    plt.title(f"Curva Apprendimento Media ({len(histories)} Folds): {metric}")
    plt.xlabel("Epoca"); plt.ylabel(metric.capitalize()); plt.legend(); plt.grid(True)
    plt.savefig(path)
    plt.show()


# --------------------------------------------------------------------------
# 1. PREPARAZIONE DATI PER LA CROSS-VALIDATION
# --------------------------------------------------------------------------
print("✅ [SEZIONE 5] Inizio Addestramento e Validazione...")

data_for_training = data_for_training.sort_values("utc_datetime").reset_index(drop=True)

is_original = ~data_for_training["period_id"].str.contains("__")
data_original = data_for_training[is_original].copy()

if data_original.empty:
    raise SystemExit("❌ Non sono stati trovati dati originali per lo split.")

X_original_df = data_original[features]
y_original_df = data_original[targets].astype(int)
groups = data_original['period_id'].str.split('__').str[0]

# --------------------------------------------------------------------------
# 2. CICLO DI CROSS-VALIDATION
# --------------------------------------------------------------------------
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

histories, val_scores, all_predictions, all_true_values = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_original_df, y_original_df, groups)):
    print(f"\n===== FOLD {fold + 1}/{n_splits} =====")

    train_fold_original = data_original.iloc[train_idx]
    val_fold = data_original.iloc[val_idx]

    target1 = ['state_Riscaldamento']
    aug_df_1 = augment_specific_groups_with_noise(
        df=train_fold_original,
        n_duplicates=2,
        target_actuators=target1,
        noise_min=0.01,
        noise_max=0.03,
    )

    target2 = ['state_Umidificatore', 'state_Deumidificatore']
    aug_df_2 = augment_specific_groups_with_noise(
        df=train_fold_original,
        n_duplicates=2,
        target_actuators=target2,
        noise_min=0.01,
        noise_max=0.03,
    )

    if not aug_df_1.empty or not aug_df_2.empty:
        righe_aggiunte = len(aug_df_1) + len(aug_df_2)
        print(f"[AUG] Aggiunte {righe_aggiunte} righe (solo training).")
        final_train_fold = pd.concat([train_fold_original, aug_df_1, aug_df_2], ignore_index=True)

    X_tr, y_tr = train_fold_original[features], train_fold_original[targets].astype(int).values
    X_va, y_va = val_fold[features], val_fold[targets].astype(int).values
    print(f"Train rows: {len(X_tr)} · Val rows: {len(X_va)}")

    imputer = SimpleImputer(strategy="mean")
    X_tr_imp = imputer.fit_transform(X_tr)
    X_va_imp = imputer.transform(X_va)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_imp)
    X_va_s = scaler.transform(X_va_imp)

    model = create_model(X_tr_s.shape[1], y_tr.shape[1])
    es = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)

    history = model.fit(
        X_tr_s, y_tr,
        validation_data=(X_va_s, y_va),
        epochs=25,
        batch_size=2048,
        verbose=1,
        callbacks=[es, rlr],
    )
    histories.append(history)

    score = model.evaluate(X_va_s, y_va, verbose=0)
    val_scores.append(score)
    print(f"Fold {fold + 1} Val Loss: {score[0]:.4f}, Val Accuracy: {score[1]:.4f}")

    y_pred_probs = model.predict(X_va_s, verbose=0)
    all_predictions.append(y_pred_probs)
    all_true_values.append(y_va)

# --------------------------------------------------------------------------
# 3. ANALISI DEI RISULTATI AGGREGATI
# --------------------------------------------------------------------------
print("\n===== ANALISI AGGREGATA SUI RISULTATI DELLA CROSS-VALIDATION =====")

# Unisci le predizioni di tutti i fold per avere un quadro completo
y_va = np.concatenate(all_true_values)
y_pred_probs = np.concatenate(all_predictions)

# Plot delle curve di apprendimento medie
plot_avg_learning_curve(histories, "loss", SAVE_DIR / "learning_curve_loss_avg.png")
plot_avg_learning_curve(histories, "precision", SAVE_DIR / "learning_curve_precision_avg.png")

# Plot: Curva ROC
plt.figure(figsize=(9,7))
for i, act in enumerate(ALL_ACTUATORS):
    if y_va[:, i].max() == 0 or y_va[:, i].min() == 1: continue
    fpr, tpr, _ = roc_curve(y_va[:, i], y_pred_probs[:, i])
    plt.plot(fpr, tpr, label=f"{act} (AUC={auc(fpr, tpr):.3f})")
plt.plot([0,1],[0,1],"k--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Curva ROC (Aggregata su CV)")
plt.legend(); plt.grid(True)
plt.savefig(SAVE_DIR / "roc_curve_aggregated.png")
plt.show()

# Plot: Curva Precision-Recall
plt.figure(figsize=(9,7))
for i, act in enumerate(ALL_ACTUATORS):
    if y_va[:, i].sum() == 0: continue
    pr, rc, _ = precision_recall_curve(y_va[:, i], y_pred_probs[:, i])
    ap = average_precision_score(y_va[:, i], y_pred_probs[:, i])
    plt.plot(rc, pr, label=f"{act} (AP={ap:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Curva Precision-Recall (Aggregata su CV)")
plt.legend(); plt.grid(True)
plt.savefig(SAVE_DIR / "precision_recall_aggregated.png")
plt.show()

# Calcolo soglie ottimali e report di classificazione
optimal_thresholds = {}
print("\n--- Soglie Ottimali (F1) - Calcolate su tutti i fold ---")
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

print("\n--- Report di Classificazione (aggregato su validation) ---")
for i, act in enumerate(ALL_ACTUATORS):
    print(f"\n[{act}]")
    print(classification_report(y_va[:, i], y_pred_bin[:, i], digits=4, zero_division=0))

emr = np.all(y_va == y_pred_bin, axis=1).mean()
print(f"\nExact Match Ratio (EMR): {emr:.4f}")

# Stampa delle performance medie finali
avg_val_loss = np.mean([s[0] for s in val_scores])
avg_val_accuracy = np.mean([s[1] for s in val_scores])
print("\n--- Performance Medie Finali ---")
print(f"Validation Loss Media: {avg_val_loss:.4f}")
print(f"Validation Accuracy Media: {avg_val_accuracy:.4f}")


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
