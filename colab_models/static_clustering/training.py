# ==============================================================================
# SEZIONE 0 — SETUP E CONFIGURAZIONE
# ==============================================================================
print("--- [SEZIONE 0] Inizio Setup e Configurazione ---")
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt
import sys
import os

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = (CURRENT_DIR / "../..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.feature_engineering import add_features_actuator_classification, final_features_actuator_classification
from colab_models.common import load_unified_dataset

SAVE_DIR = Path(__file__).parent / "output"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Directory di output pronta: {SAVE_DIR}")

BASE_PATH = Path("/content/drive/MyDrive/Tesi")
DATASET_COMPLETO_PATH = BASE_PATH / "DatasetSegmentato"
TEST_PERIODS_FILE = BASE_PATH / "test_periods.csv"

print("✅ [SEZIONE 0] Setup completato.")


# ==============================================================================
# SEZIONE 2 — CARICAMENTO DEL DATASET
# ==============================================================================
print("\n--- [SEZIONE 2] Caricamento Dati ---")

final_df = load_unified_dataset(DATASET_COMPLETO_PATH)
final_df['utc_datetime'] = pd.to_datetime(final_df['utc_datetime'])
if final_df.empty: raise SystemExit("DataFrame vuoto. Interrompo.")

print(f"Dataset completo caricato. Shape iniziale: {final_df.shape}")
print(final_df.head())
print("✅ [SEZIONE 2] Dati caricati.")


# ==============================================================================
# SEZIONE 3 — FEATURE ENGINEERING
# ==============================================================================
print("\n--- [SEZIONE 3] Feature Engineering ---")

final_df = add_features_actuator_classification(final_df)
print(f"✅ [SEZIONE 3] Completata. Shape: {final_df.shape}")


# ==============================================================================
# SEZIONE 4 — DEFINIZIONE FEATURE E DATI (TUTTI escluso TEST)
# ==============================================================================
print("\n--- [SEZIONE 4] Esclusione dei periodi di Test ---")

try:
    test_periods_df = pd.read_csv(TEST_PERIODS_FILE, parse_dates=['start_time', 'end_time'])
    print(f"Caricati {len(test_periods_df)} periodi di test da escludere.")

    test_periods_df['start_time'] = test_periods_df['start_time'].dt.tz_localize('UTC')
    test_periods_df['end_time'] = test_periods_df['end_time'].dt.tz_localize('UTC')

    # Inizializza una maschera booleana con tutti False
    exclusion_mask = pd.Series(False, index=final_df.index)

    for period in test_periods_df.itertuples():
        device = period.device
        start = period.start_time
        end = period.end_time

        period_mask = (
            (final_df['device'] == device) &
            (final_df['utc_datetime'] >= start) &
            (final_df['utc_datetime'] <= end)
        )
        exclusion_mask = exclusion_mask | period_mask
        print(f"Esclusi {period_mask.sum()} punti per il periodo {device} da {start.date()} a {end.date()}")

    data_for_clustering = final_df[~exclusion_mask].copy()

    print(f"\nRighe totali: {len(final_df)}")
    print(f"Righe escluse (test set): {exclusion_mask.sum()}")
    print(f"Righe rimanenti per il clustering: {len(data_for_clustering)}")

except FileNotFoundError:
    print(f"Avviso: File '{TEST_PERIODS_FILE}' non trovato. Uso l'intero dataset per il clustering.")
    data_for_clustering = final_df.copy()

features = [
    "temp_diff_in_out","ah_diff_in_out", "vpd_in","vpd_diff","temp_diff_x_wind",
    "temperature_external_trend_5m",
    "absolute_humidity_external_trend_5m",
    "temperature_sensor_trend_5m", "temperature_sensor_trend_30m","temperature_sensor_trend_60m",
    "absolute_humidity_sensor_trend_5m", "absolute_humidity_sensor_trend_30m", "absolute_humidity_sensor_trend_60m",
    "voc_trend_5m", "voc_trend_30m", "voc_trend_60m",
    "temperature_sensor_mean_5m", "temperature_sensor_mean_30m", "temperature_sensor_mean_60m",
    "absolute_humidity_sensor_mean_5m", "absolute_humidity_sensor_mean_30", "absolute_humidity_sensor_mean_60m",
    "voc_mean_5m", "voc_mean_30m", "voc_mean_60m",
    "temperature_sensor_std_5m", "temperature_sensor_std_30m", "temperature_sensor_std_60m",
    "absolute_humidity_sensor_std_5m", "absolute_humidity_sensor_std_30m", "absolute_humidity_sensor_std_60m",
    "voc_std_5m", "voc_std_30m", "voc_std_60m",
    "temperature_sensor_accel_5m",
    "absolute_humidity_sensor_accel_5m",
    "voc_accel_5m",
    "temperature_sensor_delta_3m", "temperature_sensor_delta_5m", "temperature_sensor_delta_10m",
    "absolute_humidity_sensor_delta_3m", "absolute_humidity_sensor_delta_5m", "absolute_humidity_sensor_delta_10m",
    "voc_delta_3m", "voc_delta_5m", "voc_delta_10m",
]
X_train = data_for_clustering[features]

print(f"\nFeatures utilizzate per il clustering: {len(features)}")
print(f"Numero totale di campioni per il clustering: {len(X_train)}")
print("✅ [SEZIONE 4] Dati pronti.")


# ==============================================================================
# SEZIONE 5 — RICERCA DEL NUMERO OTTIMALE DI CLUSTER (K)
# ==============================================================================
print("\n--- [SEZIONE 5] Ricerca del K ottimale con il Metodo del Gomito ---")

imputer = SimpleImputer(strategy="mean")
X_train_imp = imputer.fit_transform(X_train)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_imp)

inertias = []
k_range = range(5, 5)

for k in k_range:
    print(f"Calcolo per k={k}...")
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(X_train_s)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, "o-")
plt.xlabel("Numero di Cluster (k)")
plt.ylabel("Inerzia")
plt.title("Metodo del Gomito per la Scelta del K Ottimale")
plt.grid(True)
plt.xticks(k_range)
plt.savefig(SAVE_DIR / "elbow_method.png")
plt.show()

print("✅ [SEZIONE 5] Analisi del K completata. Ispeziona il grafico per scegliere il 'gomito'.")

# =================================================================================
# SEZIONE 5B — CONFERMA DEL K OTTIMALE CON IL SILHOUETTE SCORE
# =================================================================================
print("\n--- [SEZIONE 5B] Conferma del K con il Silhouette Score ---")
from sklearn.metrics import silhouette_score

silhouette_scores = []
X_sample = X_train_s[np.random.choice(X_train_s.shape[0], 10000, replace=False)]
for k in k_range:
    print(f"Calcolo per k={k}...")
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(X_sample) # Usa X_train_s o X_sample

    score = silhouette_score(X_sample, labels) # Usa X_train_s o X_sample
    silhouette_scores.append(score)

# Plot dei risultati
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, "o-")
plt.xlabel("Numero di Cluster (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score per la Scelta del K Ottimale")
plt.grid(True)
plt.xticks(k_range)
plt.savefig(SAVE_DIR / "silhouette_score.png")
plt.show()

# ==============================================================================
# SEZIONE 6 — ADDESTRAMENTO FINALE E SALVATAGGIO
# ==============================================================================
print("\n--- [SEZIONE 6] Addestramento Finale del Modello di Clustering ---")

# --- MODIFICA QUI ---
# Sulla base del grafico, scegli il valore di k che ritieni ottimale
OPTIMAL_K = 5
# --------------------
print(f"K ottimale scelto: {OPTIMAL_K}")

final_kmeans_model = KMeans(n_clusters=OPTIMAL_K, n_init='auto', random_state=42)
final_kmeans_model.fit(X_train_s)
print("Modello di clustering addestrato con successo.")

joblib.dump(final_kmeans_model, SAVE_DIR / "kmeans_model.joblib")
joblib.dump(scaler, SAVE_DIR / "scaler_clustering.joblib")
joblib.dump(imputer, SAVE_DIR / "imputer_clustering.joblib") # Salva anche l'imputer

with open(SAVE_DIR / "features_clustering.json", "w") as f: json.dump(features, f, indent=2)
metrics = {"optimal_k": OPTIMAL_K, "inertia": final_kmeans_model.inertia_}
with open(SAVE_DIR / "metrics_clustering.json", "w") as f: json.dump(metrics, f, indent=2)

print(f"✅ Modello, scaler, imputer e features salvati in: {SAVE_DIR}")


# ==============================================================================
# SEZIONE 7 — (OPZIONALE) ANALISI DEI CLUSTER
# ==============================================================================
print("\n--- [SEZIONE 7] Analisi Descrittiva dei Cluster Trovati ---")

data_for_clustering['cluster'] = final_kmeans_model.labels_
cluster_summary = data_for_clustering.groupby('cluster')[features].mean().round(2)

print("Valori medi delle feature per ogni cluster:")
print(cluster_summary)

cluster_summary.to_csv(SAVE_DIR / "cluster_summary.csv")
print(f"Sommario salvato in: {SAVE_DIR / 'cluster_summary.csv'}")

# ==============================================================================
# NUOVO BLOCCO: ANALISI DELLA VARIANZA PER LA FEATURE SELECTION
# ==============================================================================
print("\n--- [SEZIONE 7B] Analisi Varianza per Feature Selection ---")
from sklearn.preprocessing import MinMaxScaler

# 1. Normalizza i dati del sommario per un confronto equo
scaler_summary = MinMaxScaler()
summary_scaled = pd.DataFrame(scaler_summary.fit_transform(cluster_summary), columns=cluster_summary.columns)

# 2. Calcola la deviazione standard per ogni feature
feature_variance = summary_scaled.std(axis=0).sort_values(ascending=False)

print("\nFeature ordinate per importanza (varianza tra i cluster):")
print(feature_variance.head(30)) # Mostra le 30 più importanti

# 3. Seleziona le N feature migliori
N_TOP_FEATURES = 30 # Puoi cambiare questo valore
top_features = feature_variance.head(N_TOP_FEATURES).index.tolist()

print(f"\n✅ Le {N_TOP_FEATURES} feature più importanti sono state identificate.")
print("Per il prossimo ciclo, modifica la SEZIONE 4 per usare questa lista:")
print("\nfeatures = [")
for feature in top_features:
    print(f"    '{feature}',")
print("]")

print("\n✅ Script completato.")