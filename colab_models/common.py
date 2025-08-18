import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import timedelta
from typing import List, Optional
import tensorflow as tf
import tensorflow.keras.backend as K

def load_unified_dataset(folder_path: Path) -> pd.DataFrame:
    if not folder_path.is_dir():
        print(f" -> Cartella inesistente: {folder_path}")
        return pd.DataFrame()
    frames = []
    for fp in folder_path.glob("*.csv"):
        try:
            df_single = pd.read_csv(fp, on_bad_lines="skip")
            df_single['period_id'] = fp.stem
            frames.append(df_single)
        except Exception as e:
            print(f" -> Errore lettura {fp.name}: {e}")

    final_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    final_df["utc_datetime"] = pd.to_datetime(final_df["utc_datetime"], errors="coerce", utc=True)
    final_df.dropna(subset=["utc_datetime", "device"], inplace=True)
    final_df.sort_values(["device", "period_id", "utc_datetime"], inplace=True)
    return final_df


def get_data_from_periods(df: pd.DataFrame, periods_file: Path) -> pd.DataFrame:
    if not periods_file.exists():
        return pd.DataFrame()
    periods = pd.read_csv(periods_file)
    if periods.empty:
        return pd.DataFrame()
    idx = []
    for _, r in periods.iterrows():
        start = pd.to_datetime(r["start_time"], utc=True)
        end   = pd.to_datetime(r["end_time"],   utc=True)
        mask = (df["device"] == r["device"]) & (df["utc_datetime"] >= start) & (df["utc_datetime"] <= end)
        idx.extend(df[mask].index)
    return df.loc[idx].copy()

def get_actuator_names():
    return ["Umidificatore", "Finestra", "Deumidificatore", "Riscaldamento", "Clima"]


def log_actuator_stats(df: pd.DataFrame, STATE_COLS: list[str], name: str) -> None:
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



def augment_specific_groups_with_noise(
    df: pd.DataFrame,
    n_duplicates: int,
    target_actuators: List[str],
    group_col: str = "period_id",
    noise_min: float = 0.005,
    noise_max: float = 0.01,
    id_suffix: str = "AUG",
    exclude_cols: Optional[List[str]] = None
):
    """
    Duplica e aggiunge rumore SOLO ai gruppi che contengono almeno un'attivazione
    (valore = 1) in una delle colonne specificate in target_actuators.

    Args:
        df (pd.DataFrame): Il DataFrame di input (tipicamente il train_fold_original).
        n_duplicates (int): Il numero di volte che ogni gruppo idoneo deve essere duplicato.
        target_actuators (List[str]): Lista di colonne "state" da controllare.
                                      Un gruppo viene aumentato se almeno una di queste
                                      colonne ha valore 1 in almeno una riga.
        ... (altri parametri come prima)
    """
    if n_duplicates <= 0 or not target_actuators:
        return pd.DataFrame(columns=df.columns)

    # Assicura che tutte le colonne target esistano nel DataFrame per evitare errori
    for act in target_actuators:
        if act not in df.columns:
            raise ValueError(f"La colonna target '{act}' non è presente nel DataFrame.")

    if exclude_cols is None: exclude_cols = []
    cols_to_exclude_from_noise = set(exclude_cols) | {group_col}
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in cols_to_exclude_from_noise]

    augmented_dfs = []

    for original_id, group_df in df.groupby(group_col):
        if (group_df[target_actuators] == 1).any().any():

            # Se la condizione è vera, procedi con l'aumento come prima
            for i in range(n_duplicates):
                new_group = group_df.copy()
                for col in numeric_cols:
                    noise_size = len(new_group)
                    random_noise = np.random.uniform(noise_min, noise_max, size=noise_size)
                    random_sign = np.random.choice([-1, 1], size=noise_size)
                    factor = 1 + (random_noise * random_sign)
                    new_group[col] = new_group[col].astype(float) * factor

                new_id = f"{original_id}_{id_suffix}__{i + 1}"
                new_group[group_col] = new_id
                augmented_dfs.append(new_group)

        # Se la condizione è falsa, il ciclo salta questo gruppo e passa al successivo.

    if not augmented_dfs:
        return pd.DataFrame(columns=df.columns)

    return pd.concat(augmented_dfs, ignore_index=True)

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementazione della Focal Loss.
    gamma: parametro di focalizzazione. Valori più alti danno più peso agli errori.
    alpha: parametro di bilanciamento per le classi.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip per evitare log(0)
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed