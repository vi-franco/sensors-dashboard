import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import timedelta
from typing import List, Optional

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



def augment_minority_periods_on_windows(
    train_base: pd.DataFrame,
    state_cols: List[str],
    time_col: str = "utc_datetime",
    group_col: str = "period_id",
    min_pos_rows_per_act: int = 5000,
    context_minutes: int = 30,
    noise_pct: float = 0.01,
    id_suffix: str = "AUG",
) -> pd.DataFrame:

    base = train_base.copy()
    base[time_col] = pd.to_datetime(base[time_col], utc=True, errors="coerce")

    # Colonne numeriche da perturbare (escludi target + id/tempo)
    exclude = set(state_cols) | {time_col, group_col}
    num_cols = [c for c in base.columns if c not in exclude and np.issubdtype(base[c].dtype, np.number)]

    # Trova segmenti contigui ON per una colonna stato in un gruppo
    def segments_on(g: pd.DataFrame, state_col: str):
        g = g.sort_values(time_col)
        on = g[state_col].astype(int).to_numpy()
        if on.size == 0:
            return []
        d = np.diff(np.r_[0, on, 0])
        starts = np.where(d == 1)[0]
        ends   = np.where(d == -1)[0] - 1
        return [(g.iloc[s][time_col], g.iloc[e][time_col]) for s, e in zip(starts, ends)]

    augmented = []
    acts = [c[6:] if c.startswith("state_") else c for c in state_cols]

    for act in acts:
        state_col = f"state_{act}"
        pos_now = int(base[state_col].sum())
        if pos_now >= min_pos_rows_per_act:
            continue

        # Colleziona segmenti ON (+/- contesto) disponibili
        meta = []  # (pid, t0c, t1c, pos_in_seg)
        for pid, g in base.groupby(group_col, sort=False):
            for t0, t1 in segments_on(g, state_col):
                t0c = t0 - timedelta(minutes=context_minutes)
                t1c = t1 + timedelta(minutes=context_minutes)
                seg = g[(g[time_col] >= t0c) & (g[time_col] <= t1c)]
                if seg.empty:
                    continue
                pos_in_seg = int(seg[state_col].sum())
                if pos_in_seg == 0:
                    continue
                meta.append((pid, t0c, t1c, pos_in_seg))

        if not meta:
            continue

        need = max(0, min_pos_rows_per_act - pos_now)
        k = 0
        while need > 0:
            pid, t0c, t1c, pos_in_seg = meta[k % len(meta)]
            g = base[base[group_col] == pid]
            seg = g[(g[time_col] >= t0c) & (g[time_col] <= t1c)].copy()
            if seg.empty:
                k += 1
                continue

            # Nuovo period_id per i duplicati
            seg[group_col] = f"{pid}__{act}_{id_suffix}_{k}"

            # Rumore ±noise_pct (moltiplicativo) sulle numeriche
            if num_cols and noise_pct:
                for c in num_cols:
                    n = np.random.uniform(-noise_pct, noise_pct, size=len(seg))
                    seg[c] = seg[c].astype(float) * (1.0 + n)

            augmented.append(seg)
            need -= pos_in_seg
            k += 1

    if not augmented:
        return base.iloc[0:0].copy()  # DataFrame vuoto con stesse colonne

    aug_df = pd.concat(augmented, ignore_index=True)
    return aug_df.sort_values(time_col).reset_index(drop=True)