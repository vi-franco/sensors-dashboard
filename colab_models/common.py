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
    # Assicura datetime
    if not np.issubdtype(train_base[time_col].dtype, np.datetime64):
        train_base = train_base.copy()
        train_base[time_col] = pd.to_datetime(train_base[time_col], utc=True, errors="coerce")

    # Colonne numeriche da “rumorizzare”
    exclude = set([time_col, group_col]) | set(state_cols)
    num_cols = [c for c in train_base.columns
                if c not in exclude and np.issubdtype(train_base[c].dtype, np.number)]

    def segments_on(g: pd.DataFrame, state_col: str):
        g = g.sort_values(time_col)
        on = g[state_col].astype(int).to_numpy()
        if len(on) == 0:
            return []
        starts = np.where((on == 1) & (np.roll(on, 1) == 0))[0]
        ends   = np.where((on == 1) & (np.roll(on, -1) == 0))[0]
        if on[0] == 1 and (len(starts) == 0 or starts[0] != 0):
            starts = np.insert(starts, 0, 0)
        if on[-1] == 1 and (len(ends) == 0 or ends[-1] != len(on) - 1):
            ends = np.append(ends, len(on) - 1)
        segs = []
        for s, e in zip(starts, ends):
            t0, t1 = g.iloc[s][time_col], g.iloc[e][time_col]
            if pd.isna(t0) or pd.isna(t1):
                continue
            segs.append((pd.Timestamp(t0), pd.Timestamp(t1)))
        return segs

    augmented = []
    acts = [c.removeprefix("state_") for c in state_cols]

    for act in acts:
        state_col = f"state_{act}"
        pos_now = int(train_base[state_col].sum())
        if pos_now >= min_pos_rows_per_act:
            continue

        # Trova segmenti ON per ogni periodo dentro il TRAIN già filtrato
        seg_meta = []  # (pid, t0c, t1c, pos_in_seg)
        for pid, g in train_base.groupby(group_col, sort=False):
            segs = segments_on(g, state_col)
            for t0, t1 in segs:
                t0c = t0 - timedelta(minutes=context_minutes)
                t1c = t1 + timedelta(minutes=context_minutes)
                seg_df = g[(g[time_col] >= t0c) & (g[time_col] <= t1c)]
                if seg_df.empty:
                    continue
                pos_in_seg = int(seg_df[state_col].sum())
                if pos_in_seg == 0:
                    continue
                seg_meta.append((pid, t0c, t1c, pos_in_seg))

        if not seg_meta:
            continue

        need = max(0, min_pos_rows_per_act - pos_now)
        k = 0
        while need > 0:
            pid, t0c, t1c, pos_in_seg = seg_meta[k % len(seg_meta)]
            g = train_base[train_base[group_col] == pid]
            seg_df = g[(g[time_col] >= t0c) & (g[time_col] <= t1c)].copy()
            if seg_df.empty:
                k += 1
                continue

            # nuovo period_id per i duplicati
            seg_df[group_col] = f"{pid}__{act}_{id_suffix}_{k}"

            # Rumore ±noise_pct
            if noise_pct and noise_pct > 0 and num_cols:
                # rumore indipendente per cella
                noise = (np.random.rand(len(seg_df), len(num_cols)) * 2 - 1) * noise_pct
                seg_df.loc[:, num_cols] = seg_df.loc[:, num_cols].to_numpy(dtype=float) * (1.0 + noise)

            augmented.append(seg_df)
            need -= pos_in_seg
            k += 1

    if not augmented:
        return pd.DataFrame(columns=train_base.columns)

    aug_df = pd.concat(augmented, ignore_index=True)
    return aug_df.sort_values(time_col).reset_index(drop=True)