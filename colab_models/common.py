import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

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
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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