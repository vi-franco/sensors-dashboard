import pandas as pd
import numpy as np

def vpd_kpa(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    # Tetens formula (T in °C) → kPa
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    ea = (rh_pct.clip(0,100) / 100.0) * es
    return (es - ea).astype(float)

def round_to_exact_minute(df):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index = df.index.floor("min")
    df = df[~df.index.duplicated(keep="last")]
    return df