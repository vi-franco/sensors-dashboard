import pandas as pd
import numpy as np
from utils.functions import vpd_kpa

def add_features_actuator_classification(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_cyclic(df)
    df = add_in_out_delta_features(df)
    df = add_rolling_features(df, ["temperature_sensor", "absolute_humidity_sensor", "co2", "voc", "temp_diff_in_out", "ah_diff_in_out", "dewpoint_diff_in_out"], win_short=5, win_long=30, extra_windows=(60, 180))
    df = add_external_trends(df, ["temperature_external", "absolute_humidity_external"])
    df = add_device_baselines(df, ["temperature_sensor", "absolute_humidity_sensor", "co2", "voc"])
    df = add_since_minutes(df)
    df = add_event_flags(df)
    return df

def ensure_min_columns_actuator_classification(df: pd.DataFrame):
    required_columns = [
        "device",
        "utc_datetime",
        "local_datetime",
        "lat",
        "lng",
        "temperature_sensor",
        "humidity_sensor",
        "co2",
        "voc",
        "temperature_external",
        "humidity_external",
        "ground_level_pressure",
        "wind_speed",
        "clouds_percentage",
        "rain_1h",
        "sunrise_time",
        "sunset_time",
        "absolute_humidity_sensor",
        "absolute_humidity_external",
        "dew_point_sensor",
        "dew_point_external"
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns for actuator classification: {', '.join(missing_columns)}")

def final_features_actuator_classification() -> list:
    return [
        # Base
        "temperature_sensor","absolute_humidity_sensor","co2","voc",
        "temperature_external","absolute_humidity_external",
        "ground_level_pressure","wind_speed","clouds_percentage","rain_1h",
        "dew_point_sensor","dew_point_external",

        # Gradienti/VPD/Interazioni
        "temp_diff_in_out","ah_diff_in_out","dewpoint_diff_in_out",
        "vpd_in","vpd_out","vpd_diff","temp_diff_x_wind",

        # Tempo locale
        "hour_sin","hour_cos","minutes_from_sunrise","minutes_to_sunset",

        # Rolling 5m/30m interni
        "temperature_sensor_accel_1m","temperature_sensor_trend_5m","temperature_sensor_trend_30m","temperature_sensor_mean_5m","temperature_sensor_mean_30m","temperature_sensor_std_5m","temperature_sensor_std_30m",
        "absolute_humidity_sensor_accel_1m", "absolute_humidity_sensor_trend_5m","absolute_humidity_sensor_trend_30m","absolute_humidity_sensor_mean_5m","absolute_humidity_sensor_mean_30m","absolute_humidity_sensor_std_5m","absolute_humidity_sensor_std_30m",
        "co2_accel_1m","co2_trend_5m","co2_trend_30m","co2_mean_5m","co2_mean_30m","co2_std_5m","co2_std_30m",
        "voc_accel_1m","voc_trend_5m","voc_trend_30m","voc_mean_5m","voc_mean_30m","voc_std_5m","voc_std_30m",

        # Trend esterni
        "temperature_external_trend_5m","temperature_external_trend_30m",
        "absolute_humidity_external_trend_5m","absolute_humidity_external_trend_30m",

        # Baseline delta
        "co2_baseline_delta","voc_baseline_delta",
        "temperature_sensor_baseline_delta","absolute_humidity_sensor_baseline_delta",

        # Flag eventi
        "co2_drop_flag_5m","voc_drop_flag_5m","temp_drop_flag_5m","ah_rise_flag_5m",

        # Since minutes per attuatore
        "since_minutes_Umidificatore",
        "since_minutes_Finestra",
        "since_minutes_Deumidificatore",
        "since_minutes_Riscaldamento",
        "since_minutes_Clima",

        # Rolling lunghi (60m, 180m)
        "temperature_sensor_trend_60m", "temperature_sensor_trend_180m",
        "temperature_sensor_mean_60m",  "temperature_sensor_mean_180m",
        "temperature_sensor_std_60m",   "temperature_sensor_std_180m",

        "absolute_humidity_sensor_trend_60m", "absolute_humidity_sensor_trend_180m",
        "absolute_humidity_sensor_mean_60m",  "absolute_humidity_sensor_mean_180m",
        "absolute_humidity_sensor_std_60m",   "absolute_humidity_sensor_std_180m",

        "co2_trend_60m", "co2_trend_180m",
        "co2_mean_60m",  "co2_mean_180m",
        "co2_std_60m",   "co2_std_180m",

        "voc_trend_60m", "voc_trend_180m",
        "voc_mean_60m",  "voc_mean_180m",
        "voc_std_60m",   "voc_std_180m",

        "temp_diff_in_out_trend_60m", "temp_diff_in_out_trend_180m",
        "temp_diff_in_out_mean_60m",  "temp_diff_in_out_mean_180m",
        "temp_diff_in_out_std_60m",   "temp_diff_in_out_std_180m",

        "ah_diff_in_out_trend_60m", "ah_diff_in_out_trend_180m",
        "ah_diff_in_out_mean_60m",  "ah_diff_in_out_mean_180m",
        "ah_diff_in_out_std_60m",   "ah_diff_in_out_std_180m",

        "dewpoint_diff_in_out_trend_60m", "dewpoint_diff_in_out_trend_180m",
        "dewpoint_diff_in_out_mean_60m",  "dewpoint_diff_in_out_mean_180m",
        "dewpoint_diff_in_out_std_60m",   "dewpoint_diff_in_out_std_180m",
    ]

def add_time_cyclic(df: pd.DataFrame) -> pd.DataFrame:
    dt_local = pd.to_datetime(df["local_datetime"], errors="coerce")
    dt_utc = pd.to_datetime(df["utc_datetime"], errors="coerce", utc=True)

    hour = dt_local.dt.hour.fillna(0).astype(int)
    minute = dt_local.dt.minute.fillna(0).astype(int)
    hour_frac = (hour + minute/60.0) % 24

    df["hour_sin"] = np.sin(2*np.pi*hour_frac/24)
    df["hour_cos"] = np.cos(2*np.pi*hour_frac/24)

    sr_utc = pd.to_datetime(df.get("sunrise_time"), errors="coerce", utc=True)
    ss_utc = pd.to_datetime(df.get("sunset_time"),  errors="coerce", utc=True)

    df["minutes_from_sunrise"] = ((dt_utc - sr_utc).dt.total_seconds() / 60).fillna(0)
    df["minutes_to_sunset"]    = ((ss_utc - dt_utc).dt.total_seconds() / 60).fillna(0)

    return df


def add_in_out_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    df["temp_diff_in_out"] = df["temperature_sensor"] - df["temperature_external"]
    df["ah_diff_in_out"]   = df["absolute_humidity_sensor"] - df["absolute_humidity_external"]
    df["dewpoint_diff_in_out"] = df["dew_point_sensor"] - df["dew_point_external"]
    df["vpd_in"]  = vpd_kpa(df["temperature_sensor"],   df["humidity_sensor"])
    df["vpd_out"] = vpd_kpa(df["temperature_external"], df["humidity_external"])
    df["vpd_diff"] = df["vpd_in"] - df["vpd_out"]
    df["temp_diff_x_wind"] = df["temp_diff_in_out"] * df["wind_speed"].fillna(0)
    return df

def add_since_minutes(df: pd.DataFrame,
                      actuator_state_cols: list | None = None,
                      initial_since_value: float = 0.0) -> pd.DataFrame:
    df = df.copy()
    df["_dt_utc_parsed"] = pd.to_datetime(df["utc_datetime"], errors="coerce", utc=True)
    df = df.sort_values(["device", "_dt_utc_parsed"])

    if actuator_state_cols is None:
        actuator_state_cols = [c for c in df.columns if c.startswith("state_")]

    for scol in actuator_state_cols:
        suffix = scol.split("state_", 1)[1] if "state_" in scol else scol
        out_col = f"since_minutes_{suffix}"

        def _since_per_group(g: pd.DataFrame) -> pd.Series:
            s = g[scol].astype("float").round().fillna(method="ffill").fillna(0).astype(int)
            change_mask = s.ne(s.shift(1, fill_value=s.iloc[0]))
            change_time = g["_dt_utc_parsed"].where(change_mask)
            last_change_time = change_time.ffill()
            first_time = g["_dt_utc_parsed"].iloc[0]
            last_change_time = last_change_time.fillna(first_time)
            delta_min = (g["_dt_utc_parsed"] - last_change_time).dt.total_seconds() / 60.0
            if initial_since_value is not None and len(delta_min) > 0:
                delta_min.iloc[0] = initial_since_value
            return delta_min

        df[out_col] = df.groupby("device", group_keys=False).apply(_since_per_group)

    df.drop(columns=["_dt_utc_parsed"], inplace=True, errors="ignore")
    return df

def add_rolling_features(df: pd.DataFrame, cols: list,
                         win_short: int = 5, win_long: int = 30,
                         extra_windows: tuple | list = ()) -> pd.DataFrame:
    """
    Retro-compatibile:
    - Mantiene le colonne già esistenti per 5m/30m con stessi nomi.
    - Aggiunge opzionalmente finestre extra (es. 60, 180) con i relativi suffissi.
    - L'accelerazione resta sul window "breve" come prima (_accel_1m).
    """
    df = df.sort_values(["device", "utc_datetime"])
    all_windows = [win_short, win_long] + [w for w in extra_windows if w not in (win_short, win_long)]

    for c in cols:
        g = df.groupby("device")[c]

        # trend per tutte le finestre richieste
        for w in all_windows:
            df[f"{c}_trend_{w}m"] = g.diff(w)

        # mean/std per tutte le finestre richieste
        for w in all_windows:
            roll = g.rolling(w, min_periods=max(2, w // 3))
            df[f"{c}_mean_{w}m"] = roll.mean().reset_index(level=0, drop=True)
            df[f"{c}_std_{w}m"]  = roll.std().reset_index(level=0, drop=True)

        # accelerazione causale (come avevi già: dal trend breve)
        df[f"{c}_accel_1m"] = df.groupby("device")[f"{c}_trend_{win_short}m"].diff()
    return df

# wrapper opzionale comodo
def add_long_rolling_features(df: pd.DataFrame, cols: list, long_windows: tuple = (60, 180)) -> pd.DataFrame:
    return add_rolling_features(df, cols, win_short=5, win_long=30, extra_windows=long_windows)

def add_external_trends(df: pd.DataFrame, cols: list, win_short: int = 5, win_long: int = 30) -> pd.DataFrame:
    for c in cols:
        g = df.groupby("device")[c]
        df[f"{c}_trend_{win_short}m"] = g.diff(win_short)
        df[f"{c}_trend_{win_long}m"]  = g.diff(win_long)
    return df


def add_device_baselines(df: pd.DataFrame, cols: list, window_minutes: int = 1440) -> pd.DataFrame:
    for c in cols:
        base = df.groupby("device")[c].rolling(window_minutes, min_periods=60).median().reset_index(level=0, drop=True)
        df[f"{c}_baseline_delta"] = df[c] - base
    return df


def add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    # flag data-driven: drop/rise quando la pendenza 5m supera la deviazione a 30m
    eps = 1e-6
    def drop_flag(var):
        return (df[f"{var}_trend_5m"] < -(df[f"{var}_std_30m"].fillna(0)+eps)).astype(int)
    def rise_flag(var):
        return (df[f"{var}_trend_5m"] >  (df[f"{var}_std_30m"].fillna(0)+eps)).astype(int)

    df["co2_drop_flag_5m"]  = drop_flag("co2")
    df["voc_drop_flag_5m"]  = drop_flag("voc")
    df["temp_drop_flag_5m"] = drop_flag("temperature_sensor")
    df["ah_rise_flag_5m"]   = rise_flag("absolute_humidity_sensor")
    return df
