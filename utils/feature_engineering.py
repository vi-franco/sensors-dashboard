import pandas as pd
import numpy as np
from utils.functions import vpd_kpa

def add_features_actuator_classification(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_cyclic(df)
    df = add_in_out_delta_features(df)
    df = add_rolling_features(df, ["temperature_sensor", "absolute_humidity_sensor", "co2", "voc", "temp_diff_in_out", "ah_diff_in_out", "dewpoint_diff_in_out"])
    df = add_external_trends(df, ["temperature_external", "absolute_humidity_external"])
    df = add_event_flags(df)
    return df

def ensure_min_columns_actuator_classification(df: pd.DataFrame):
    required_columns = [
        "device",
        "group_by"
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
        "temperature_sensor_accel_1m","temperature_sensor_trend_5m","temperature_sensor_trend_30m","temperature_sensor_mean_30m","temperature_sensor_std_30m",
        "absolute_humidity_sensor_accel_1m", "absolute_humidity_sensor_trend_5m","absolute_humidity_sensor_trend_30m","absolute_humidity_sensor_mean_30m","absolute_humidity_sensor_std_30m",
        "co2_accel_1m","co2_trend_5m","co2_trend_30m","co2_mean_30m","co2_std_30m",
        "voc_accel_1m","voc_trend_5m","voc_trend_30m","voc_mean_30m","voc_std_30m",

        # Trend esterni
        "temperature_external_trend_5m","temperature_external_trend_30m",
        "absolute_humidity_external_trend_5m","absolute_humidity_external_trend_30m",

        # Flag eventi
        "co2_drop_flag_5m","voc_drop_flag_5m","temp_drop_flag_5m","temp_rise_flag_5m", "ah_drop_flag_5m","ah_rise_flag_5m",

        "temperature_sensor_mean_60m",
        "temperature_sensor_std_60m",

        "absolute_humidity_sensor_mean_60m",
        "absolute_humidity_sensor_std_60m",

        "co2_mean_60m",
        "co2_std_60m",

        "voc_mean_60m",
        "voc_std_60m",

        "temp_diff_in_out_mean_60m",
        "temp_diff_in_out_std_60m",

        "ah_diff_in_out_mean_60m",
        "ah_diff_in_out_std_60m",

        "dewpoint_diff_in_out_mean_60m",
        "dewpoint_diff_in_out_std_60m",
    ]

def add_time_cyclic(df: pd.DataFrame) -> pd.DataFrame:
    dt_local = pd.to_datetime(df["local_datetime"], errors="coerce")
    dt_utc = pd.to_datetime(df["utc_datetime"], errors="coerce", utc=True)
    hour = dt_local.dt.hour.fillna(0).astype(int)
    minute = dt_local.dt.minute.fillna(0).astype(int)
    hour_frac = (hour + minute/60.0) % 24
    df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)
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

def add_rolling_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    feature_config = {
        'trend': [5, 15, 30],
        'mean': [15, 30, 60],
        'std': [15, 30, 60],
        'accel': 5
    }
    df = df.sort_values(by=["device", "period_id", "utc_datetime"])
    grouper = df.groupby(["device", "period_id"])
    for c in cols:
        g = grouper[c]
        if 'mean' in feature_config:
            for w in feature_config['mean']:
                roll = g.rolling(w, min_periods=max(2, w // 3))
                df[f"{c}_mean_{w}m"] = roll.mean().reset_index(level=0, drop=True)

        if 'std' in feature_config:
            for w in feature_config['std']:
                roll = g.rolling(w, min_periods=max(2, w // 3))
                df[f"{c}_std_{w}m"] = roll.std().reset_index(level=0, drop=True)

        if 'trend' in feature_config:
            for w in feature_config['trend']:
                df[f"{c}_trend_{w}m"] = g.diff(w)

        if 'accel' in feature_config:
            base_window = feature_config['accel']
            trend_col = f"{c}_trend_{base_window}m"

            if trend_col not in df:
                df[trend_col] = g.diff(base_window)

            df[f"{c}_accel_1m"] = df.groupby("device")[trend_col].diff()
    return df

def add_external_trends(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    win_short = 5
    win_long = 30
    for c in cols:
        g = df.groupby("device")[c]
        df[f"{c}_trend_{win_short}m"] = g.diff(win_short)
        df[f"{c}_trend_{win_long}m"]  = g.diff(win_long)
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
    df["temp_rise_flag_5m"] = rise_flag("temperature_sensor")
    df["ah_drop_flag_5m"]   = drop_flag("absolute_humidity_sensor")
    df["ah_rise_flag_5m"]   = rise_flag("absolute_humidity_sensor")
    return df
