import pandas as pd
import numpy as np
from utils.functions import vpd_kpa

def add_features_baseline_prediction(df: pd.DataFrame) -> pd.DataFrame:
    ensure_min_columns_baseline_prediction(df)
    return add_features_actuator_classification(df)

def get_rolling_features() -> list:
    return ["temperature_sensor", "absolute_humidity_sensor", "co2", "voc", "temp_diff_in_out", "ah_diff_in_out", "dewpoint_diff_in_out"]

def add_features_actuator_classification(df: pd.DataFrame) -> pd.DataFrame:
    rolling_features = get_rolling_features()
    ensure_min_columns_actuator_classification(df)
    df = add_time_cyclic(df)
    df = add_in_out_delta_features(df)
    df = add_rolling_features(df, rolling_features)
    df = add_external_trends(df, ["temperature_external", "absolute_humidity_external"])
    df = add_event_flags(df)
    return df


def add_targets_baseline_prediction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    columns_to_predict = ["temperature_sensor", "absolute_humidity_sensor", "co2", "voc"]
    horizons = [15, 30, 60]

    df.sort_values(by=["device", "period_id", "utc_datetime"], inplace=True)

    for col in columns_to_predict:
        if col not in df.columns: continue
        for h in horizons:
            # Colonna per la valutazione (valore assoluto futuro)
            eval_name = f"{col}_eval_{h}m"
            df[eval_name] = df.groupby(["device", "period_id"])[col].shift(-h)

            # Colonna target per il training (il delta, cioè la differenza)
            target_name = f"{col}_pred_{h}m"
            df[target_name] = df[eval_name] - df[col]

    return df


def ensure_min_columns_baseline_prediction(df: pd.DataFrame):
    return ensure_min_columns_actuator_classification(df)

def ensure_min_columns_actuator_classification(df: pd.DataFrame):
    required_columns = [
        "device",
        "period_id",
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

def final_features_baseline_prediction() -> list:
    return final_features_actuator_classification()

def final_features_actuator_classification() -> list:
    features = [
        # Base
        "temperature_sensor","absolute_humidity_sensor","co2","voc",
        "temperature_external","absolute_humidity_external",
        "ground_level_pressure","wind_speed","clouds_percentage","rain_1h",

        # Gradienti/VPD/Interazioni
        "temp_diff_in_out","ah_diff_in_out","dewpoint_diff_in_out",
        "vpd_diff","temp_diff_x_wind",

        # Tempo locale
        "hour_sin","hour_cos","minutes_from_sunrise","minutes_to_sunset",

        # Trend esterni
        "temperature_external_trend_5m","temperature_external_trend_30m",
        "absolute_humidity_external_trend_5m","absolute_humidity_external_trend_30m",

        # Flag eventi
        "co2_drop_flag_5m","voc_drop_flag_5m","temp_drop_flag_5m","temp_rise_flag_5m", "ah_drop_flag_5m","ah_rise_flag_5m",
    ]

    rolling_features = get_rolling_features()
    feature_config = get_rolling_features_config()
    generated_features = []

    for stat, windows in feature_config.items():
        if isinstance(windows, int):
            windows = [windows]
        for window in windows:
            for feature in rolling_features:
                generated_features.append(f"{feature}_{stat}_{window}m")

    features.extend(generated_features)
    return features

def final_features_actuator_classification_lstm() -> list:
    features = [
        # Base
        "temperature_sensor","absolute_humidity_sensor","co2","voc",
        "temperature_external","absolute_humidity_external",
        "wind_speed","rain_1h",

        # Gradienti/VPD/Interazioni
        "temp_diff_in_out","ah_diff_in_out", "vpd_in","vpd_diff","temp_diff_x_wind",

        # Tempo locale
        "hour_sin","hour_cos","minutes_from_sunrise","minutes_to_sunset",

        # Trend esterni
        "temperature_external_trend_5m",
        "absolute_humidity_external_trend_5m"
    ]

    rolling_features = get_rolling_features()
    feature_config = get_rolling_features_config()
    generated_features = []

    for stat, windows in feature_config.items():
        if isinstance(windows, int):
            windows = [windows]
        for window in windows:
            for feature in rolling_features:
                generated_features.append(f"{feature}_{stat}_{window}m")

    features.extend(generated_features)
    return features

def add_time_cyclic(df: pd.DataFrame) -> pd.DataFrame:
    dt_local = pd.to_datetime(df["local_datetime"], errors="coerce")
    dt_utc = pd.to_datetime(df["utc_datetime"], errors="coerce", utc=True)
    hour = dt_local.dt.hour.fillna(0).astype(int)
    minute = dt_local.dt.minute.fillna(0).astype(int)
    hour_frac = (hour + minute/60.0) % 24
    df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)
    sr_utc = pd.to_datetime(df["sunrise_time"], unit="s", errors="coerce", utc=True)
    ss_utc = pd.to_datetime(df["sunset_time"], unit="s", errors="coerce", utc=True)
    if sr_utc.notna().any() and ss_utc.notna().any():
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

def get_rolling_features_config() -> dict:
    return {
       'trend': [5, 60],
       'mean': [5, 60],
       'std': [5, 60],
       'accel': 5
   }

def add_rolling_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    feature_config = get_rolling_features_config()
    df = df.sort_values(by=["device", "period_id", "utc_datetime"])
    grouper = df.groupby(["device", "period_id"])
    for c in cols:
        g = grouper[c]
        if 'mean' in feature_config:
            for w in feature_config['mean']:
                roll = g.rolling(w, min_periods=max(2, w // 3))
                df[f"{c}_mean_{w}m"] = roll.mean().reset_index(level=[0, 1], drop=True)

        if 'std' in feature_config:
            for w in feature_config['std']:
                roll = g.rolling(w, min_periods=max(2, w // 3))
                df[f"{c}_std_{w}m"] = roll.std().reset_index(level=[0, 1], drop=True)

        if 'trend' in feature_config:
            for w in feature_config['trend']:
                df[f"{c}_trend_{w}m"] = g.diff(w)

        if 'accel' in feature_config:
            base_window = feature_config['accel']
            trend_col = f"{c}_trend_{base_window}m"

            if trend_col not in df:
                df[trend_col] = g.diff(base_window)

            df[f"{c}_accel_{base_window}m"] = df.groupby(["device", "period_id"])[trend_col].diff()
    return df

def add_external_trends(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        g = df.groupby("device")[c]
        df[f"{c}_trend_5m"] = g.diff(5)
        df[f"{c}_trend_30m"]  = g.diff(30)
    return df


def add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    # flag data-driven: drop/rise quando la pendenza 5m supera la deviazione a 30m
    eps = 1e-6
    def drop_flag(var):
        return (df[f"{var}_trend_5m"] < -(df[f"{var}_std_60m"].fillna(0)+eps)).astype(int)
    def rise_flag(var):
        return (df[f"{var}_trend_5m"] >  (df[f"{var}_std_60m"].fillna(0)+eps)).astype(int)

    df["co2_drop_flag_5m"]  = drop_flag("co2")
    df["voc_drop_flag_5m"]  = drop_flag("voc")
    df["temp_drop_flag_5m"] = drop_flag("temperature_sensor")
    df["temp_rise_flag_5m"] = rise_flag("temperature_sensor")
    df["ah_drop_flag_5m"]   = drop_flag("absolute_humidity_sensor")
    df["ah_rise_flag_5m"]   = rise_flag("absolute_humidity_sensor")
    return df

def build_sequences(df, feature_cols, target_cols, window=180, stride=1, time_col="utc_datetime", group_cols=("device", "period_id")):
    """
    Genera finestre causali di lunghezza `window`.
    Ogni finestra: [t-window+1 ... t] -> target a t.
    Ritorna:
        X: array [N, window, F]
        y: array [N, C]
        times: array con i timestamp finali di ogni finestra
    """
    if any(c not in df.columns for c in feature_cols + target_cols):
        raise ValueError("Mancano alcune colonne richieste")

    X_list, y_list, t_list = [], [], []

    df = df.sort_values(list(group_cols) + [time_col]).reset_index(drop=True)

    for _, g in df.groupby(list(group_cols), sort=False):
        if len(g) < window:
            continue

        F = g[feature_cols].to_numpy(dtype=np.float32, copy=False)
        Y = g[target_cols].to_numpy(dtype=np.int8,   copy=False)
        times = pd.to_datetime(g[time_col]).values

        for end in range(window - 1, len(g), stride):
            start = end - window + 1
            X_list.append(F[start:end+1])
            y_list.append(Y[end])
            t_list.append(times[end])

    if not X_list:
        raise ValueError("Nessuna finestra creata, controlla window/stride.")

    return (np.asarray(X_list, dtype=np.float32),
                np.asarray(y_list, dtype=np.int8),
                np.asarray(t_list))