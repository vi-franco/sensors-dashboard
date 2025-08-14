# utils.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import json
import time
import sqlite3
import requests
from influxdb import InfluxDBClient
from datetime import datetime, timezone, timedelta
import config
import pandas as pd

# ==============================================================================
# --- DATA FETCHING FUNCTIONS ---
# ==============================================================================

def get_external_weather_df(lat: float, lon: float) -> pd.DataFrame:
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            query = "SELECT * FROM external_weather WHERE lat = ? AND lng = ? ORDER BY dt;"
            df = pd.read_sql_query(query, conn, params=(lat, lon))
    except sqlite3.Error as e:
        print(f"Errore DB: {e}", file=sys.stderr)
        return pd.DataFrame()

    if df.empty:
        print(f"Nessun dato meteo trovato nel DB per ({lat}, {lon}).")
        return df

    rename_map = {
        'temperature': 'temperature_external', 'humidity': 'humidity_external',
        'pressure': 'ground_level_pressure', 'timezone_offset': 'timezone'
    }
    df = df.rename(columns=rename_map)

    df['dew_point_external'] = df.apply(lambda r: calculate_dew_point(r.get('temperature_external'), r.get('humidity_external')), axis=1)
    df['absolute_humidity_external'] = df.apply(lambda r: calculate_absolute_humidity(r.get('temperature_external'), r.get('humidity_external')), axis=1)
    df['sunrise_time'] = pd.to_datetime(df['sunrise_time'], unit='s', utc=True)
    df['sunset_time'] = pd.to_datetime(df['sunset_time'], unit='s', utc=True)
    df['local_datetime'] = pd.to_datetime(df['dt'] + df['timezone'], unit='s', utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    df['utc_datetime'] = pd.to_datetime(df['dt'], unit='s', utc=True)
    df = df.set_index('utc_datetime')

    now_utc = pd.Timestamp.now(tz='UTC')
    df.loc[now_utc] = df.iloc[-1].copy()

    numeric_cols = df.select_dtypes(include=np.number).columns
    df_num = df[numeric_cols].resample('min').mean().interpolate(method='linear')

    non_numeric_cols = df.columns.difference(numeric_cols)
    df_non_num = df[non_numeric_cols].resample('min').ffill()

    df_final = df_num.join(df_non_num)
    df_final['utc_datetime'] = df_final.index

    final_columns = [
        "temperature_external", "humidity_external", "ground_level_pressure",
        "wind_speed", "clouds_percentage", "rain_1h", "lat", "lng",
        "sunrise_time", "sunset_time", "local_datetime", "dt", "timezone",
        "absolute_humidity_external", "dew_point_external", "utc_datetime"
    ]

    return df_final[[col for col in final_columns if col in df_final.columns]]

def get_sensor_history(device_id: str, status: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    try:
        client = InfluxDBClient(
            host=config.INFLUXDB_HOST,
            port=config.INFLUXDB_PORT,
            database=config.INFLUXDB_DATABASE
        )
        query = f'SELECT * FROM "{config.INFLUXDB_MEASUREMENT}" WHERE "device" = \'{device_id}\' AND time >= now() - {config.HISTORY_MINUTES}m'
        points = client.query(query).get_points()
        df = pd.DataFrame(list(points))

        if df.empty:
            status.update({'message': 'Nessun dato recente trovato su InfluxDB.', 'level': 2})
            return None, status

        df['utc_datetime'] = pd.to_datetime(df['time'])
        df = df.set_index('utc_datetime').sort_index()

        expected_numeric = ['temperature', 'humidity', 'co2', 'voc']
        numeric_cols = [col for col in expected_numeric if col in df.columns]
        df_numeric = df[numeric_cols].apply(pd.to_numeric, errors='coerce').dropna()

        df_resampled = df_numeric.resample('min').mean().interpolate(method='linear', limit_direction='both')

        if len(df_resampled) < config.MIN_RECORDS_REQUIRED:
            status.update({'message': f'Dati insufficienti dopo la pulizia ({len(df_resampled)} punti).', 'level': 2})
            return None, status

        df_final = df_resampled.rename(columns={"temperature": "temperature_sensor", "humidity": "humidity_sensor"})
        df_final['device'] = device_id

        df_final['dew_point_sensor'] = df_final.apply(
            lambda row: calculate_dew_point(row['temperature_sensor'], row['humidity_sensor']),
            axis=1
        )
        df_final['absolute_humidity_sensor'] = df_final.apply(
            lambda row: calculate_absolute_humidity(row['temperature_sensor'], row['humidity_sensor']),
            axis=1
        )

        status.update({'message': 'Dati dei sensori elaborati con successo.', 'level': 0})
        return df_final, status

    except Exception as e:
        status.update({'message': f"Errore InfluxDB: {e}", 'level': 2})
        return None, status

def get_last_known_states(device_id):
    last_states = {f"state_{act_it}": 0 for act_it in config.ALL_ACTUATORS_IT}
    if not os.path.exists(config.DB_PATH):
        print("[WARNING] Database file not found. Returning default actuator states (all OFF).")
        return last_states
    
    with sqlite3.connect(config.DB_PATH) as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        en_cols = [f"state_{config.ACTUATOR_MAP_IT_TO_EN[act_it]}" for act_it in config.ALL_ACTUATORS_IT]
        try:
            cur.execute(f"SELECT {', '.join(en_cols)} FROM latest_status WHERE device_id = ?", (device_id,))
            row = cur.fetchone()
            if row:
                for act_it in config.ALL_ACTUATORS_IT:
                    en_name = config.ACTUATOR_MAP_IT_TO_EN[act_it]
                    last_states[f'state_{act_it}'] = row[f'state_{en_name}'] if row[f'state_{en_name}'] is not None else 0
                print(f"[LOG] Last known states for {device_id} loaded from DB.")
            else:
                print(f"[LOG] No previous state found for {device_id}. Using defaults (all OFF).")
        except sqlite3.OperationalError as e:
            print(f"[ERROR] Could not read from latest_status table: {e}. Using default states.")
    return last_states

# ==============================================================================
# --- CALCULATION FUNCTIONS ---
# ==============================================================================
def calculate_absolute_humidity(temp, rh):
    if pd.isna(temp) or pd.isna(rh): return np.nan
    e_s = 611.2 * np.exp((17.67 * temp) / (temp + 243.5))
    e = e_s * (rh / 100.0)
    return (e / (461.5 * (temp + 273.15))) * 1000

def calculate_relative_humidity(temp_c, abs_hum_g_m3):
    if pd.isna(temp_c) or pd.isna(abs_hum_g_m3): return np.nan
    temp_k = temp_c + 273.15
    e_s = 611.2 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    if e_s == 0: return np.nan
    e = (abs_hum_g_m3 * 461.5 * temp_k) / 1000
    rh = (e / e_s) * 100
    return max(0, min(100, rh))

def calculate_dew_point(temp, rh):
    if pd.isna(temp) or pd.isna(rh) or rh <= 0: return np.nan
    a, b = 17.27, 237.7
    alpha = ((a * temp) / (b + temp)) + np.log(rh / 100.0)
    return (b * alpha) / (a - alpha)

def calculate_heat_index(temp, rh):
    if pd.isna(temp) or pd.isna(rh) or temp < 26.7 or rh < 40: return temp
    return -8.7847 + 1.6114 * temp + 2.3385 * rh - 0.1461 * temp * rh - 0.0123 * (temp**2) - 0.0164 * (rh**2) + 0.0022 * (temp**2) * rh + 0.0007 * temp * (rh**2) - 0.0000036 * (temp**2) * (rh**2)

def calculate_iaq_index(co2, voc):
    if co2 is None or voc is None or np.isnan(co2) or np.isnan(voc): return None
    co2_score = 100 * max(0, (2000 - co2) / (2000 - 800)) if co2 > 800 else 100
    voc_score = 100 * max(0, (2000 - voc) / (2000 - 250)) if voc > 250 else 100
    return min(co2_score, voc_score)

def calculate_global_comfort(heat_index, iaq_index):
    if heat_index is None or iaq_index is None: return None
    thermal_score = 100 * max(0, (31 - heat_index) / (31 - 24)) if heat_index > 24 else 100
    return (thermal_score * 0.6) + (iaq_index * 0.4)


# ==============================================================================
# --- FEATURE ENGINEERING ---
# ==============================================================================
def create_features_for_actuator_model(
    df_hist,
    weather_df,
    available_actuators_it,
    actuator_states,
    required_features_list
):
    import numpy as np
    import pandas as pd

    # ---------------- helpers ----------------
    def vpd_kpa(temp_c, rh_pct):
        es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
        ea = (rh_pct.clip(0, 100) / 100.0) * es
        return (es - ea).astype(float)

    def dew_point_from_t_rh(temp_c, rh_pct):
        b, c = 17.62, 243.12
        rh = np.clip(rh_pct, 1e-3, 100.0)
        gamma = np.log(rh / 100.0) + (b * temp_c) / (c + temp_c)
        return (c * gamma) / (b - gamma)

    def ensure_time_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Garantisce coerenza di local_datetime e utc_datetime senza usare np.issubdtype,
        compatibile con dtype tz-aware (datetime64[ns, UTC]).
        """
        import pandas as pd

        # local_datetime
        if "local_datetime" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                # "wall clock": rimuovo tz se presente mantenendo l'orario locale
                local_idx = df.index
                if local_idx.tz is not None:
                    df["local_datetime"] = local_idx.tz_convert(None)
                else:
                    df["local_datetime"] = local_idx
            else:
                df["local_datetime"] = pd.NaT
        df["local_datetime"] = pd.to_datetime(df["local_datetime"], errors="coerce")

        # utc_datetime
        if "utc_datetime" in df.columns:
            # interpreta tutto come UTC (naive → UTC; aware → convert to UTC)
            df["utc_datetime"] = pd.to_datetime(df["utc_datetime"], errors="coerce", utc=True)
        else:
            if isinstance(df.index, pd.DatetimeIndex):
                utc_idx = df.index
                if utc_idx.tz is None:
                    # indice naive: trattalo come UTC
                    df["utc_datetime"] = utc_idx.tz_localize("UTC")
                else:
                    # indice già tz-aware: converti a UTC
                    df["utc_datetime"] = utc_idx.tz_convert("UTC")
            else:
                df["utc_datetime"] = pd.NaT

        return df

    def add_time_cyclic_and_sun(df):
        # Feature cicliche da LOCAL TIME (wall clock)
        local_dt = df["local_datetime"]
        # se tz-aware, rimuovo tz mantenendo l'orario locale
        if pd.api.types.is_datetime64tz_dtype(local_dt):
            local_dt = local_dt.dt.tz_convert(None)

        hour = local_dt.dt.hour.fillna(0).astype(int)
        minute = local_dt.dt.minute.fillna(0).astype(int)
        hour_frac = (hour + minute / 60.0) % 24
        dow = local_dt.dt.dayofweek.fillna(0).astype(int)
        doy = local_dt.dt.dayofyear.fillna(1).astype(int)

        df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)
        df["dow_sin"]  = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"]  = np.cos(2 * np.pi * dow / 7)
        df["doy_sin"]  = np.sin(2 * np.pi * doy / 366)
        df["doy_cos"]  = np.cos(2 * np.pi * doy / 366)

        # Differenze sunrise/sunset:
        if df["utc_datetime"].notna().any():
            # via UTC (ideale)
            dt_utc = df["utc_datetime"]
            sr_utc = pd.to_datetime(df.get("sunrise_time"), errors="coerce", utc=True)
            ss_utc = pd.to_datetime(df.get("sunset_time"),  errors="coerce", utc=True)
            df["minutes_from_sunrise"] = ((dt_utc - sr_utc).dt.total_seconds() / 60).fillna(0)
            df["minutes_to_sunset"]    = ((ss_utc - dt_utc).dt.total_seconds() / 60).fillna(0)
            df["is_daylight"] = ((dt_utc >= sr_utc) & (dt_utc <= ss_utc)).astype("Int64").fillna(0).astype(int)
        else:
            # fallback locale (naive) per non crashare
            sr = pd.to_datetime(df.get("sunrise_time"), errors="coerce")
            ss = pd.to_datetime(df.get("sunset_time"),  errors="coerce")
            if pd.api.types.is_datetime64tz_dtype(sr): sr = sr.dt.tz_convert(None)
            if pd.api.types.is_datetime64tz_dtype(ss): ss = ss.dt.tz_convert(None)
            dt = local_dt
            df["minutes_from_sunrise"] = ((dt - sr).dt.total_seconds() / 60).fillna(0)
            df["minutes_to_sunset"]    = ((ss - dt).dt.total_seconds() / 60).fillna(0)
            df["is_daylight"] = ((dt >= sr) & (dt <= ss)).astype("Int64").fillna(0).astype(int)

        return df

    def add_rolling_schema(df, cols, win_short=5, win_long=30):
        df.sort_values(["device", "utc_datetime"], inplace=True)
        for c in cols:
            g = df.groupby("device")[c]
            df[f"{c}_trend_{win_short}m"] = g.diff(win_short)
            df[f"{c}_trend_{win_long}m"]  = g.diff(win_long)
            df[f"{c}_mean_{win_short}m"]  = g.rolling(win_short, min_periods=2).mean().reset_index(level=0, drop=True)
            df[f"{c}_mean_{win_long}m"]   = g.rolling(win_long,  min_periods=2).mean().reset_index(level=0, drop=True)
            df[f"{c}_std_{win_short}m"]   = g.rolling(win_short, min_periods=2).std().reset_index(level=0, drop=True)
            df[f"{c}_std_{win_long}m"]    = g.rolling(win_long,  min_periods=2).std().reset_index(level=0, drop=True)
            df[f"{c}_accel_1m"]           = df.groupby("device")[f"{c}_trend_{win_short}m"].diff()
        return df

    def add_external_trends(df, cols, win_short=5, win_long=30):
        for c in cols:
            g = df.groupby("device")[c]
            df[f"{c}_trend_{win_short}m"] = g.diff(win_short)
            df[f"{c}_trend_{win_long}m"]  = g.diff(win_long)
        return df

    def add_device_baselines(df, cols, window_minutes=1440):
        for c in cols:
            base = (
                df.groupby("device")[c]
                .rolling(window_minutes, min_periods=60)
                .median()
                .reset_index(level=0, drop=True)
            )
            df[f"{c}_baseline_delta"] = df[c] - base
        return df

    def add_event_flags(df):
        eps = 1e-6
        def drop_flag(var): return (df[f"{var}_trend_5m"] < -(df[f"{var}_std_30m"].fillna(0) + eps)).astype(int)
        def rise_flag(var): return (df[f"{var}_trend_5m"] >  (df[f"{var}_std_30m"].fillna(0) + eps)).astype(int)
        df["co2_drop_flag_5m"]  = drop_flag("co2")
        df["voc_drop_flag_5m"]  = drop_flag("voc")
        df["temp_drop_flag_5m"] = drop_flag("temperature_sensor")
        df["ah_rise_flag_5m"]   = rise_flag("absolute_humidity_sensor")
        return df

    # ---------------- pipeline ----------------
    df = pd.concat([df_hist.copy(), weather_df.copy()], axis=1)
    df.ffill(inplace=True); df.bfill(inplace=True)

    # assicurati che ci siano device & tempi
    if "device" not in df.columns:
        # fallback: singolo device
        df["device"] = df_hist.get("device", pd.Series(["device_0"] * len(df), index=df.index))

    df = ensure_time_columns(df)
    df.sort_values(["device", "utc_datetime"], inplace=True)

    # absolute humidity e dew point
    df["absolute_humidity_sensor"] = df.apply(
        lambda r: calculate_absolute_humidity(r["temperature_sensor"], r["humidity_sensor"]), axis=1
    )
    df["absolute_humidity_external"] = df.apply(
        lambda r: calculate_absolute_humidity(r["temperature_external"], r["humidity_external"]), axis=1
    )
    if "dew_point_sensor" not in df.columns or df["dew_point_sensor"].isna().any():
        df["dew_point_sensor"] = dew_point_from_t_rh(df["temperature_sensor"], df["humidity_sensor"])
    if "dew_point_external" not in df.columns or df["dew_point_external"].isna().any():
        df["dew_point_external"] = dew_point_from_t_rh(df["temperature_external"], df["humidity_external"])

    # tempo ciclico + daylight
    df = add_time_cyclic_and_sun(df)

    # gradienti, VPD, interazioni
    df["temp_diff_in_out"]   = df["temperature_sensor"] - df["temperature_external"]
    df["ah_diff_in_out"]     = df["absolute_humidity_sensor"] - df["absolute_humidity_external"]
    df["dewpoint_diff_in_out"] = df["dew_point_sensor"] - df["dew_point_external"]
    df["vpd_in"]   = vpd_kpa(df["temperature_sensor"],   df["humidity_sensor"])
    df["vpd_out"]  = vpd_kpa(df["temperature_external"], df["humidity_external"])
    df["vpd_diff"] = df["vpd_in"] - df["vpd_out"]
    df["temp_diff_x_wind"] = df["temp_diff_in_out"] * df.get("wind_speed", 0).fillna(0)

    # rolling & trends
    INTERNAL_SERIES = ["temperature_sensor", "absolute_humidity_sensor", "co2", "voc"]
    df = add_rolling_schema(df, INTERNAL_SERIES, win_short=5, win_long=30)
    EXTERNAL_SERIES = ["temperature_external", "absolute_humidity_external"]
    df = add_external_trends(df, EXTERNAL_SERIES, win_short=5, win_long=30)

    # baseline (mediana ~24h)
    BASELINE_COLS = ["co2", "voc", "temperature_sensor", "absolute_humidity_sensor"]
    df = add_device_baselines(df, BASELINE_COLS, window_minutes=1440)

    # event flags
    df = add_event_flags(df)

    # pulizia NaN su nuove feature (per-device bfill/ffill) + numeriche
    base_cols = set(df_hist.columns).union(set(weather_df.columns))
    new_cols = [c for c in df.columns if c not in base_cols]
    if new_cols:
        df[new_cols] = df.groupby("device")[new_cols].transform(lambda x: x.bfill().ffill())
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # availability/state se ti servono a valle (non tornano tra le feature)
    if "config" in globals() and hasattr(config, "ALL_ACTUATORS_IT"):
        aux = {}
        for act_it in config.ALL_ACTUATORS_IT:
            aux[f"available_{act_it}"] = 1 if act_it in available_actuators_it else 0
            aux[f"state_{act_it}"] = (actuator_states or {}).get(f"state_{act_it}", 0)
        for k, v in aux.items():
            df.loc[df.index[-1], k] = v

    # ultima riga per l'inferenza
    latest = df.iloc[-1:].copy()

    # garantisci tutte le required features
    for col in required_features_list:
        if col not in latest.columns:
            latest[col] = 0

    return latest[required_features_list]

def create_features_for_prediction_model(df_hist, weather_df, required_features_list):
    """
    Crea feature per il modello di PREDIZIONE, che usa feature cicliche temporali
    (hod_sin, hod_cos) e ignora gli attuatori.
    """
    print("[LOG] Calculating features for the PREDICTION model...")
    df = df_hist.copy()
    df = pd.concat([df, weather_df], axis=1)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Aggiunge feature cicliche temporali
    hour_of_day = df.index.hour + df.index.minute / 60.0
    df['hod_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
    df['hod_cos'] = np.cos(2 * np.pi * hour_of_day / 24)

    # Calcola feature derivate e time-series (rolling, trend, etc.)
    # (qui va inserita tutta la logica di calcolo delle medie, deviazioni, trend, etc.
    # che avevamo nella versione precedente)
    df['absolute_humidity_sensor'] = df.apply(lambda r: calculate_absolute_humidity(r['temperature_sensor'], r['humidity_sensor']), axis=1)
    # ... e tutti gli altri calcoli ...

    # Prepara il vettore finale
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    latest_vector = df.iloc[-1:].copy()
    
    missing_cols = set(required_features_list) - set(latest_vector.columns)
    if missing_cols:
        for col in missing_cols:
            latest_vector[col] = 0

    return latest_vector[required_features_list]


# ==============================================================================
# --- DATABASE SAVING FUNCTIONS ---
# ==============================================================================
def save_results_to_db(record):
    """
    Salva/aggiorna lo stato corrente in latest_status.
    Supporta dinamicamente campi extra come prob_<Actuator> oltre a state_<Actuator>.

    record: dict con almeno:
      - device_id, timestamp, temperature, humidity, co2, voc, heatIndex, iaqIndex,
        globalComfort, status_level, status_message
      - opzionali: state_<EN>, prob_<EN> per ciascun attuatore
    """
    try:
        cols = list(record.keys())
        vals = [record[c] for c in cols]

        placeholders = ",".join(["?"] * len(cols))
        col_list = ",".join(cols)
        update_clause = ",".join([f"{c}=excluded.{c}" for c in cols if c != "device_id"])

        with sqlite3.connect(config.DB_PATH) as con:
            con.execute(
                f"""
                INSERT INTO latest_status ({col_list})
                VALUES ({placeholders})
                ON CONFLICT(device_id) DO UPDATE SET
                    {update_clause}
                """,
                vals
            )
            con.commit()
        print(f"[DB] Saved current state for {record.get('device_id')}.")
    except sqlite3.Error as e:
        print(f"[ERROR] Failed to save current state: {e}")

def save_predictions_and_suggestions_to_db(device_id, baseline_predictions, winning_suggestions):
    print(f"[DB] Saving predictions and suggestions for {device_id}...")
    try:
        with sqlite3.connect(config.DB_PATH) as con:
            cur = con.cursor()
            
            cur.execute("DELETE FROM suggestions WHERE device_id = ?", (device_id,))
            cur.execute("DELETE FROM predictions WHERE device_id = ?", (device_id,))

            insert_query = "INSERT INTO predictions (device_id, horizon, is_suggestion_for, temperature, humidity, co2, voc, heatIndex, iaqIndex, globalComfort) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            if baseline_predictions:
                cur.executemany(insert_query, [tuple(rec.values()) for rec in baseline_predictions])
                print(f"[DB] Saved {len(baseline_predictions)} baseline predictions.")

            if winning_suggestions:
                for suggestion in winning_suggestions:
                    sugg_states = suggestion['states']
                    sugg_cols = ['device_id'] + [f'state_{config.ACTUATOR_MAP_IT_TO_EN[act]}' for act in config.ALL_ACTUATORS_IT]
                    sugg_vals = [device_id] + [sugg_states[f'state_{act}'] for act in config.ALL_ACTUATORS_IT]
                    cur.execute(f"INSERT INTO suggestions ({', '.join(sugg_cols)}) VALUES (?{',?' * len(config.ALL_ACTUATORS_IT)})", sugg_vals)
                    suggestion_id = cur.lastrowid
                    
                    suggestion_predictions = [tuple(dict(rec, is_suggestion_for=suggestion_id).values()) for rec in suggestion['predictions']]
                    cur.executemany(insert_query, suggestion_predictions)
                print(f"[DB] Saved {len(winning_suggestions)} suggestions to the database.")
    except sqlite3.Error as e:
        print(f"[ERROR] Failed to save predictions to database: {e}")


def get_all_device_ids():
    """
    Ritorna la lista di tutti i device_id presenti nella tabella devices.
    """
    try:
        with sqlite3.connect(config.DB_PATH) as con:
            cur = con.cursor()
            cur.execute("SELECT device_id FROM devices")
            return [row[0] for row in cur.fetchall()]
    except sqlite3.Error as e:
        print(f"[ERROR] get_all_device_ids: {e}")
        return []


def save_prediction_timestamp(device_id):
    """
    Upsert del timestamp (UTC) dell'ultima predizione in predictions_meta.
    """
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with sqlite3.connect(config.DB_PATH) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS predictions_meta (
                    device_id TEXT PRIMARY KEY,
                    generated_at TEXT
                )
            """)
            con.execute("""
                INSERT INTO predictions_meta (device_id, generated_at)
                VALUES (?, ?)
                ON CONFLICT(device_id) DO UPDATE SET
                    generated_at = excluded.generated_at
            """, (device_id, ts))
            con.commit()
    except sqlite3.Error as e:
        print(f"[ERROR] save_prediction_timestamp: {e}")


def load_hysteresis_map_en():
    """
    Legge le soglie di isteresi globali (per attuatore EN) dalla tabella hysteresis_global.
    Fallback: usa i default di config.HYSTERESIS_THRESHOLDS mappati IT->EN se la tabella non esiste/vuota.
    """
    try:
        with sqlite3.connect(config.DB_PATH) as con:
            con.row_factory = sqlite3.Row
            # se la tabella non esiste, scatta l'eccezione nel SELECT -> si va in fallback
            rows = con.execute("SELECT actuator_name, th_on, th_off FROM hysteresis_global").fetchall()
            if rows:
                return {r["actuator_name"]: {"on": float(r["th_on"]), "off": float(r["th_off"])} for r in rows}
    except Exception as e:
        print(f"[WARN] load_hysteresis_map_en: {e} (using defaults)")

    # fallback ai default definiti in config (IT->EN)
    try:
        en_map = {}
        for it_name, en_name in config.ACTUATOR_MAP_IT_TO_EN.items():
            th = config.HYSTERESIS_THRESHOLDS.get(it_name, {"on": 0.7, "off": 0.4})
            en_map[en_name] = {"on": float(th["on"]), "off": float(th["off"])}
        return en_map
    except Exception as e:
        print(f"[ERROR] load_hysteresis_map_en defaults: {e}")
        return {}


def upsert_hysteresis_en(act_en, th_on, th_off):
    """
    Aggiorna/crea le soglie globali per un attuatore EN nella tabella hysteresis_global.
    """
    from datetime import datetime, timezone
    th_on = float(th_on); th_off = float(th_off)
    if not (0.0 <= th_off < th_on <= 1.0):
        raise ValueError("Soglie non valide: deve valere 0 <= OFF < ON <= 1")

    try:
        with sqlite3.connect(config.DB_PATH) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS hysteresis_global (
                    actuator_name TEXT PRIMARY KEY,
                    th_on REAL NOT NULL,
                    th_off REAL NOT NULL,
                    updated_at TEXT
                )
            """)
            con.execute("""
                INSERT INTO hysteresis_global (actuator_name, th_on, th_off, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(actuator_name) DO UPDATE SET
                    th_on = excluded.th_on,
                    th_off = excluded.th_off,
                    updated_at = excluded.updated_at
            """, (act_en, th_on, th_off, datetime.now(timezone.utc).isoformat()))
            con.commit()
    except sqlite3.Error as e:
        print(f"[ERROR] upsert_hysteresis_en: {e}")
        raise


def get_previous_state_for_device(device_id, actuator_en):
    """
    Restituisce lo stato precedente (0/1) per un attuatore EN dal latest_status.
    Se non trovato, torna 0.
    """
    try:
        with sqlite3.connect(config.DB_PATH) as con:
            con.row_factory = sqlite3.Row
            row = con.execute(
                f"SELECT state_{actuator_en} AS st FROM latest_status WHERE device_id = ?",
                (device_id,)
            ).fetchone()
            if row and row["st"] is not None:
                return int(row["st"])
    except sqlite3.Error as e:
        print(f"[WARN] get_previous_state_for_device: {e}")
    return 0


def load_features_for_device(device_id):
    """
    Costruisce il vettore di feature 'current' per il modello di classificazione.
    Usa l'infrastruttura esistente:
      - history da InfluxDB (get_sensor_history)
      - meteo per device (manage_weather_history)
      - available_actuators da device_actuators (mappati in IT)
      - create_feature_vector(..., required_features_list=config.CLASSIFICATION_FEATURES)

    Ritorna: DataFrame a singola riga con colonne nell'ordine richiesto e una colonna 'device_id'.
    """
    # 1) recupera lat/lon per meteo + lista attuatori abilitati (EN -> IT)
    with sqlite3.connect(config.DB_PATH) as con:
        con.row_factory = sqlite3.Row
        dev = con.execute("SELECT owm_lat, owm_lon FROM devices WHERE device_id = ?", (device_id,)).fetchone()
        if not dev:
            print(f"[ERROR] load_features_for_device: device '{device_id}' non trovato in devices.")
            return None
        lat, lon = dev["owm_lat"], dev["owm_lon"]

        act_rows = con.execute(
            "SELECT actuator_name, is_enabled FROM device_actuators WHERE device_id = ? AND is_enabled = 1",
            (device_id,)
        ).fetchall()
        enabled_en = [r["actuator_name"] for r in act_rows]
    # mappa EN->IT
    en_to_it = {v: k for k, v in config.ACTUATOR_MAP_IT_TO_EN.items()}
    available_actuators_it = [en_to_it[a] for a in enabled_en if a in en_to_it]

    # 2) meteo e history
    weather_df = manage_weather_history(device_id=device_id, lat=lat, lon=lon)
    status = {'level': 0, 'message': 'OK'}
    history_df, status = get_sensor_history(device_id, status)
    if history_df is None or weather_df is None or weather_df.empty:
        print(f"[ERROR] load_features_for_device: dati insufficienti per {device_id}.")
        return None

    # 3) calcolo feature nell'ordine richiesto dal training
    try:
        fv = create_feature_vector(
            df_hist=history_df,
            weather_df=weather_df,
            available_actuators_it=available_actuators_it,
            actuator_states=None,
            required_features_list=getattr(config, "CLASSIFICATION_FEATURES", [])
        )
    except Exception as e:
        print(f"[ERROR] load_features_for_device/create_feature_vector: {e}")
        return None

    # 4) aggiungi device_id per l'isteresi per-attuatore
    fv = fv.copy()
    fv["device_id"] = device_id
    return fv

def create_features_for_action_model(df_hist, weather_df, required_features_list):
    """
    Replica il feature engineering del training per il REGRESSORE AZIONE.
    NON gestisce one-hot action_* (le mette chi le usa).
    Se required_features_list contiene colonne non calcolabili (es. action_*, *_eval_*),
    questa funzione NON fallisce: calcola il massimo possibile e lascia le altre a cura del chiamante.
    """
    df = df_hist.copy()
    df = pd.concat([df, weather_df], axis=1)
    df.ffill(inplace=True); df.bfill(inplace=True)

    # AH interna/esterna + delta
    df['absolute_humidity_sensor'] = df.apply(
        lambda r: calculate_absolute_humidity(r['temperature_sensor'], r['humidity_sensor']), axis=1)
    df['absolute_humidity_external'] = df.apply(
        lambda r: calculate_absolute_humidity(r['temperature_external'], r['humidity_external']), axis=1)
    df['temperature_delta'] = df['temperature_sensor'] - df['temperature_external']
    df['humidity_delta'] = df['absolute_humidity_sensor'] - df['absolute_humidity_external']

    # Dew point esterno (alcuni modelli lo avevano)
    df['dew_point_external'] = df.apply(
        lambda r: calculate_dew_point(r['temperature_external'], r['humidity_external']), axis=1)
    # Dew point interno se serve
    df['dew_point_sensor'] = df.apply(
        lambda r: calculate_dew_point(r['temperature_sensor'], r['humidity_sensor']), axis=1)

    # rolling/trend/accel per interni
    base_feats = ['temperature_sensor', 'absolute_humidity_sensor', 'co2', 'voc']
    roll_w = [5,10,15,30]
    diff_w = [1,3,5,10,15]
    for feat in base_feats:
        if feat not in df.columns: continue
        g = df[feat]
        for w in roll_w:
            df[f"{feat}_mean_{w}m"] = g.rolling(window=w, min_periods=2).mean()
            df[f"{feat}_std_{w}m"]  = g.rolling(window=w, min_periods=2).std()
        for w in diff_w:
            df[f"{feat}_trend_{w}m"] = g.diff(w)
        df[f"{feat}_accel_1m"] = df[f"{feat}_trend_1m"].diff()

    # trend esterni
    for w in [5,15,30]:
        if 'temperature_external' in df.columns:
            df[f"temperature_external_trend_{w}m"] = df['temperature_external'].diff(w)
        if 'absolute_humidity_external' in df.columns:
            df[f"absolute_humidity_external_trend_{w}m"] = df['absolute_humidity_external'].diff(w)

    # cicliche orarie (naming del training)
    hour_of_day = df.index.hour + df.index.minute/60.0
    df['hour_sin'] = np.sin(2*np.pi*hour_of_day/24)
    df['hour_cos'] = np.cos(2*np.pi*hour_of_day/24)

    # clamp + log1p per co2/voc
    co2_lo, co2_hi = float(getattr(config, "MIN_CO2", 400.0)), 2500.0
    voc_lo, voc_hi = float(getattr(config, "MIN_VOC", 0.0)), 5000.0
    if 'co2' in df.columns:
        df['co2_clamped'] = df['co2'].clip(co2_lo, co2_hi)
        df['co2_log1p']   = np.log1p(df['co2_clamped'])
    if 'voc' in df.columns:
        df['voc_clamped'] = df['voc'].clip(voc_lo, voc_hi)
        df['voc_log1p']   = np.log1p(df['voc_clamped'])

    df.ffill(inplace=True); df.bfill(inplace=True)
    latest = df.iloc[-1:].copy()

    # NON validare required_features_list qui (potrebbe contenere action_* o *_eval_*):
    return latest


def clamp_and_log_feats(df):
    """
    Aggiunge colonne clamped/log1p per CO2/VOC se mancano.
    Non modifica le colonne originali.
    """
    out = df.copy()
    co2_lo, co2_hi = float(getattr(config, "MIN_CO2", 400.0)), 2000.0
    voc_lo, voc_hi = float(getattr(config, "MIN_VOC", 0.0)), 3000.0
    if "co2" in out.columns and "co2_clamped" not in out.columns:
        out["co2_clamped"] = out["co2"].clip(co2_lo, co2_hi)
        out["co2_log1p"]   = np.log1p(out["co2_clamped"])
    if "voc" in out.columns and "voc_clamped" not in out.columns:
        out["voc_clamped"] = out["voc"].clip(voc_lo, voc_hi)
        out["voc_log1p"]   = np.log1p(out["voc_clamped"])
    return out