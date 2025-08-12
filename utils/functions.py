def vpd_kpa(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    # Tetens formula (T in °C) → kPa
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    ea = (rh_pct.clip(0,100) / 100.0) * es
    return (es - ea).astype(float)
