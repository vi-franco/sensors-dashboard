#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import sqlite3
import json
import time

import requests

import config

def get_unique_locations(cur):
    """Recupera le coordinate uniche dei dispositivi."""
    print("Recupero coordinate uniche...")
    try:
        cur.execute("SELECT DISTINCT lat, lng FROM devices WHERE lat IS NOT NULL AND lng IS NOT NULL;")
        return cur.fetchall()
    except sqlite3.Error as e:
        print(f"Errore DB nel leggere le coordinate: {e}", file=sys.stderr)
        return []

def fetch_weather_from_api(lat, lng):
    """Chiama l'API di OpenWeatherMap."""
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lng={lng}&appid={config.OWM_API_KEY}&units=metric"
    print(f"Chiamata API per ({lat}, {lng})...")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Errore API per ({lat}, {lng}): {e}", file=sys.stderr)
        return None

def save_weather_data(cur, lat, lng, data):
    """Salva i nuovi dati meteo nel database, ignorando i duplicati."""
    if not data:
        return

    try:
        weather_list = data.get('weather', [])
        weather_main = weather_list[0].get('main') if weather_list else None

        record_tuple = (
            data.get('dt'),
            float(lat),
            float(lng),
            data.get('main', {}).get('temp'),
            data.get('main', {}).get('humidity'),
            data.get('main', {}).get('pressure'),
            data.get('wind', {}).get('speed'),
            data.get('clouds', {}).get('all'),
            data.get('rain', {}).get('1h', 0.0),
            data.get('sys', {}).get('sunrise'),
            data.get('sys', {}).get('sunset'),
            data.get('timezone'),
            weather_main
        )

        cur.execute("""
            INSERT OR IGNORE INTO external_weather
            (dt, lat, lng, temperature, humidity, pressure, wind_speed, clouds_percentage, rain_1h, sunrise_time, sunset_time, timezone_offset, weather_main)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, record_tuple)

        if cur.rowcount > 0:
            print(f"Dati per ({lat}, {lng}) con timestamp {data.get('dt')} salvati.")
        else:
            print(f"Dati per ({lat}, {lng}) con timestamp {data.get('dt')} già presenti, ignorati.")

    except sqlite3.Error as e:
        print(f"Errore DB nel salvare i dati per ({lat}, {lng}): {e}", file=sys.stderr)
    except KeyError as e:
        print(f"Formato API inatteso per ({lat}, {lng}). Campo mancante: {e}", file=sys.stderr)

def main():
    print("\n--- Avvio script aggiornamento meteo ---")
    conn = None
    try:
        conn = sqlite3.connect(config.DB_PATH)
        with conn:
            cur = conn.cursor()

            four_hour_ago_ts = int(time.time()) - (4 * 3600)
            print(f"Pulizia record più vecchi di {four_hour_ago_ts}...")
            cur.execute("DELETE FROM external_weather WHERE dt < ?", (four_hour_ago_ts,))
            print(f"{cur.rowcount} righe obsolete rimosse.")

            locations = get_unique_locations(cur)
            if not locations:
                print("Nessuna coordinata da processare.")
                return

            print(f"Trovate {len(locations)} coordinate uniche da processare.")
            for lat, lng in locations:
                weather_data = fetch_weather_from_api(lat, lng)
                if weather_data:
                    save_weather_data(cur, lat, lng, weather_data)

    except sqlite3.Error as e:
        print(f"Errore DB: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()
        print("--- Script terminato ---")

if __name__ == "__main__":
    main()