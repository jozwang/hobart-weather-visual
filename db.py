"""
Database access layer.

Matches the schema created by collector.py:
  - current_weather(fetched_at, location, dt, temp_c, feels_like, humidity,
                     wind_speed, wind_deg, weather, description, visibility,
                     rain_1h, raw_json)
  - hourly_forecast(fetched_at, location, forecast_dt, temp_c, feels_like,
                     humidity, wind_speed, pop, rain_1h, weather, description)
  - daily_forecast(fetched_at, location, forecast_dt, ..., pop, rain_mm, ...)

rain_1h and rain_mm are NULL when there was no rain (not 0) -- this matters
for the accuracy queries below, so we treat "rain observed" as
`rain_1h IS NOT NULL` rather than `rain_1h > 0`.
"""

import os

import pandas as pd
import psycopg2
import streamlit as st

LOCATIONS = ["Hobart", "West Moonah", "Saltwater River"]


def get_database_url() -> str:
    """Resolve DATABASE_URL from Streamlit secrets first, then environment.

    Locally: put DATABASE_URL in .streamlit/secrets.toml (gitignored).
    On Streamlit Community Cloud: set it in the app's Secrets manager
    (Settings -> Secrets), not in the GitHub repo itself.
    """
    try:
        return st.secrets["DATABASE_URL"]
    except Exception:
        pass
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL not found. Set it in .streamlit/secrets.toml locally, "
            "or in the Streamlit Community Cloud app's Secrets settings."
        )
    return url


@st.cache_resource(show_spinner=False)
def get_connection():
    return psycopg2.connect(get_database_url(), sslmode="require")


def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run a read-only query, reconnecting once if the cached connection died."""
    conn = get_connection()
    try:
        return pd.read_sql(sql, conn, params=params)
    except psycopg2.OperationalError:
        st.cache_resource.clear()
        conn = get_connection()
        return pd.read_sql(sql, conn, params=params)


@st.cache_data(ttl=300, show_spinner=False)
def get_latest_current_weather() -> pd.DataFrame:
    sql = """
        SELECT DISTINCT ON (location)
            location, fetched_at, temp_c, feels_like, humidity,
            wind_speed, wind_deg, weather, description, rain_1h
        FROM current_weather
        ORDER BY location, fetched_at DESC
    """
    return run_query(sql)


@st.cache_data(ttl=300, show_spinner=False)
def get_current_weather_trend(days_back: int = 3) -> pd.DataFrame:
    sql = """
        SELECT location, fetched_at, temp_c, humidity, wind_speed, rain_1h
        FROM current_weather
        WHERE fetched_at >= NOW() - (%(days)s || ' days')::interval
        ORDER BY fetched_at
    """
    return run_query(sql, {"days": days_back})


@st.cache_data(ttl=300, show_spinner=False)
def get_current_weather_window(days_back: int = 10) -> pd.DataFrame:
    sql = """
        SELECT location, fetched_at, dt, temp_c, humidity, rain_1h
        FROM current_weather
        WHERE fetched_at >= NOW() - (%(days)s || ' days')::interval
        ORDER BY location, fetched_at
    """
    return run_query(sql, {"days": days_back})


@st.cache_data(ttl=300, show_spinner=False)
def get_hourly_forecast_window(days_back: int = 10) -> pd.DataFrame:
    """All hourly forecast snapshots fetched in the last N days.

    Because the collector runs hourly and each run writes up to 48 rows of
    hourly_forecast, this table naturally contains many overlapping
    predictions for the same forecast_dt made at different fetched_at times
    -- that's what lets us pick "the forecast made N hours ahead".
    """
    sql = """
        SELECT location, fetched_at, forecast_dt, pop, rain_1h, temp_c
        FROM hourly_forecast
        WHERE fetched_at >= NOW() - (%(days)s || ' days')::interval
        ORDER BY location, forecast_dt, fetched_at
    """
    return run_query(sql, {"days": days_back})


@st.cache_data(ttl=300, show_spinner=False)
def get_daily_forecast_latest() -> pd.DataFrame:
    """Most recent daily_forecast fetch per location (covers ~next 8 days)."""
    sql = """
        WITH latest_fetch AS (
            SELECT location, MAX(fetched_at) AS fetched_at
            FROM daily_forecast
            GROUP BY location
        )
        SELECT d.location, d.fetched_at, d.forecast_dt, d.pop, d.rain_mm,
               d.temp_min, d.temp_max
        FROM daily_forecast d
        JOIN latest_fetch lf
          ON d.location = lf.location AND d.fetched_at = lf.fetched_at
        ORDER BY d.location, d.forecast_dt
    """
    return run_query(sql)
