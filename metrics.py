"""
Metric computations, all operating on DataFrames pulled via db.py.

Timezone note: forecast_dt / dt are unix-epoch seconds from OpenWeather.
We convert to Australia/Hobart local time before deriving a "calendar day",
since UTC day boundaries would misclassify evening/morning observations
for this timezone.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

LOCAL_TZ = ZoneInfo("Australia/Hobart")
RAIN_POP_THRESHOLD_DEFAULT = 0.5


def _epoch_to_local(epoch_series: pd.Series) -> pd.Series:
    return pd.to_datetime(epoch_series, unit="s", utc=True).dt.tz_convert(LOCAL_TZ)


def prep_hourly_forecast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["forecast_ts"] = _epoch_to_local(df["forecast_dt"])
    df["target_date"] = df["forecast_ts"].dt.date
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True).dt.tz_convert(LOCAL_TZ)
    df["lead_hours"] = (df["forecast_ts"] - df["fetched_at"]).dt.total_seconds() / 3600
    return df


def prep_current_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True).dt.tz_convert(LOCAL_TZ)
    df["local_date"] = df["fetched_at"].dt.date
    return df


def daily_actual_rain(current_weather: pd.DataFrame) -> pd.DataFrame:
    """One row per location/day: did it rain that day (any observed rain_1h)."""
    cw = prep_current_weather(current_weather)
    agg = (
        cw.groupby(["location", "local_date"])
        .agg(actual_rain=("rain_1h", lambda s: s.notna().any()))
        .reset_index()
        .rename(columns={"local_date": "target_date"})
    )
    return agg


def select_forecast_by_lead(hourly: pd.DataFrame, lead_hours: float | None) -> pd.DataFrame:
    """
    For each (location, target_date), pick one representative forecast.

    lead_hours=None  -> "latest available": the most recent forecast that was
                         fetched before the target day started.
    lead_hours=24/48/... -> the forecast whose lead time is closest to that
                         value (e.g. 24 = "made ~1 day ahead").
    """
    df = prep_hourly_forecast(hourly)
    day_pop = (
        df.groupby(["location", "target_date", "fetched_at"])
        .agg(pop=("pop", "mean"), lead_hours=("lead_hours", "mean"))
        .reset_index()
    )

    if lead_hours is None:
        day_pop = day_pop[day_pop["lead_hours"] >= 0]
        if day_pop.empty:
            return day_pop
        idx = day_pop.groupby(["location", "target_date"])["lead_hours"].idxmin()
    else:
        day_pop = day_pop.copy()
        day_pop["lead_diff"] = (day_pop["lead_hours"] - lead_hours).abs()
        if day_pop.empty:
            return day_pop
        idx = day_pop.groupby(["location", "target_date"])["lead_diff"].idxmin()

    return day_pop.loc[idx].reset_index(drop=True)


def build_rain_accuracy_table(
    hourly: pd.DataFrame,
    current_weather: pd.DataFrame,
    lead_hours: float | None,
    threshold: float = RAIN_POP_THRESHOLD_DEFAULT,
) -> pd.DataFrame:
    """Past-days table: forecast vs actual, with hit/false_alarm/miss/correct_no_rain."""
    forecast_pick = select_forecast_by_lead(hourly, lead_hours)
    actual = daily_actual_rain(current_weather)
    if forecast_pick.empty or actual.empty:
        return pd.DataFrame(
            columns=["location", "target_date", "pop", "actual_rain", "outcome"]
        )

    merged = forecast_pick.merge(actual, on=["location", "target_date"], how="inner")
    merged["predicted_rain"] = merged["pop"].fillna(0) >= threshold
    merged["outcome"] = np.select(
        [
            merged["predicted_rain"] & merged["actual_rain"],
            merged["predicted_rain"] & ~merged["actual_rain"],
            ~merged["predicted_rain"] & merged["actual_rain"],
            ~merged["predicted_rain"] & ~merged["actual_rain"],
        ],
        ["hit", "false_alarm", "miss", "correct_no_rain"],
        default="unknown",
    )
    return merged


def precision_recall(accuracy_table: pd.DataFrame) -> dict:
    if accuracy_table.empty:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    counts = accuracy_table["outcome"].value_counts()
    tp = int(counts.get("hit", 0))
    fp = int(counts.get("false_alarm", 0))
    fn = int(counts.get("miss", 0))
    tn = int(counts.get("correct_no_rain", 0))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def build_future_pop_table(hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Stitch hourly (next ~48h) and daily (days 3-7) forecasts into one
    per-location per-day PoP series for the next 7 days, always using the
    most recently fetched data (there's no lead-time choice for the future).
    """
    today = datetime.now(LOCAL_TZ).date()
    horizon_end = today + timedelta(days=7)

    if not hourly.empty:
        hf = prep_hourly_forecast(hourly)
        hf = hf[(hf["target_date"] > today) & (hf["target_date"] <= horizon_end)]
        latest_hf = (
            hf.sort_values("fetched_at")
            .groupby(["location", "target_date"])
            .agg(pop=("pop", "max"))
            .reset_index()
        )
        latest_hf["source_rank"] = 0
    else:
        latest_hf = pd.DataFrame(columns=["location", "target_date", "pop", "source_rank"])

    if not daily.empty:
        df = daily.copy()
        df["target_date"] = _epoch_to_local(df["forecast_dt"]).dt.date
        df = df[(df["target_date"] > today) & (df["target_date"] <= horizon_end)]
        df = df[["location", "target_date", "pop"]]
        df["source_rank"] = 1
    else:
        df = pd.DataFrame(columns=["location", "target_date", "pop", "source_rank"])

    combined = pd.concat([latest_hf, df], ignore_index=True)
    if combined.empty:
        return combined.drop(columns="source_rank", errors="ignore")

    combined = combined.sort_values("source_rank").drop_duplicates(
        subset=["location", "target_date"], keep="first"
    )
    return combined.drop(columns="source_rank").sort_values(["location", "target_date"])


def build_14day_timeline(
    hourly: pd.DataFrame,
    current_weather: pd.DataFrame,
    daily: pd.DataFrame,
    lead_hours: float | None,
    threshold: float = RAIN_POP_THRESHOLD_DEFAULT,
) -> pd.DataFrame:
    """7 past days (forecast @ chosen lead time vs actual) + 7 future days
    (latest available forecast), combined into one table for the chart."""
    past = build_rain_accuracy_table(hourly, current_weather, lead_hours, threshold)
    if not past.empty:
        past = past[["location", "target_date", "pop", "actual_rain", "outcome"]].copy()
        past["past"] = True

    future = build_future_pop_table(hourly, daily)
    if not future.empty:
        future["actual_rain"] = None
        future["outcome"] = None
        future["past"] = False

    combined = pd.concat([past, future], ignore_index=True)
    if combined.empty:
        return combined
    return combined.sort_values(["location", "target_date"])


def hourly_detail_for_day(
    hourly: pd.DataFrame, location: str, target_date, lead_hours: float | None = None
) -> pd.DataFrame:
    """Hour-by-hour PoP breakdown for a single day/location, matching the
    lead-time mode selected at the top of the page where data supports it.

    lead_hours=None -> latest available snapshot for each hour (the only
        option that exists for days beyond ~48h out, since daily_forecast
        only retains the most recent fetch).
    lead_hours=24/48/... -> the snapshot whose lead time is closest to that
        value, PER HOUR -- falls back to the latest available snapshot for
        any hour where nothing close to the requested lead time exists
        (e.g. a "5 days ahead" request for an hour only 30h out).

    Returns an extra `used_lead_hours` column so the caller can show which
    lead time was actually achieved per hour (it may not exactly match the
    request).
    """
    empty = pd.DataFrame(columns=["forecast_ts", "pop", "rain_1h", "used_lead_hours"])
    if hourly.empty:
        return empty

    df = prep_hourly_forecast(hourly)
    day_df = df[(df["location"] == location) & (df["target_date"] == target_date)]
    if day_df.empty:
        return empty

    if lead_hours is None:
        latest = (
            day_df.sort_values("fetched_at")
            .groupby("forecast_ts", as_index=False)
            .tail(1)
            .sort_values("forecast_ts")
        )
        latest = latest.rename(columns={"lead_hours": "used_lead_hours"})
        return latest[["forecast_ts", "pop", "rain_1h", "used_lead_hours"]].reset_index(drop=True)

    day_df = day_df.copy()
    day_df["lead_diff"] = (day_df["lead_hours"] - lead_hours).abs()
    idx = day_df.groupby("forecast_ts")["lead_diff"].idxmin()
    picked = day_df.loc[idx].sort_values("forecast_ts")
    picked = picked.rename(columns={"lead_hours": "used_lead_hours"})
    return picked[["forecast_ts", "pop", "rain_1h", "used_lead_hours"]].reset_index(drop=True)


def forecast_stability_index(hourly: pd.DataFrame) -> pd.DataFrame:
    """For each location, how much predicted PoP for the same target hour
    changes across successive forecast fetches. Lower revision = more stable.
    Returned as a 0-100 score where 100 = most stable location in the set.
    """
    if hourly.empty:
        return pd.DataFrame(columns=["location", "avg_pop_revision", "stability_index"])

    df = prep_hourly_forecast(hourly).sort_values(["location", "forecast_dt", "fetched_at"])
    df["pop_prev"] = df.groupby(["location", "forecast_dt"])["pop"].shift(1)
    df["pop_delta"] = (df["pop"] - df["pop_prev"]).abs()

    result = (
        df.dropna(subset=["pop_delta"])
        .groupby("location")["pop_delta"]
        .mean()
        .reset_index()
        .rename(columns={"pop_delta": "avg_pop_revision"})
    )
    if result.empty:
        return result.assign(stability_index=pd.Series(dtype=float))

    max_rev = result["avg_pop_revision"].max() or 1
    result["stability_index"] = (100 * (1 - result["avg_pop_revision"] / max_rev)).round(1)
    return result


def microclimate_ranking(current_weather: pd.DataFrame) -> pd.DataFrame:
    """Rank locations by day-to-day variability (temp + humidity spread)."""
    if current_weather.empty:
        return pd.DataFrame(columns=["location", "temp_std", "humidity_std", "avg_temp", "volatility_score"])

    cw = prep_current_weather(current_weather)
    grouped = cw.groupby("location").agg(
        temp_std=("temp_c", "std"),
        humidity_std=("humidity", "std"),
        avg_temp=("temp_c", "mean"),
    ).reset_index()
    grouped[["temp_std", "humidity_std", "avg_temp"]] = grouped[
        ["temp_std", "humidity_std", "avg_temp"]
    ].round(2)
    grouped["volatility_score"] = (
        grouped["temp_std"].fillna(0) + grouped["humidity_std"].fillna(0) / 10
    ).round(2)
    return grouped.sort_values("volatility_score", ascending=False).reset_index(drop=True)


def forecast_confidence_score(pr_metrics: dict | None, stability_row: pd.Series | None) -> float | None:
    """Blend rain-forecast F1 with forecast stability into one 0-100 score."""
    if pr_metrics is None:
        return None
    f1_component = pr_metrics.get("f1", 0) * 100
    stability_component = (
        stability_row["stability_index"] if stability_row is not None else 50
    )
    return round(0.6 * f1_component + 0.4 * stability_component, 1)
