# hobart-weather-visual
# Microclimate forecast dashboard

Streamlit dashboard built on top of the `collector.py` OpenWeather pipeline
(`current_weather`, `hourly_forecast`, `daily_forecast` tables in Supabase
Postgres) for Hobart, West Moonah, and Saltwater River.

## Tabs

- **Overview** — current conditions per location + 72h temperature trend
- **Rain forecast accuracy** — 14-day timeline (7 days back verified against
  actuals, 7 days ahead forecast), with a selector for which forecast
  snapshot to score against ("Latest available", 1/2/3/5 days ahead),
  a rain-probability threshold slider, and precision/recall/F1
- **Rankings & stability** — microclimate volatility ranking across the
  three locations, and a forecast stability index (how much predictions
  get revised between successive collector runs)
- **Daily summary** — auto-generated markdown report, downloadable

## Local setup

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml with your real Supabase DATABASE_URL
streamlit run streamlit_app.py
```

## Deploying on Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at `streamlit_app.py` in that repo.
3. In the app's **Settings -> Secrets**, add:
   ```toml
   DATABASE_URL = "postgresql://...your Supabase connection string..."
   ```
   Do not put this in the GitHub repo itself — GitHub repo/Actions secrets
   are not read by Streamlit Cloud; only the Secrets box in the Streamlit
   Cloud UI is.

## Notes on the metric definitions

- **Rain observed / predicted**: `rain_1h` and `rain_mm` are `NULL` (not 0)
  when there's no rain in the source data, so "did it rain" is
  `rain_1h IS NOT NULL`, not `> 0`. Predicted rain is `pop >= threshold`
  (default 0.5, adjustable in the UI).
- **Lead time selection**: for a past target day, "1 day ahead" picks the
  `hourly_forecast` row whose `fetched_at` is closest to 24h before that
  day started; "Latest available" picks whichever forecast was freshest
  before the day began. Future days always show the latest forecast, since
  there's nothing to pick a lead time against yet.
- **Future 7-day PoP**: stitched from `hourly_forecast` (covers ~48h ahead)
  and `daily_forecast.pop` (days 3-7), preferring the more granular hourly
  data where both exist for the same day.
- **Stability index**: average absolute change in `pop` for the same
  `forecast_dt` across successive fetches, normalized to 0-100 (100 =
  most stable of the locations shown).
- **Confidence score**: `0.6 * F1(rain accuracy) + 0.4 * stability_index`,
  a simple first-pass blend — tune the weights once you have more history.

## Known limitations / next steps

- Metrics currently look at a fixed 10-day query window (`days_back=10`
  in `db.py`) to keep query size bounded; widen if you want longer
  history views.
- The confidence score formula is a reasonable starting blend, not a
  statistically validated model — revisit once you have enough days of
  data to check whether it actually tracks forecast quality.
- No caching invalidation trigger tied to the collector's run cadence;
  the app currently refreshes cached queries every 5 minutes
  (`ttl=300` in `db.py`).
