from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from db import (
    LOCATIONS,
    get_current_weather_trend,
    get_current_weather_window,
    get_daily_forecast_latest,
    get_hourly_forecast_window,
    get_latest_current_weather,
)
from metrics import (
    RAIN_POP_THRESHOLD_DEFAULT,
    build_14day_timeline,
    build_rain_accuracy_table,
    daily_detail_for_day,
    forecast_confidence_score,
    forecast_stability_index,
    hourly_actual_for_day,
    hourly_detail_for_day,
    microclimate_ranking,
    precision_recall,
)

st.set_page_config(page_title="Microclimate forecast dashboard", layout="wide")
st.title("Microclimate forecast dashboard")

LEAD_OPTIONS = {
    "Latest available": None,
    "1 day ahead": 24,
    "2 days ahead": 48,
    "3 days ahead": 72,
    "5 days ahead": 120,
}

tab_overview, tab_rain, tab_rank, tab_summary = st.tabs(
    ["Overview", "Rain forecast accuracy", "Rankings & stability", "Daily summary"]
)

# ---------------------------------------------------------------- Overview --
with tab_overview:
    st.subheader("Current conditions")
    latest = get_latest_current_weather()

    if latest.empty:
        st.info("No current_weather data yet — check the collector is running.")
    else:
        cols = st.columns(len(LOCATIONS))
        for col, loc in zip(cols, LOCATIONS):
            row = latest[latest["location"] == loc]
            with col:
                st.markdown(f"**{loc}**")
                if row.empty:
                    st.caption("No data yet")
                    continue
                r = row.iloc[0]
                st.metric("Temperature", f"{r['temp_c']:.1f} °C" if pd.notna(r["temp_c"]) else "—")
                if pd.notna(r["feels_like"]):
                    st.caption(f"Feels like {r['feels_like']:.1f} °C")
                if pd.notna(r["humidity"]):
                    st.caption(f"Humidity {r['humidity']:.0f}%")
                if pd.notna(r["wind_speed"]):
                    st.caption(f"Wind {r['wind_speed']:.1f} m/s")
                st.caption("🌧️ Raining now" if pd.notna(r["rain_1h"]) else "No rain right now")
                st.caption(f"Updated {r['fetched_at']:%Y-%m-%d %H:%M} UTC")

    st.subheader("72-hour trend")
    trend = get_current_weather_trend(days_back=3)
    if trend.empty:
        st.info("Not enough data yet for a trend chart.")
    else:
        fig = go.Figure()
        for loc in LOCATIONS:
            sub = trend[trend["location"] == loc]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(x=sub["fetched_at"], y=sub["temp_c"], name=loc, mode="lines"))
        fig.update_layout(yaxis_title="Temperature (°C)", height=350, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------- Rain accuracy --
with tab_rain:
    st.subheader("14-day rain timeline")
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        location_choice = st.selectbox("Location", LOCATIONS)
    with c2:
        lead_label = st.selectbox("Forecast made", list(LEAD_OPTIONS.keys()))
    with c3:
        threshold = st.slider("Rain probability threshold", 0.1, 0.9, RAIN_POP_THRESHOLD_DEFAULT, 0.05)

    hourly = get_hourly_forecast_window(days_back=10)
    cw = get_current_weather_window(days_back=10)
    daily = get_daily_forecast_latest()

    if hourly.empty or cw.empty:
        st.info("Not enough collected data yet to build the timeline.")
    else:
        lead_hours = LEAD_OPTIONS[lead_label]
        timeline = build_14day_timeline(hourly, cw, daily, lead_hours, threshold)
        loc_timeline = timeline[timeline["location"] == location_choice].sort_values("target_date")

        if loc_timeline.empty:
            st.info("No matching forecast/actual rows yet for this location.")
        else:
            past_df = loc_timeline[loc_timeline["past"] == True]  # noqa: E712
            future_df = loc_timeline[loc_timeline["past"] == False]  # noqa: E712

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=past_df["target_date"], y=past_df["pop"] * 100,
                name="Forecast PoP (past)", marker_color="#1f77b4",
            ))
            fig.add_trace(go.Bar(
                x=future_df["target_date"], y=future_df["pop"] * 100,
                name="Forecast PoP (future)", marker_color="#c7c7c7",
            ))

            hits = past_df[past_df["outcome"] == "hit"]
            false_alarms = past_df[past_df["outcome"] == "false_alarm"]
            fig.add_trace(go.Scatter(
                x=hits["target_date"], y=hits["pop"] * 100 + 8, mode="markers",
                name="Rain observed", marker=dict(symbol="circle", size=11, color="#1f77b4"),
            ))
            fig.add_trace(go.Scatter(
                x=false_alarms["target_date"], y=false_alarms["pop"] * 100 + 8, mode="markers",
                name="Forecasted, didn't rain", marker=dict(symbol="triangle-up", size=12, color="#d62728"),
            ))

            fig.update_layout(
                barmode="overlay", yaxis_title="Rain probability (%)",
                height=420, margin=dict(t=20), legend=dict(orientation="h", y=1.15),
            )
            event = st.plotly_chart(
                fig, use_container_width=True, key="rain_timeline_chart", on_select="rerun"
            )

            # --- Click-through detail for any day (past or future) ---
            # Past days: show the archived hourly forecast overlaid with
            #   actual observed rain (both exist).
            # Near-future days (~0-2 out): hourly_forecast still has data,
            #   show the forecast only (nothing observed yet).
            # Far-future days (~3-7 out): hourly_forecast has NOTHING that
            #   far ahead -- fall back to the single daily_forecast value
            #   instead of showing an empty chart.
            all_dates = set(loc_timeline["target_date"])
            clicked_date = None
            try:
                points = event["selection"]["points"]
            except Exception:
                points = []
            for pt in points:
                raw_x = pt.get("x") if isinstance(pt, dict) else None
                if raw_x is None:
                    continue
                try:
                    parsed = pd.to_datetime(raw_x).date()
                except Exception:
                    continue
                if parsed in all_dates:
                    clicked_date = parsed
                    break

            st.divider()
            if clicked_date is None:
                st.caption("Click a bar above to see the hour-by-hour detail for that day.")
            else:
                is_past = clicked_date in set(past_df["target_date"])
                st.markdown(f"**Detail — {location_choice}, {clicked_date:%a %b %d}**")

                hourly_fc = hourly_detail_for_day(hourly, location_choice, clicked_date, lead_hours)
                actual_hourly = (
                    hourly_actual_for_day(cw, location_choice, clicked_date)
                    if is_past else pd.DataFrame()
                )

                if not hourly_fc.empty:
                    detail_fig = go.Figure()
                    detail_fig.add_trace(go.Bar(
                        x=hourly_fc["forecast_ts"].dt.strftime("%H:%M"),
                        y=hourly_fc["pop"] * 100,
                        name="Forecast PoP",
                        marker_color=[
                            "#d62728" if p >= threshold else "#c7c7c7" for p in hourly_fc["pop"]
                        ],
                    ))

                    if not actual_hourly.empty:
                        actual_hourly = actual_hourly.copy()
                        actual_hourly["hour_label"] = actual_hourly["fetched_at"].dt.floor("h").dt.strftime("%H:%M")
                        rained = actual_hourly[actual_hourly["rain_1h"].notna()]
                        if not rained.empty:
                            # anchor markers just above the matching bar, where one exists
                            bar_lookup = dict(zip(
                                hourly_fc["forecast_ts"].dt.strftime("%H:%M"), hourly_fc["pop"] * 100
                            ))
                            marker_y = [bar_lookup.get(h, 0) + 8 for h in rained["hour_label"]]
                            detail_fig.add_trace(go.Scatter(
                                x=rained["hour_label"], y=marker_y, mode="markers",
                                name="Rain observed",
                                marker=dict(symbol="circle", size=10, color="#1f77b4"),
                            ))

                    detail_fig.update_layout(
                        yaxis_title="Rain probability (%)", xaxis_title="Hour of day",
                        height=280, margin=dict(t=20),
                        legend=dict(orientation="h", y=1.2) if not actual_hourly.empty else None,
                    )
                    st.plotly_chart(detail_fig, use_container_width=True)

                    rain_hours = hourly_fc[hourly_fc["pop"] >= threshold]
                    if not rain_hours.empty:
                        hour_list = ", ".join(rain_hours["forecast_ts"].dt.strftime("%H:%M"))
                        st.caption(f"Rain most likely around: {hour_list}")
                    else:
                        st.caption("No hours currently cross the rain probability threshold.")

                    if lead_hours is not None:
                        off_target = (hourly_fc["used_lead_hours"] - lead_hours).abs() > 6
                        if off_target.any():
                            st.caption(
                                f"Note: some hours this far out don't have a snapshot near "
                                f"'{lead_label}' -- showing the closest available forecast "
                                f"for those hours instead."
                            )

                    if is_past and actual_hourly.empty:
                        st.caption("No actual observations recorded for this day.")

                elif not is_past:
                    # Far-future day: hourly_forecast doesn't reach this far,
                    # fall back to the single daily_forecast value.
                    daily_row = daily_detail_for_day(daily, location_choice, clicked_date)
                    if daily_row is None:
                        st.caption("No forecast data available yet for this day.")
                    else:
                        st.caption(
                            "Hourly breakdown isn't available this far out (only the next "
                            "~2 days have hourly detail) — showing the daily forecast instead."
                        )
                        d1, d2, d3 = st.columns(3)
                        pop_val = daily_row.get("pop")
                        d1.metric("Rain probability", f"{pop_val*100:.0f}%" if pd.notna(pop_val) else "—")
                        rain_mm = daily_row.get("rain_mm")
                        d2.metric("Expected rainfall", f"{rain_mm:.1f} mm" if pd.notna(rain_mm) else "0 mm")
                        tmin, tmax = daily_row.get("temp_min"), daily_row.get("temp_max")
                        if pd.notna(tmin) and pd.notna(tmax):
                            d3.metric("Temp range", f"{tmin:.0f}–{tmax:.0f}°C")

                else:
                    # Past day with no archived hourly forecast (collector
                    # likely wasn't running yet that far back).
                    st.caption("No hourly forecast archive available for this day.")
                    if not actual_hourly.empty:
                        rained = actual_hourly[actual_hourly["rain_1h"].notna()]
                        if not rained.empty:
                            hour_list = ", ".join(rained["fetched_at"].dt.strftime("%H:%M"))
                            st.caption(f"Actual rain observed around: {hour_list}")
                        else:
                            st.caption("No rain was observed this day.")
                    else:
                        st.caption("No actual observations recorded for this day either.")

            accuracy_table = build_rain_accuracy_table(hourly, cw, lead_hours, threshold)
            loc_accuracy = accuracy_table[accuracy_table["location"] == location_choice]
            pr = precision_recall(loc_accuracy)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision", f"{pr['precision']*100:.0f}%")
            m2.metric("Recall", f"{pr['recall']*100:.0f}%")
            m3.metric("F1", f"{pr['f1']*100:.0f}%")
            m4.metric("False alarms (7d)", pr["fp"])

# --------------------------------------------------------- Rank/stability --
with tab_rank:
    st.subheader("Microclimate ranking")
    cw_all = get_current_weather_window(days_back=10)
    if cw_all.empty:
        st.info("Not enough data yet.")
    else:
        ranking = microclimate_ranking(cw_all)
        st.dataframe(ranking, use_container_width=True, hide_index=True)
        st.caption("Higher volatility score = more day-to-day swing in temp/humidity.")

    st.subheader("Forecast stability index")
    hourly_all = get_hourly_forecast_window(days_back=10)
    if hourly_all.empty:
        st.info("Not enough data yet.")
    else:
        stability = forecast_stability_index(hourly_all)
        st.dataframe(stability, use_container_width=True, hide_index=True)
        st.caption("100 = predictions barely change between forecast runs; lower = forecasts revised more.")

# -------------------------------------------------------------- Summary --
with tab_summary:
    st.subheader("Daily summary report")
    hourly_all = get_hourly_forecast_window(days_back=10)
    cw_all = get_current_weather_window(days_back=10)

    if hourly_all.empty or cw_all.empty:
        st.info("Not enough data yet to generate a summary.")
    else:
        stability = forecast_stability_index(hourly_all)
        lines = [f"### Daily summary — {datetime.now():%Y-%m-%d}", ""]

        for loc in LOCATIONS:
            acc = build_rain_accuracy_table(hourly_all, cw_all, None, RAIN_POP_THRESHOLD_DEFAULT)
            loc_acc = acc[acc["location"] == loc]
            pr = precision_recall(loc_acc)

            stab_row_df = stability[stability["location"] == loc]
            stab_row = stab_row_df.iloc[0] if not stab_row_df.empty else None
            score = forecast_confidence_score(pr, stab_row)

            lines.append(f"**{loc}**")
            lines.append(
                f"- Rain forecast (last 7 days, latest-available forecast): "
                f"precision {pr['precision']*100:.0f}%, recall {pr['recall']*100:.0f}%, "
                f"{pr['fp']} false alarm(s)"
            )
            if score is not None:
                lines.append(f"- Forecast confidence score: {score}/100")
            if stab_row is not None:
                lines.append(f"- Stability index: {stab_row['stability_index']:.0f}/100")
            lines.append("")

        report_md = "\n".join(lines)
        st.markdown(report_md)
        st.download_button(
            "Download report",
            report_md,
            file_name=f"daily_summary_{datetime.now():%Y%m%d}.md",
        )
