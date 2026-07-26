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
    forecast_confidence_score,
    forecast_stability_index,
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

            # --- Click-through detail: clicking a future bar shows the
            # hour-by-hour forecast for that day. Past/future bars sit at
            # non-overlapping x positions, so any clicked point that lands
            # on a future date can only have come from the future-bar trace.
            future_dates = set(future_df["target_date"]) if not future_df.empty else set()
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
                if parsed in future_dates:
                    clicked_date = parsed
                    break

            st.divider()
            if clicked_date is not None:
                st.markdown(f"**Hourly detail — {location_choice}, {clicked_date:%a %b %d}**")
                detail = hourly_detail_for_day(hourly, location_choice, clicked_date, lead_hours)
                if detail.empty:
                    st.caption("No hourly breakdown available for this day yet.")
                else:
                    detail_fig = go.Figure()
                    detail_fig.add_trace(go.Bar(
                        x=detail["forecast_ts"].dt.strftime("%H:%M"),
                        y=detail["pop"] * 100,
                        marker_color=[
                            "#d62728" if p >= threshold else "#c7c7c7" for p in detail["pop"]
                        ],
                    ))
                    detail_fig.update_layout(
                        yaxis_title="Rain probability (%)", xaxis_title="Hour of day",
                        height=280, margin=dict(t=20),
                    )
                    st.plotly_chart(detail_fig, use_container_width=True)

                    rain_hours = detail[detail["pop"] >= threshold]
                    if not rain_hours.empty:
                        hour_list = ", ".join(rain_hours["forecast_ts"].dt.strftime("%H:%M"))
                        st.caption(f"Rain most likely around: {hour_list}")
                    else:
                        st.caption("No hours currently cross the rain probability threshold.")

                    # Be transparent when the requested lead time couldn't be
                    # matched and we fell back to whatever was closest.
                    if lead_hours is not None:
                        off_target = (detail["used_lead_hours"] - lead_hours).abs() > 6
                        if off_target.any():
                            st.caption(
                                f"Note: some hours this far out don't have a snapshot near "
                                f"'{lead_label}' -- showing the closest available forecast "
                                f"for those hours instead."
                            )
            else:
                st.caption("Click a bar on the right (future) side of the chart to see the hour-by-hour forecast for that day.")

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
