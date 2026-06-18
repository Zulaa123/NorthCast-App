"""
Northern Ontario Winter Road Risk Dashboard
Streamlit + Folium interactive map of predicted lake states for 30 First Nations communities.
Fake predictions are used until the real MLP model (ice_model.pkl) is connected.
"""

import datetime
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Northern Ontario Winter Road Risk",
    page_icon="❄️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Community data — 30 Northern Ontario First Nations (lat/lon ± ~1 km accuracy)
# ---------------------------------------------------------------------------
COMMUNITIES = [
    {"name": "Attawapiskat First Nation",          "lat": 52.926, "lon": -82.432, "lake": "Attawapiskat River"},
    {"name": "Kashechewan First Nation",            "lat": 52.282, "lon": -81.678, "lake": "Albany River"},
    {"name": "Fort Albany First Nation",            "lat": 52.237, "lon": -81.596, "lake": "Albany River"},
    {"name": "Moose Cree First Nation",             "lat": 51.282, "lon": -80.645, "lake": "Moose River"},
    {"name": "Fort Severn First Nation",            "lat": 55.986, "lon": -87.616, "lake": "Severn River"},
    {"name": "Peawanuck (Weenusk) First Nation",    "lat": 54.983, "lon": -85.433, "lake": "Winisk River"},
    {"name": "Eabametoong (Fort Hope) First Nation","lat": 51.535, "lon": -87.899, "lake": "Lake Eabametoong"},
    {"name": "Nibinamik (Summer Beaver) First Nation","lat": 52.736, "lon": -88.699, "lake": "Summer Beaver Lake"},
    {"name": "Marten Falls (Ogoki) First Nation",  "lat": 51.791, "lon": -86.413, "lake": "Albany River"},
    {"name": "Webequie First Nation",               "lat": 52.960, "lon": -87.369, "lake": "Winisk Lake"},
    {"name": "Neskantaga First Nation",             "lat": 52.234, "lon": -87.977, "lake": "Attwood Lake"},
    {"name": "Kitchenuhmaykoosib Inninuwug",        "lat": 53.832, "lon": -89.868, "lake": "Big Trout Lake"},
    {"name": "Mishkeegogamang First Nation",        "lat": 51.194, "lon": -89.968, "lake": "Osnaburgh Lake"},
    {"name": "Pikangikum First Nation",             "lat": 51.822, "lon": -93.978, "lake": "Pikangikum Lake"},
    {"name": "Cat Lake First Nation",               "lat": 51.717, "lon": -91.800, "lake": "Cat Lake"},
    {"name": "Sandy Lake First Nation",             "lat": 53.022, "lon": -93.344, "lake": "Sandy Lake"},
    {"name": "Deer Lake First Nation",              "lat": 52.619, "lon": -94.067, "lake": "Deer Lake"},
    {"name": "North Spirit Lake First Nation",      "lat": 52.491, "lon": -93.564, "lake": "North Spirit Lake"},
    {"name": "Sachigo Lake First Nation",           "lat": 53.884, "lon": -92.190, "lake": "Sachigo Lake"},
    {"name": "Muskrat Dam Lake First Nation",       "lat": 53.001, "lon": -91.762, "lake": "Muskrat Dam Lake"},
    {"name": "Kasabonika Lake First Nation",        "lat": 53.525, "lon": -88.625, "lake": "Kasabonika Lake"},
    {"name": "Kingfisher Lake First Nation",        "lat": 53.012, "lon": -89.856, "lake": "Kingfisher Lake"},
    {"name": "Wapekeka (Angling Lake) First Nation","lat": 53.847, "lon": -89.579, "lake": "Angling Lake"},
    {"name": "Wawakapewin (Long Dog) First Nation", "lat": 53.951, "lon": -89.089, "lake": "Long Dog Lake"},
    {"name": "Lac Seul First Nation",               "lat": 50.430, "lon": -91.990, "lake": "Lac Seul"},
    {"name": "Poplar Hill First Nation",            "lat": 52.116, "lon": -94.253, "lake": "Poplar Hill Lake"},
    {"name": "Bearskin Lake First Nation",          "lat": 53.651, "lon": -90.997, "lake": "Bearskin Lake"},
    {"name": "Constance Lake First Nation",         "lat": 49.801, "lon": -83.691, "lake": "Constance Lake"},
    {"name": "Aroland First Nation",                "lat": 49.815, "lon": -87.016, "lake": "Kenogami River"},
    {"name": "Ginoogaming First Nation",            "lat": 49.801, "lon": -86.701, "lake": "Long Lake"},
]

# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------
STATES = ["Frozen", "Unstable", "Open"]

STATE_COLOR = {
    "Frozen":   "#2196F3",  # blue
    "Unstable": "#FF9800",  # orange
    "Open":     "#F44336",  # red
}
STATE_EMOJI = {"Frozen": "❄️", "Unstable": "⚠️", "Open": "🌊"}

# Base probability weights [Frozen, Unstable, Open] per scenario
SCENARIO_WEIGHTS = {
    "Optimistic (SSP1-2.6)":     [0.72, 0.20, 0.08],
    "Baseline (SSP2-4.5)":       [0.55, 0.30, 0.15],
    "High Emissions (SSP5-8.5)": [0.38, 0.35, 0.27],
}

SCENARIOS = list(SCENARIO_WEIGHTS.keys())

# Seasonal base weights [Frozen, Unstable, Open] by month — Northern Ontario
MONTH_WEIGHTS = {
    1:  [0.80, 0.15, 0.05],
    2:  [0.78, 0.17, 0.05],
    3:  [0.55, 0.30, 0.15],
    4:  [0.20, 0.45, 0.35],
    5:  [0.05, 0.20, 0.75],
    6:  [0.02, 0.05, 0.93],
    7:  [0.01, 0.04, 0.95],
    8:  [0.01, 0.04, 0.95],
    9:  [0.02, 0.08, 0.90],
    10: [0.05, 0.20, 0.75],
    11: [0.35, 0.40, 0.25],
    12: [0.70, 0.22, 0.08],
}

# Per-scenario additive shift on top of seasonal base
SCENARIO_SHIFT = {
    "Optimistic (SSP1-2.6)":     [ 0.08,  0.00, -0.08],
    "Baseline (SSP2-4.5)":       [ 0.00,  0.00,  0.00],
    "High Emissions (SSP5-8.5)": [-0.08,  0.00,  0.08],
}

FORECAST_WINDOW = 7  # days ahead the Open-Meteo free tier provides

# ---------------------------------------------------------------------------
# Fake prediction engine
# Deterministic per (community, year, scenario) using MD5 seeding so results
# are stable across Python runs (unlike hash(), which uses PYTHONHASHSEED).
# Replace fake_predict() with a call to ice_model.pkl when ready.
# ---------------------------------------------------------------------------

def _seed(name: str, date: datetime.date, scenario: str) -> int:
    key = f"{name}|{date.isoformat()}|{scenario}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def fake_predict(date: datetime.date, scenario: str) -> pd.DataFrame:
    # Seasonal base + scenario shift + long-term climate drift
    base = np.array(MONTH_WEIGHTS[date.month], dtype=float)
    base += np.array(SCENARIO_SHIFT[scenario], dtype=float)
    drift = (date.year - 2025) * 0.025
    adjusted = base + np.array([-drift * 1.5, drift * 0.5, drift])
    adjusted = np.clip(adjusted, 0.01, 0.99)
    adjusted /= adjusted.sum()

    rows = []
    for c in COMMUNITIES:
        rng = np.random.default_rng(_seed(c["name"], date, scenario))
        state = rng.choice(STATES, p=adjusted)

        risk_lo, risk_hi = {"Frozen": (5, 35), "Unstable": (40, 70), "Open": (65, 95)}[state]
        risk = round(float(rng.uniform(risk_lo, risk_hi)), 1)

        ice_cm = (
            round(float(rng.uniform(40, 110)), 1) if state == "Frozen"
            else round(float(rng.uniform(0, 25)), 1)
        )
        temp_anom = round(float(rng.uniform(-3, 2) + (date.year - 2025) * 0.07), 2)
        snow_pct = round(float(rng.uniform(20, 95)), 1)
        road_days = (
            int(rng.uniform(40, 100)) if state == "Frozen"
            else int(rng.uniform(5, 40)) if state == "Unstable"
            else int(rng.uniform(0, 15))
        )

        rows.append({**c, "state": state, "risk_score": risk,
                     "ice_thickness_cm": ice_cm, "temp_anomaly_c": temp_anom,
                     "snow_cover_pct": snow_pct, "road_open_days": road_days})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Real-time weather — Open-Meteo (free, no API key required)
# Fetches today ± 7 days forecast + 92 days of history for cum-FDD calculation.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_weather(today_iso: str) -> dict:
    """Fetch daily forecast + history for all 30 communities in parallel."""
    def _one(c):
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": c["lat"],
                    "longitude": c["lon"],
                    "daily": [
                        "temperature_2m_mean",
                        "windspeed_10m_max",
                        "precipitation_sum",
                        "snowfall_sum",
                        "snow_depth",
                    ],
                    "wind_speed_unit": "ms",
                    "timezone": "America/Toronto",
                    "past_days": 92,
                    "forecast_days": FORECAST_WINDOW,
                },
                timeout=10,
            )
            r.raise_for_status()
            return c["name"], r.json()
        except Exception:
            return c["name"], None

    results = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        for name, data in pool.map(_one, COMMUNITIES):
            results[name] = data
    return results


def _cum_fdd(daily: dict, up_to: datetime.date) -> float:
    """Sum freezing degree-days from Oct 1 of the current freeze season up to `up_to`."""
    year = up_to.year if up_to.month >= 10 else up_to.year - 1
    season_start = datetime.date(year, 10, 1)
    fdd = 0.0
    for d_str, t in zip(daily.get("time", []), daily.get("temperature_2m_mean", [])):
        d = datetime.date.fromisoformat(d_str)
        if d < season_start:
            continue
        if d > up_to:
            break
        if t is not None and t < 0:
            fdd -= t  # t is negative, so -t adds a positive FDD contribution
    return round(fdd, 1)


def realtime_predict(weather_cache: dict, target_date: datetime.date, scenario: str):
    """Build a prediction DataFrame from live weather data.
    Falls back to fake_predict for any community where the fetch failed."""
    _fake_df = None
    rows = []
    n_failed = 0

    for c in COMMUNITIES:
        data = weather_cache.get(c["name"])
        row = None

        if data is not None:
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            target_iso = target_date.isoformat()

            if target_iso in dates:
                idx = dates.index(target_iso)
                temp_c        = daily["temperature_2m_mean"][idx] or 0.0
                wind_ms       = daily["windspeed_10m_max"][idx] or 0.0
                precip_mm     = daily["precipitation_sum"][idx] or 0.0
                snow_depth_cm = (daily.get("snow_depth", [None] * len(dates))[idx] or 0.0) * 100.0
                cum_fdd_val   = _cum_fdd(daily, target_date)

                state, score = whatif_predict(temp_c, wind_ms, snow_depth_cm, precip_mm, cum_fdd_val)
                lo, hi = {"Frozen": (5, 35), "Unstable": (40, 70), "Open": (65, 95)}[state]
                risk = round(lo + (score / 100.0) * (hi - lo), 1)
                # Stefan's-law ice thickness estimate: h_cm ≈ 3.4 × sqrt(FDD)
                ice_cm = round(min(130.0, 3.4 * (max(0.0, cum_fdd_val) ** 0.5)), 1)

                row = {
                    **c,
                    "state":            state,
                    "risk_score":       risk,
                    "ice_thickness_cm": ice_cm,
                    "temp_anomaly_c":   round(temp_c - (-15.0), 2),
                    "snow_cover_pct":   min(100.0, round(snow_depth_cm, 1)),
                    "road_open_days":   {"Frozen": 70, "Unstable": 25, "Open": 5}[state],
                    # Live weather fields shown in popup
                    "rt_temp_c":        round(temp_c, 1),
                    "rt_wind_ms":       round(wind_ms, 1),
                    "rt_precip_mm":     round(precip_mm, 1),
                    "rt_snow_depth_cm": round(snow_depth_cm, 1),
                    "rt_cum_fdd":       cum_fdd_val,
                }

        if row is None:
            n_failed += 1
            if _fake_df is None:
                _fake_df = fake_predict(target_date, scenario)
            fake_row = _fake_df[_fake_df["name"] == c["name"]].iloc[0].to_dict()
            fake_row["rt_temp_c"] = None  # marks this as a fallback row
            row = fake_row

        rows.append(row)

    return pd.DataFrame(rows), n_failed


# ---------------------------------------------------------------------------
# What-if predictor — rule-based scoring from field conditions
# ---------------------------------------------------------------------------

def whatif_predict(temp_c: float, wind_ms: float, snow_depth_cm: float,
                   precip_mm: float, cum_fdd: float):
    """Returns (state, condition_score 0–100) from ERA5-style meteorological inputs."""
    score = 0.0
    # Cumulative FDD: Stefan's-law driver of ice growth, up to 50 pts
    score += min(50.0, cum_fdd * 50.0 / 400.0)
    # Air temperature: up to 25 pts; -30°C → ~25 pts, 0°C → ~5.5 pts, +10°C → 0 pts
    score += max(0.0, min(25.0, (-temp_c + 10.0) * 25.0 / 45.0))
    # Snow depth: insulation slows melt, up to 15 pts
    score += min(15.0, snow_depth_cm * 15.0 / 100.0)
    # Wind: surface turbulence destabilises forming ice, up to -10 pts
    score -= min(10.0, wind_ms * 10.0 / 20.0)
    # Precipitation: rainfall accelerates melt, up to -10 pts
    score -= min(10.0, precip_mm * 10.0 / 20.0)

    score = max(0.0, min(100.0, score))

    if score >= 62:
        return "Frozen", score
    elif score >= 30:
        return "Unstable", score
    else:
        return "Open", score


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

def build_map(df: pd.DataFrame, date: datetime.date, scenario: str,
              focus_community: str = None) -> folium.Map:
    center = [52.5, -89.5]
    zoom = 6

    if focus_community:
        fc = df[df["name"] == focus_community]
        if not fc.empty:
            center = [float(fc.iloc[0]["lat"]), float(fc.iloc[0]["lon"])]
            zoom = 9

    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")

    for _, row in df.iterrows():
        is_focused = bool(focus_community and row["name"] == focus_community)
        color = STATE_COLOR[row["state"]]
        emoji = STATE_EMOJI[row["state"]]

        popup_html = f"""
        <div style="font-family:sans-serif;min-width:210px;font-size:13px;">
          <b style="font-size:14px;">{row['name']}</b><br>
          <span style="color:#555;">{row['lake']}</span>
          <hr style="margin:6px 0;">
          <span style="font-size:15px;">{emoji}</span>
          <b style="color:{color};">&nbsp;{row['state']}</b>
          &nbsp;· Risk&nbsp;<b>{row['risk_score']}/100</b>
          <hr style="margin:6px 0;">
          <table style="width:100%;border-collapse:collapse;">"""

        has_rt = "rt_temp_c" in row.index and row["rt_temp_c"] is not None
        if has_rt:
            popup_html += f"""
            <tr><td>Air temp</td><td align="right"><b>{row['rt_temp_c']} °C</b></td></tr>
            <tr><td>Wind speed</td><td align="right"><b>{row['rt_wind_ms']} m/s</b></td></tr>
            <tr><td>Snow depth</td><td align="right"><b>{row['rt_snow_depth_cm']} cm</b></td></tr>
            <tr><td>Precipitation</td><td align="right"><b>{row['rt_precip_mm']} mm</b></td></tr>
            <tr><td>Cum. FDD</td><td align="right"><b>{row['rt_cum_fdd']}</b></td></tr>
            <tr><td>Est. ice (Stefan)</td><td align="right"><b>{row['ice_thickness_cm']} cm</b></td></tr>"""
            data_badge = '<span style="color:#2e7d32;font-weight:bold;">⬤ Live · Open-Meteo</span>'
        else:
            popup_html += f"""
            <tr><td>Ice thickness</td><td align="right"><b>{row['ice_thickness_cm']} cm</b></td></tr>
            <tr><td>Temp anomaly</td><td align="right"><b>{row['temp_anomaly_c']:+.2f}°C</b></td></tr>
            <tr><td>Snow cover</td><td align="right"><b>{row['snow_cover_pct']}%</b></td></tr>
            <tr><td>Est. road-open days</td><td align="right"><b>{row['road_open_days']}</b></td></tr>"""
            data_badge = f'<span style="color:#999;">{date.strftime("%b %d, %Y")} · {scenario}</span>'

        popup_html += f"""
          </table>
          <hr style="margin:6px 0;">
          <span style="font-size:10px;">{data_badge}</span>
        </div>
        """

        # Outer ring highlight for the focused marker
        if is_focused:
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=22,
                color="#FFD700",
                weight=2.5,
                fill=False,
                opacity=0.85,
            ).add_to(m)

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=16 if is_focused else 11,
            color="#FFD700" if is_focused else "white",
            weight=3 if is_focused else 1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.95 if is_focused else 0.88,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{row['name']} — {row['state']} (Risk {row['risk_score']})",
        ).add_to(m)

    return m


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
today = datetime.date.today()

with st.sidebar:
    st.title("⚙️ Controls")

    # --- Forecast settings ---
    st.subheader("Forecast")
    forecast_date = st.date_input(
        "Forecast Date",
        value=datetime.date(2027, 1, 15),
        min_value=datetime.date(2025, 1, 1),
        max_value=datetime.date(2035, 12, 31),
    )
    scenario = st.selectbox("Climate Scenario", SCENARIOS, index=1)

    st.divider()

    # --- Real-time mode ---
    realtime_mode = st.toggle(
        "🌐 Real-Time Weather",
        value=False,
        help="Fetch live forecast data from Open-Meteo (free). "
             f"Available for today through {(today + datetime.timedelta(days=FORECAST_WINDOW)).strftime('%b %d')}. "
             "Dates outside this window fall back to the scenario model.",
    )
    if realtime_mode:
        rt_end = today + datetime.timedelta(days=FORECAST_WINDOW)
        st.caption(
            f"Live window: **{today.strftime('%b %d')} – {rt_end.strftime('%b %d, %Y')}**  \n"
            "Data: Open-Meteo · cached 1 h"
        )

    st.divider()

    # --- Lake / Community Search ---
    st.subheader("🔍 Find Lake")

    # Options show "Community  ·  Lake" so both names are searchable by typing
    search_options = {f"{c['name']}  ·  {c['lake']}": c["name"] for c in COMMUNITIES}
    search_list = [""] + list(search_options.keys())

    selected_label = st.selectbox(
        "Find lake or community",
        search_list,
        index=0,
        format_func=lambda x: "— type to search —" if x == "" else x,
        label_visibility="collapsed",
    )

    focus_community = search_options.get(selected_label)  # None when placeholder

    if focus_community:
        fc_meta = next(c for c in COMMUNITIES if c["name"] == focus_community)
        st.caption(f"📍 {fc_meta['lake']}")

    st.divider()

    # --- What-if Ice Condition Predictor ---
    st.subheader("🧪 Ice Condition Predictor")
    st.caption("Enter current field conditions to estimate lake state.")

    wi_temp = st.slider("Air Temperature (°C)", -35.0, 10.0, -15.0, 0.5,
                        format="%.1f °C")
    wi_wind = st.slider("Wind Speed (m/s)", 0.0, 25.0, 5.0, 0.5,
                        format="%.1f m/s")
    wi_snow = st.slider("Snow Depth (cm)", 0, 150, 50, 5,
                        format="%d cm")
    wi_precip = st.slider("Total Precipitation (mm)", 0.0, 50.0, 0.0, 0.5,
                          format="%.1f mm",
                          help="Rainfall equivalent — rain accelerates ice melt")
    wi_fdd = st.slider("Cumulative Freezing Degree-Days", 0, 600, 200, 10,
                       format="%d FDD",
                       help="Accumulated cold since freeze-up; drives ice thickness via Stefan's law")

    wi_state, wi_score = whatif_predict(wi_temp, wi_wind, wi_snow, wi_precip, wi_fdd)
    wi_color = STATE_COLOR[wi_state]
    wi_emoji = STATE_EMOJI[wi_state]

    st.markdown(
        f'<div style="background:{wi_color};color:white;padding:10px 12px;'
        f'border-radius:8px;text-align:center;font-size:1.05em;font-weight:bold;'
        f'margin-top:6px;">'
        f'{wi_emoji} &nbsp;Predicted: <span style="font-size:1.15em">{wi_state}</span>'
        f'<br><span style="font-size:0.72em;opacity:0.9;font-weight:normal;">'
        f'Condition score: {wi_score:.0f} / 100</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Legend")
    for state, color in STATE_COLOR.items():
        st.markdown(
            f'<span style="color:{color};font-size:1.3em;">●</span> &nbsp;**{STATE_EMOJI[state]} {state}**',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption(
        "Winter road season: **Nov–Mar**.  \n"
        "Beyond 2030, treat outputs as **climate-risk scenarios**, not exact forecasts."
    )
    st.caption("🔧 **Fake data** — connect `ice_model.pkl` to see real predictions.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Northern Ontario Winter Road Risk Dashboard")
st.markdown(
    f"**{forecast_date.strftime('%b %d, %Y')}** &nbsp;·&nbsp; {scenario} &nbsp;·&nbsp; "
    "Predicted lake states for 30 First Nations communities"
)

# ---------------------------------------------------------------------------
# Compute predictions
# ---------------------------------------------------------------------------
rt_end = today + datetime.timedelta(days=FORECAST_WINDOW)
realtime_active = realtime_mode and (today <= forecast_date <= rt_end)

if realtime_active:
    with st.spinner("Fetching live weather from Open-Meteo…"):
        weather_cache = fetch_all_weather(today.isoformat())
    df, n_failed = realtime_predict(weather_cache, forecast_date, scenario)
    if n_failed:
        st.warning(
            f"{n_failed} of 30 communities fell back to scenario data "
            "(weather fetch failed — check your connection)."
        )
else:
    if realtime_mode:
        st.info(
            f"Real-time data is only available for **{today.strftime('%b %d')} – "
            f"{rt_end.strftime('%b %d, %Y')}**. "
            "Showing scenario forecast for the selected date."
        )
    df = fake_predict(forecast_date, scenario)

# ---------------------------------------------------------------------------
# Layout: map (left) + detail panel (right)
# ---------------------------------------------------------------------------
col_map, col_detail = st.columns([3, 1], gap="medium")

with col_map:
    m = build_map(df, forecast_date, scenario, focus_community)
    map_result = st_folium(m, width="100%", height=620, returned_objects=["last_object_clicked"])

with col_detail:
    st.subheader("Community Detail")

    # Search selection takes precedence; fall back to map click
    selected = None

    if focus_community:
        rows = df[df["name"] == focus_community]
        if not rows.empty:
            selected = rows.iloc[0]
    else:
        clicked = (map_result or {}).get("last_object_clicked")
        if clicked:
            clat, clon = clicked.get("lat"), clicked.get("lng")
            if clat is not None and clon is not None:
                dist = df.apply(lambda r: abs(r.lat - clat) + abs(r.lon - clon), axis=1)
                selected = df.iloc[dist.idxmin()]

    if selected is not None:
        color = STATE_COLOR[selected["state"]]
        emoji = STATE_EMOJI[selected["state"]]
        st.markdown(f"#### {selected['name']}")
        st.markdown(f"*{selected['lake']}*")
        st.markdown(
            f'<div style="background:{color};color:white;padding:10px;border-radius:8px;'
            f'text-align:center;font-size:1.15em;font-weight:bold;">'
            f'{emoji} {selected["state"]}</div>',
            unsafe_allow_html=True,
        )
        st.metric("Risk Score", f"{selected['risk_score']} / 100")
        st.divider()
        col_a, col_b = st.columns(2)
        has_rt_detail = (
            "rt_temp_c" in selected.index
            and selected["rt_temp_c"] is not None
        )
        if has_rt_detail:
            col_a.metric("Air Temp (°C)",    f"{selected['rt_temp_c']}")
            col_b.metric("Wind (m/s)",       f"{selected['rt_wind_ms']}")
            col_a.metric("Snow Depth (cm)",  f"{selected['rt_snow_depth_cm']}")
            col_b.metric("Precip (mm)",      f"{selected['rt_precip_mm']}")
            col_a.metric("Cum. FDD",         f"{selected['rt_cum_fdd']}")
            col_b.metric("Est. Ice (cm)",    f"{selected['ice_thickness_cm']}")
            st.caption("⬤ Live — Open-Meteo")
        else:
            col_a.metric("Ice (cm)",    f"{selected['ice_thickness_cm']}")
            col_b.metric("Snow (%)",    f"{selected['snow_cover_pct']}")
            col_a.metric("Temp Δ (°C)", f"{selected['temp_anomaly_c']:+.2f}")
            col_b.metric("Road days",   f"{selected['road_open_days']}")
    else:
        st.info("Click a community on the map to see its detail.")

    st.divider()

    # State summary counts
    st.subheader(f"Summary — {forecast_date.strftime('%b %d, %Y')}")
    counts = df["state"].value_counts()
    for state in STATES:
        n = counts.get(state, 0)
        color = STATE_COLOR[state]
        pct = round(n / len(df) * 100)
        st.markdown(
            f'<span style="color:{color};font-size:1.1em;">●</span> '
            f'**{STATE_EMOJI[state]} {state}:** {n} &nbsp;({pct}%)',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Data table
# ---------------------------------------------------------------------------
st.divider()
st.subheader("All Communities")

display_cols = {
    "name": "Community", "lake": "Lake / River",
    "state": "State", "risk_score": "Risk Score",
    "ice_thickness_cm": "Ice (cm)", "temp_anomaly_c": "Temp Δ (°C)",
    "snow_cover_pct": "Snow (%)", "road_open_days": "Est. Road Days",
}
table_df = df[list(display_cols)].rename(columns=display_cols).sort_values("Risk Score", ascending=False)

BG = {"Frozen": "#dbeeff", "Unstable": "#fff3cd", "Open": "#ffe0e0"}


def highlight_state(row):
    color = BG.get(row["State"], "")
    return [f"background-color:{color}" for _ in row]


styled = table_df.style.apply(highlight_state, axis=1).format(
    {"Risk Score": "{:.1f}", "Ice (cm)": "{:.1f}",
     "Temp Δ (°C)": "{:+.2f}", "Snow (%)": "{:.1f}"}
)

st.dataframe(styled, use_container_width=True, hide_index=True)
