"""
Northern Ontario Winter Road Risk Dashboard
Streamlit + Folium interactive map of predicted lake states for 30 First Nations communities.
Fake predictions are used until the real MLP model (ice_model.pkl) is connected.
"""

import datetime
import hashlib

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
# What-if predictor — rule-based scoring from field conditions
# ---------------------------------------------------------------------------

def whatif_predict(temp_c: float, ice_cm: float, snow_pct: float, cold_days: int):
    """Returns (state, condition_score 0–100) from manually-entered field variables."""
    score = 0.0
    # Ice thickness: up to 40 pts (100 cm → full points)
    score += min(40.0, ice_cm * 40.0 / 100.0)
    # Temperature: up to 30 pts; -20°C → 30pts, 0°C → ~7pts, +10°C → 0pts
    score += max(0.0, min(30.0, (-temp_c + 10.0) * 30.0 / 45.0))
    # Snow cover: up to 15 pts
    score += snow_pct * 15.0 / 100.0
    # Consecutive cold days: up to 15 pts
    score += cold_days * 15.0 / 90.0

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
          <table style="width:100%;border-collapse:collapse;">
            <tr><td>Ice thickness</td><td align="right"><b>{row['ice_thickness_cm']} cm</b></td></tr>
            <tr><td>Temp anomaly</td><td align="right"><b>{row['temp_anomaly_c']:+.2f}°C</b></td></tr>
            <tr><td>Snow cover</td><td align="right"><b>{row['snow_cover_pct']}%</b></td></tr>
            <tr><td>Est. road-open days</td><td align="right"><b>{row['road_open_days']}</b></td></tr>
          </table>
          <hr style="margin:6px 0;">
          <span style="font-size:10px;color:#999;">{date.strftime('%b %d, %Y')} · {scenario}</span>
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

    wi_temp = st.slider("Air Temperature (°C)", -35.0, 10.0, -10.0, 0.5,
                        format="%.1f °C")
    wi_ice = st.slider("Ice Thickness (cm)", 0, 130, 60, 5,
                       format="%d cm")
    wi_snow = st.slider("Snow Cover (%)", 0, 100, 60, 5,
                        format="%d %%")
    wi_cold_days = st.slider("Consecutive Cold Days", 0, 90, 30, 5,
                             format="%d days")

    wi_state, wi_score = whatif_predict(wi_temp, wi_ice, wi_snow, wi_cold_days)
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
