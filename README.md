# NorthCast AI (Temporary README)

## AI4Good Lab Project

NorthCast AI is a machine learning mapping platform for social good.

It predicts future ice stability on northern lakes to support earlier intervention for northern communities that are increasingly vulnerable to climate-driven transport disruptions.

## Why This Matters

Many northern communities rely on seasonal ice routes for transportation, supply access, and essential services. As climate patterns shift, ice conditions become less predictable, increasing safety and access risks.

By forecasting changing lake ice stability, this project aims to help partners:

- Improve risk awareness and planning
- Support safer, earlier decision-making
- Reduce disruptions to community mobility and logistics

## Current Status

This repository is in early development and this README is temporary.

## Initial Project Goals

- Build a data pipeline for historical and near-real-time environmental inputs
- Train and evaluate ML models for ice stability forecasting
- Provide map-based outputs for decision support
- Communicate uncertainty clearly for operational use

## Repository Contents

- `app.py` - Main application entry point
- `requirements.txt` - Python dependencies

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app with Streamlit:

```bash
streamlit run app.py
```

## How Real-Time Forecasting Works

When the **🌐 Real-Time Weather** toggle is enabled and the selected forecast date falls within today + 7 days, the app fetches live weather data and runs it through a physics-based ice condition scorer. Here is how the pipeline works end-to-end.

### 1. Data Fetch — Open-Meteo API

The app fires 30 HTTP requests in parallel (10 at a time) to the [Open-Meteo](https://open-meteo.com/) free forecast API — one per community. Each request retrieves the following daily variables:

| Variable | Description |
|---|---|
| `temperature_2m_mean` | Daily mean air temperature |
| `windspeed_10m_max` | Max wind speed (m/s) |
| `precipitation_sum` | Total precipitation (mm) |
| `snowfall_sum` | Snowfall (cm water-equivalent) |
| `snow_depth` | Snow depth on ground (m) |

Each request also pulls **92 days of historical data** alongside the 7-day forecast, providing a continuous daily temperature record back to roughly October 1 — the start of the freeze season. Results are cached for **1 hour** so subsequent interactions are instant.

### 2. Cumulative Freezing Degree-Days (FDD)

For each community, the app walks the historical temperature record from **October 1** of the current freeze season up to the selected date and accumulates negative degree-days:

```
FDD += max(0, -T_mean)  for each day where T_mean < 0°C
```

This cumulative cold total is the physical input that drives ice growth via Stefan's law.

### 3. Ice Condition Prediction

Each community's live weather variables are fed into a rule-based scorer (`whatif_predict`):

| Variable | Max contribution | Physical reasoning |
|---|---|---|
| Cumulative FDD | +50 pts | Primary ice-growth driver (Stefan's law) |
| Air temperature | +25 pts | Cold = safer, stable ice |
| Snow depth | +15 pts | Insulation slows melt |
| Wind speed | −10 pts | Turbulence destabilises forming ice |
| Precipitation | −10 pts | Rainfall accelerates melt |

The total score (0–100) maps to one of three states: **Frozen**, **Unstable**, or **Open**. The risk score is placed within the corresponding band (e.g. Frozen → 5–35).

Ice thickness is estimated using **Stefan's law**:

```
h_cm ≈ 3.4 × √FDD
```

This is the standard empirical formula for black ice growth in still water.

### 4. Map Display

The resulting predictions drive the same colour-coded circle markers as the scenario mode. When a community has live data, its map popup switches from scenario fields to the actual weather variables (air temp, wind, snow depth, precipitation, cumulative FDD, estimated ice thickness) and shows a green **⬤ Live · Open-Meteo** badge.

### Known Limitations

- **Rule-based scorer, not ML**: `whatif_predict` is a hand-tuned physics approximation, not the trained `ice_model.pkl`. Results are physically plausible but not calibrated against observed ice conditions.
- **No rain/snow distinction**: `precipitation_sum` includes both rain and snow. Rain is treated as a melt risk even in deep winter, which slightly underestimates safety during heavy-snowfall periods.
- **Seasonal relevance**: Cumulative FDD since October 1 is near zero in summer, so all communities will correctly show "Open" outside of the November–March winter road season.
- **Stefan's law assumptions**: The ice thickness estimate applies to ideal black ice in still, clear water. It overpredicts thickness on rivers and underpredicts on shallow, wind-sheltered lakes.

## Next Steps (Planned)

- Define model inputs and feature engineering workflow
- Add training/evaluation scripts and metrics
- Build a minimal map visualization interface
- Add documentation for setup, data sources, and usage

## License

TBD

## Contact

AI4Good Lab project team
