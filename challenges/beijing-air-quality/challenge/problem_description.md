# From-Scratch Neural Forecasting of Air Quality at Unmonitored Sites

## Overview

City agencies issue next-day air-quality forecasts, but only where they have monitors. This challenge asks for **next-day, hour-by-hour forecasts of six pollutants (PM2.5, PM10, SO2, NO2, CO, O3) at four sites that have never had a pollutant monitor**. The forecasts must come from the recent readings of the surrounding monitoring network and from weather information.

The data comes from 12 air-quality monitoring sites in one large city, from March 2013 to February 2017. Every site is identified only by an anonymous id (`S01` to `S12`):
- **8 network sites** (S01, S02, S04, S07, S08, S09, S10, S12) have pollutant monitors.
- **4 target sites** (S03, S05, S06, S11) are treated as unmonitored. Their pollutant values are never published, not even in the training period. Only their weather is given.

The challenge combines three shifts:
1. **Forecasting:** each forecast is issued at midnight and covers the next 24 hours (horizons 1–24 h). Only the past is available.
2. **Cold-start locations:** there are no target labels at the sites being forecast. Any model must learn from the network sites and transfer to new locations.
3. **Temporal shift:** training covers March 2013 to February 2016. The test covers March 2016 to February 2017, a later year with its own pollution levels and weather.

## Required From-Scratch Setting

This is a **From Scratch** challenge. Competitive solutions should design and train their own neural network from random initialisation. A good fit is a sequence model (for example a recurrent, convolutional or attention-based encoder–decoder) that reads the 72-hour context and the target-day weather and emits the 24-hour, six-pollutant forecast.

- No pre-trained weights or foundation models of any kind (no pre-trained time-series, language or vision models).
- Classical models such as gradient boosting or linear models may be used as baselines or as auxiliary inputs. The intended main model is a neural network trained from scratch on the provided data.
- The requirement concerns the intended solution setting, not the formal target. The formal target is the measured concentrations described below.

## Evaluation

Submissions are scored with a **site-robust RMSLE**. Lower is better, and 0 is perfect.

```
e_i      = log(1 + p_i) - log(1 + y_i)
RMSLE_s  = sqrt( mean of e_i^2 over all test rows of target site s )
SCORE    = 0.5 * mean over the 4 target sites of RMSLE_s
         + 0.5 * max  over the 4 target sites of RMSLE_s
```

- `y_i` is the measured concentration (µg/m³) and `p_i` is your forecast.
- Each test row is one (forecast episode, target site, forecast hour, pollutant). The site is the second field of the row id.
- The log transform scores relative errors, so all six pollutants count on a comparable scale, from CO in the thousands of µg/m³ to SO2 in single digits.
- **Why the worst-site term:** a forecast is only useful if it holds up at every unmonitored location. Half of the score is the error at the hardest target site, so a model that does well on average but fails at one site is penalised.
- Only hours with a real measurement are test rows. The score is capped at 100.

```python
import numpy as np
import pandas as pd

def evaluate(ids, y_true, y_pred):
    sq = (np.log1p(np.asarray(y_pred, float)) - np.log1p(np.asarray(y_true, float))) ** 2
    site = pd.Series(ids).str.split("_").str[1]
    per_site = np.sqrt(pd.Series(sq).groupby(site.values).mean())
    return min(0.5 * per_site.mean() + 0.5 * per_site.max(), 100.0)
```

Reference points on the test set:

- Training-period median of each pollutant (`sample_submission.csv`): 1.042
- Training-period month × hour climatology of the network: 0.918
- Persistence (the network-mean value at the last context hour, held for all 24 hours): 0.870

## Dataset

All files are in `public/`.

**`weather.csv`**: hourly weather for all 12 sites from 2013-03-01 00:00 to 2016-02-29 23:00 (26,304 hours × 12 sites = 315,648 rows).

Columns:

- site (string): anonymous site id, S01–S12
- year, month, day, hour (int): timestamp, local time (hour 0–23)
- TEMP (float): air temperature, °C
- PRES (float): air pressure, hPa
- DEWP (float): dew-point temperature, °C
- RAIN (float): precipitation in the hour, mm
- wd (string): wind direction on a 16-point compass (N, NNE, …, NNW); blank when missing
- WSPM (float): wind speed, m/s

**`train.csv`**: 1,224,554 labelled rows, one for every measured pollutant value at the 8 network sites in the training period. The layout is exactly that of `test.csv` plus the target column `value`.
- `forecast_id` is `T<YYYYMMDD>`, the day of the measurement, and `rel_hour` / `hour` is the hour of that day.
- To train a forecaster, combine these labels with `weather.csv` into an hourly series. The 72 hours before 00:00 of a day form that day's context.
- Columns: id, forecast_id, site, rel_hour, hour, pollutant, value (µg/m³).

**`test_context.csv`**: the 73 test forecast episodes. There are 96 rows per site per episode: 72 context hours (rel_hour −72 to −1) and 24 target hours (rel_hour 0 to 23). That gives 73 × 12 × 96 = 84,096 rows.

Columns:

- forecast_id (string): episode identifier, e.g. F001. Ids are not in chronological order.
- site (string): anonymous site id
- rel_hour (int): hours relative to the forecast origin. The origin is 00:00 of the target day, so −72…−1 is the context and 0…23 is the target day.
- month (int), weekday (int, 0 = Monday), hour (int): calendar information for that hour. The calendar date and year are not given.
- TEMP, PRES, DEWP, RAIN, wd, WSPM: weather at that site and hour. Weather is given for the target day too, as a stand-in for a numerical weather forecast.
- PM2.5, PM10, SO2, NO2, CO, O3: network-site readings for the context hours only. They are blank for all target hours and always blank for the 4 target sites. These blanks make up about half of the pollutant cells, by design.

**`test.csv`**: the 41,491 rows to forecast, in the same layout as `train.csv` without `value`.

Columns:

- id (string): `<forecast_id>_<site>_h<rel_hour>_<pollutant>`, e.g. `F001_S03_h07_PM2.5`
- forecast_id, site, rel_hour, hour, pollutant (string): the same information, already split out

**`sample_submission.csv`**: a correctly formatted submission.

Notes:
- Test episodes follow the pattern 3 context days, 1 target day, 1 unused day. No target hour appears in any episode's context, and no two episodes touch in time.
- Each site's weather comes from the nearest weather station, so some sites share an identical weather record.
- Missing values are blank. Missing pollutant readings are not zero. Treat them as unknown.
- Some extreme values sit at instrument reporting limits (999 for PM, 10000 for CO).
- **No future information:** your forecast for an episode must use only that episode's context rows, the target-day weather, and the training files (`weather.csv`, `train.csv`). Do not reconstruct the test-period timeline by linking episodes.
- **External data:** do not use any external air-quality or weather data, and do not try to identify the real sites or look up public copies of the source measurements. No external data is needed.

## Submission

Submit a CSV with two columns:

- id (string): row identifier from `test.csv`
- value (float): forecast concentration in µg/m³ for that site, hour and pollutant

**Requirements**
- One row for every id in `test.csv` (41,491 rows, any order), plus a header row.
- Every id appears exactly once.
- `value` must be numeric, finite, non-missing and non-negative.
- Submissions that break these rules are rejected rather than scored.
