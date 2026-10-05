# Cold-Start Air-Quality Forecasting at Unmonitored Beijing Sites

## Overview

City agencies issue next-day air-quality forecasts, but only where they have monitors. This challenge asks for **next-day, hour-by-hour forecasts of six pollutants (PM2.5, PM10, SO2, NO2, CO, O3) at four sites that have never had a pollutant monitor**. The forecasts must come from the recent readings of the surrounding monitoring network and from weather information.

The data comes from 12 air-quality sites in Beijing, from March 2013 to February 2017:
- **8 network sites** have pollutant monitors.
- **4 target sites** (Gucheng, Huairou, Nongzhanguan, Wanliu) are treated as unmonitored. Their pollutant values are never published, not even in the training period. Only their weather is given.

The challenge combines three shifts:
1. **Forecasting:** each forecast is issued at midnight and covers the next 24 hours (horizons 1–24 h). Only the past is available.
2. **Cold-start locations:** there are no target labels at the sites being forecast. Any model must learn from the network sites and transfer to new locations.
3. **Temporal shift:** training covers March 2013 to February 2016. The test covers March 2016 to February 2017, a later year with its own pollution levels and weather.

## Evaluation

Submissions are scored with **Root Mean Squared Logarithmic Error (RMSLE)** over all test rows. Lower is better, and 0 is perfect.

```
RMSLE = sqrt( mean over all test rows i of ( log(1 + p_i) - log(1 + y_i) )^2 )
```

- `y_i` is the measured concentration (µg/m³) and `p_i` is your forecast.
- Each test row is one (forecast episode, target site, forecast hour, pollutant).
- The log transform scores relative errors, so all six pollutants count on a comparable scale, from CO in the thousands of µg/m³ to SO2 in single digits.
- Only hours with a real measurement are test rows.
- The score is capped at 100.

```python
import numpy as np

def evaluate(y_true, y_pred):
    return min(float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))), 100.0)
```

Reference points on the test set:

- Training-period median of each pollutant (`sample_submission.csv`): 1.014
- Training-period month × hour climatology of the network: 0.876
- Persistence (the network-mean value at the last context hour, held for all 24 hours): 0.831

## Dataset

All files are in `public/`.

**`train.csv`**: continuous hourly data for all 12 sites from 2013-03-01 00:00 to 2016-02-29 23:00 (26,304 hours × 12 sites = 315,648 rows).
- The six pollutant columns are filled for the 8 network sites.
- The pollutant columns are always blank for the 4 target sites.

Columns:

- station (string): site name
- year, month, day, hour (int): timestamp, Beijing local time (hour 0–23)
- TEMP (float): air temperature, °C
- PRES (float): air pressure, hPa
- DEWP (float): dew-point temperature, °C
- RAIN (float): precipitation in the hour, mm
- wd (string): wind direction on a 16-point compass (N, NNE, …, NNW); blank when missing
- WSPM (float): wind speed, m/s
- PM2.5, PM10, SO2, NO2, CO, O3 (float): pollutant concentrations, µg/m³; blank when not measured

**`test_context.csv`**: the 73 test forecast episodes. There are 96 rows per site per episode: 72 context hours (rel_hour −72 to −1) and 24 target hours (rel_hour 0 to 23). That gives 73 × 12 × 96 = 84,096 rows.

Columns:

- forecast_id (string): episode identifier, e.g. F001. Ids are not in chronological order.
- station (string): site name
- rel_hour (int): hours relative to the forecast origin. The origin is 00:00 of the target day, so −72…−1 is the context and 0…23 is the target day.
- month (int), weekday (int, 0 = Monday), hour (int): calendar information for that hour. The calendar date and year are not given.
- TEMP, PRES, DEWP, RAIN, wd, WSPM: weather at that site and hour. Weather is given for the target day too, as a stand-in for a numerical weather forecast.
- PM2.5, PM10, SO2, NO2, CO, O3: network-site readings for the context hours only. They are blank for all target hours and always blank for the 4 target sites.

**`test.csv`**: the 41,491 rows to forecast.

Columns:

- id (string): `<forecast_id>_<station>_h<rel_hour>_<pollutant>`, e.g. `F001_Gucheng_h07_PM2.5`
- forecast_id, station, rel_hour, hour, pollutant (string): the same information, already split out

**`sample_submission.csv`**: a correctly formatted submission.

Notes:
- Test episodes follow the pattern 3 context days, 1 target day, 1 unused day. No target hour appears in any episode's context, and no two episodes touch in time.
- Weather for each site was matched by the data publisher to the nearest China Meteorological Administration weather station. Several sites therefore share identical weather records.
- Missing values are blank. Missing pollutant readings in the context are not zero. Treat them as unknown.
- Some extreme values sit at instrument reporting limits (999 for PM, 10000 for CO).
- **No future information:** your forecast for an episode must use only that episode's context rows, the target-day weather, and the training data. Do not reconstruct the test-period timeline by linking episodes.
- **External data:** do not use any external air-quality measurements. In particular, do not use the public UCI/Kaggle copies of this dataset, which contain the answers. No external data is needed.

## Submission

Submit a CSV with two columns:

- id (string): row identifier from `test.csv`
- value (float): forecast concentration in µg/m³ for that site, hour and pollutant

**Requirements**
- One row for every id in `test.csv` (41,491 rows, any order), plus a header row.
- Every id appears exactly once.
- `value` must be numeric, finite, non-missing and non-negative.
- Submissions that break these rules are rejected rather than scored.
