# Beijing Virtual Air-Quality Stations

## Overview

Pollution monitors are expensive to install and maintain, so city networks are sparse. Environmental agencies often want a "virtual station": an estimate of the hourly pollution at a site that has a weather feed but no working pollutant analysers. A virtual station is built from the local weather and the readings of the monitors around it.

This challenge uses four years (March 2013 to February 2017) of hourly data from 12 nationally controlled air-quality sites in Beijing. **Eight sites form the "network"**: for these you get hourly meteorology and all six pollutant measurements. **Four other sites are held out**: for these you only get hourly meteorology. Your task is to reconstruct the **hourly concentrations of all six pollutants (PM2.5, PM10, SO2, NO2, CO, O3) at the four held-out sites** for every hour of the four years.

This is a spatial reconstruction problem, not a forecasting problem. The network sites' measurements cover exactly the same hours as the held-out sites. You may use network readings from any hour, including hours before and after the one you are predicting. The held-out sites lie at different places in and around the city, and their pollution levels need not match the network average. Your method must therefore generalise to **locations it has never seen labelled**.

## Evaluation

Submissions are scored with the **Mean Column-wise Root Mean Squared Logarithmic Error (MCRMSLE)**. Lower is better, and 0 is perfect.

For each pollutant *j*:

```
RMSLE_j = sqrt( mean over i in V_j of ( log(1 + p_ij) - log(1 + y_ij) )^2 )
```

```
MCRMSLE = ( RMSLE_PM2.5 + RMSLE_PM10 + RMSLE_SO2 + RMSLE_NO2 + RMSLE_CO + RMSLE_O3 ) / 6
```

- `y_ij` is the measured concentration and `p_ij` is your prediction for test row *i*, pollutant *j*.
- `V_j` is the set of test rows where pollutant *j* was actually measured. If a site's analyser reported nothing for an hour, that hour is skipped **for that pollutant only**. You must still submit a valid prediction for every cell.
- The log transform gives each pollutant equal weight even though their scales differ a lot (CO is in the thousands, SO2 often in single digits). It also scores relative rather than absolute errors, which suits these right-skewed concentrations.
- The score is capped at 100.

```python
import numpy as np

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]

def evaluate(answers, submission):
    # answers and submission are DataFrames aligned on "id"
    scores = []
    for col in POLLUTANTS:
        y = answers[col].to_numpy(float)
        p = submission[col].to_numpy(float)
        mask = ~np.isnan(y)
        scores.append(np.sqrt(np.mean((np.log1p(p[mask]) - np.log1p(y[mask])) ** 2)))
    return min(float(np.mean(scores)), 100.0)
```

Reference points on the test set:

| Approach | MCRMSLE |
|---|---|
| Network-median constant (`sample_submission.csv`) | 1.029 |
| Month × hour climatology of the network | 0.906 |
| Hourly mean (in log space) of the 8 network sites | 0.4815 |

## Dataset

All files are in `public/`. The timestamps form a complete hourly grid from `2013-03-01 00:00` to `2017-02-28 23:00` (35,064 hours) for every site. Local time is Beijing time.

**`train.csv`**: the 8 network sites (Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Shunyi, Tiantan, Wanshouxigong), 280,512 rows.

**`test.csv`**: the 4 held-out sites (Gucheng, Huairou, Nongzhanguan, Wanliu), 140,256 rows. It has the same columns as `train.csv` but without the six pollutant columns.

**`sample_submission.csv`**: a correctly formatted submission with one row per test row.

| Column | Type | Description |
|---|---|---|
| `id` | string | Row identifier: `<station>_<YYYYMMDDHH>`, e.g. `Gucheng_2013030100` |
| `station` | string | Monitoring site name |
| `year`, `month`, `day`, `hour` | int | Timestamp of the hourly observation (hour 0–23) |
| `TEMP` | float | Air temperature, °C |
| `PRES` | float | Air pressure, hPa |
| `DEWP` | float | Dew-point temperature, °C |
| `RAIN` | float | Precipitation in the hour, mm |
| `wd` | string | Wind direction, 16-point compass (`N`, `NNE`, …, `NNW`); blank when missing |
| `WSPM` | float | Wind speed, m/s |
| `PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3` | float | Pollutant concentrations, µg/m³ (**train only**). Blank when the analyser reported nothing |

Notes:
- The meteorological readings for each site were matched by the data publisher to the nearest China Meteorological Administration weather station.
- Every column except `id`, `station` and the timestamp has missing values, from under 0.1% for weather to about 5% for CO. Missing values in the network's pollutant readings are not zero. Treat them as unknown.
- Pollutant values are hourly averages reported by the monitors. Some extreme values sit at instrument reporting limits (for example 999 for PM, 10000 for CO).
- **External data:** do not use any external air-quality measurements. In particular, do not use the original public UCI/Kaggle copies of this dataset, which contain the held-out sites' answers. Publicly known site metadata, such as approximate coordinates, is allowed. No external data is required to solve the task.

## Submission

Submit a CSV file with exactly these columns:

| Column | Type | Description |
|---|---|---|
| `id` | string | Row identifier from `test.csv` |
| `PM2.5` | float | Predicted PM2.5 concentration, µg/m³ |
| `PM10` | float | Predicted PM10 concentration, µg/m³ |
| `SO2` | float | Predicted SO2 concentration, µg/m³ |
| `NO2` | float | Predicted NO2 concentration, µg/m³ |
| `CO` | float | Predicted CO concentration, µg/m³ |
| `O3` | float | Predicted O3 concentration, µg/m³ |

**Requirements**
- Exactly 140,256 rows, one for each `id` in `test.csv` (any order), plus a header row.
- Every `id` must appear exactly once.
- All six prediction columns must be numeric, finite, non-missing and non-negative.
- Submissions that break these rules are rejected rather than scored.
