# Dataset Description

## Overview

The raw upload is the `PRSA2017_Data_20130301-20170228.zip` archive from the UCI Machine Learning Repository ("Beijing Multi-Site Air Quality", dataset id 501). It contains hourly air-pollutant and meteorological observations from 12 nationally controlled air-quality monitoring sites in Beijing, covering 2013-03-01 00:00 to 2017-02-28 23:00 (local time).

- **Pollutant data:** from the Beijing Municipal Environmental Monitoring Center.
- **Meteorological data:** each site's weather was matched by the publisher to the nearest China Meteorological Administration weather station, so several sites share an identical weather record.
- **Missing values:** source values that are missing are written as the literal `NA`.

**Source:** https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
**Direct file:** https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip
**Citation:** Zhang, S., Guo, B., Dong, A., He, J., Xu, Z. and Chen, S.X. (2017). Cautionary Tales on Air-Quality Improvement in Beijing. *Proceedings of the Royal Society A*, 473(2205), 20170457.
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0), which permits commercial use and redistribution with attribution.
**SHA-256 of the uploaded zip:** `d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`

## File Structure

- `PRSA2017_Data_20130301-20170228.zip`: the only uploaded file (7.6 MB). The platform extracts it into:
  - `PRSA_Data_20130301-20170228/PRSA_Data_<Station>_20130301-20170228.csv`: one file for each of the 12 stations:
    - Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng
    - Huairou, Nongzhanguan, Shunyi, Tiantan, Wanliu, Wanshouxigong

Each station file has a header row and 35,064 data rows (1,461 days × 24 hours), giving 420,768 rows in total. All 12 files use the same 18-column schema.

## Features

- No (int): Row counter within the file (1–35,064). It is not an identifier across files.
- year (int): Year of the observation (2013–2017)
- month (int): Month (1–12)
- day (int): Day of month (1–31)
- hour (int): Hour of day (0–23), Beijing local time
- PM2.5 (float): Fine particulate matter concentration, µg/m³ (2.1% missing)
- PM10 (float): Coarse particulate matter concentration, µg/m³ (1.5% missing)
- SO2 (float): Sulphur dioxide concentration, µg/m³ (2.1% missing)
- NO2 (float): Nitrogen dioxide concentration, µg/m³ (2.9% missing)
- CO (float): Carbon monoxide concentration, µg/m³ (4.9% missing)
- O3 (float): Ozone concentration, µg/m³ (3.2% missing)
- TEMP (float): Air temperature, °C (0.09% missing)
- PRES (float): Air pressure, hPa (0.09% missing)
- DEWP (float): Dew-point temperature, °C (0.10% missing)
- RAIN (float): Precipitation, mm (0.09% missing)
- wd (categorical): Wind direction on a 16-point compass: N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW (0.43% missing)
- WSPM (float): Wind speed, m/s (0.08% missing)
- station (categorical): Monitoring site name (constant within a file)

## Data Characteristics

- **Complete time grid:** every station has exactly one row for each hour from 2013-03-01 00:00 to 2017-02-28 23:00, with no duplicated hours. `(station, year, month, day, hour)` is a unique key.
- **Missing values:** missing values appear only as `NA` and occur in every measurement column. Missing hours often come in runs (analyser outages), not as isolated cells.
- **Skewed pollutants:** the pollutant distributions are strongly right-skewed. Approximate medians and maxima are:
  - PM2.5: median 55, max 999
  - PM10: median 82, max 999
  - SO2: median 7, max 500
  - NO2: median 43, max 290
  - CO: median 900, max 10,000
  - O3: median 45, max 1,071

  Values of 999 (PM) and 10,000 (CO) are instrument reporting ceilings.
- **Shared weather records:** sites matched to the same weather station have identical meteorology:
  - Changping = Dingling
  - Dongsi = Nongzhanguan = Tiantan
  - Aotizhongxin = Guanyuan
- **Correlation across sites:** pollution is strongly correlated across sites (regional haze episodes), but local sources (traffic, heating, terrain) produce site-specific offsets and diurnal patterns. NO2 and O3 vary most between sites.
- **Seasonality:** there is strong winter heating-season seasonality and a diurnal cycle (O3 peaks in the afternoon, NO2 and CO peak at night).

## Notes

- The leakage-sensitive grouping variable is **`station`**. Hourly observations at one site are strongly autocorrelated and share site-specific calibration, so random row-level splits overstate how well a model generalises to a new site.
- The challenge built on this dataset (`prepare.py`):
  - replaces every site name with an anonymous id (S01–S12), so no place name appears in the prepared files;
  - drops `No`; keeps all weather variables, `wd` and all 6 pollutants;
  - treats 4 sites as unmonitored: only their weather is published;
  - publishes March 2013 to February 2016 as hourly weather (`weather.csv`) plus every measured network-site pollutant value (`train.csv`), with no value repeated across files;
  - turns March 2016 to February 2017 into independent next-day forecast episodes (72 hours of network context, then a 24-hour target day).
- These are historical measurements from one metropolitan region, from site-specific instruments. They cannot be used to infer individual exposure or health outcomes.
