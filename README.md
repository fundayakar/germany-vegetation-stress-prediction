# Predicting Vegetation Stress from Antecedent Hydroclimatic Conditions Across Germany

This repository contains the data processing pipeline, machine learning models, and figures for the manuscript:

> **Predicting Vegetation Stress from Antecedent Hydroclimatic Conditions Across Germany: A Machine Learning Approach with Spatially Blocked Validation**
> Submitted to *Remote Sensing of Environment* (2026)

---

## Overview

This study develops a predictive framework for monthly vegetation stress events across Germany (2017–2024) using lagged hydroclimatic anomalies derived from MODIS MOD13A3 and ERA5-Land reanalysis data. Vegetation stress is defined as a negative NDVI anomaly exceeding one standard deviation below the long-term monthly mean. Three machine learning models (logistic regression, random forest, XGBoost) are evaluated using spatially blocked cross-validation at the federal state level. SHAP analysis is used to quantify predictor contributions globally and by land cover type.

---

## Repository Structure

```
germany-vegetation-stress-prediction/
├── gee/
│   └── gee_export_germany_ndvi_era5_lc_monthly.js   # Google Earth Engine export script
├── data/
│   ├── germany_ndvi_era5_lc_monthly_raw.csv          # Raw zonal statistics from GEE
│   └── germany_model_ready.csv                        # Processed dataset with anomalies and lags
├── python/
│   ├── 01_anomaly_lag_stress_labels.py               # Anomaly calculation, lag construction, stress labeling
│   ├── 02_ml_pipeline.py                             # Model training and spatial CV evaluation
│   ├── 03_shap_analysis.py                           # SHAP feature importance
│   ├── 04_shap_figures.py                            # SHAP visualizations
│   ├── 05_lc_models_figures.py                       # Land-cover-specific models and figures
│   ├── 06_robustness_check.py                        # Robustness check with alternative stress threshold
│   └── 07_study_design_figure.py                     # Study design diagram
│   └── 09_lead_time_analysis.py
│   └── 08_temporal_transferability.py
└── figure/
    ├── fig_study_design.png
    ├── fig_shap_importance.png
    ├── fig_shap_beeswarm.png
    ├── fig_shap_landcover.png
    ├── fig_lc_performance.png
    ├── fig_lc_shap_top3.png
    ├── fig_stress_prob_map.png
    └── fig_robustness.png
```

---

## Data Sources

| Dataset | Product | Resolution | Access |
|---|---|---|---|
| NDVI | MODIS MOD13A3 v6.1 | 1 km, monthly | [NASA LP DAAC](https://lpdaac.usgs.gov/) via GEE |
| Temperature, Soil Moisture, Precipitation | ERA5-Land monthly aggregates | ~9 km, monthly | [Copernicus](https://cds.climate.copernicus.eu/) via GEE |
| Land Cover | ESA WorldCover 2020 | 10 m | [ESA](https://esa-worldcover.org/) via GEE |

All data processing and zonal statistics extraction were performed in [Google Earth Engine](https://earthengine.google.com/).

---

## Workflow

**Step 1 — GEE Export**
Run `gee/gee_export_germany_ndvi_era5_lc_monthly.js` in the GEE Code Editor to export monthly NDVI and ERA5-Land zonal statistics for all 16 German federal states × 4 land cover classes × 2017–2024. Output: `germany_ndvi_era5_lc_monthly_raw.csv`

**Step 2 — Preprocessing**
```bash
python python/01_anomaly_lag_stress_labels.py
```
Computes z-score anomalies, constructs 1–3 month lags, and defines stress event labels. Output: `germany_model_ready.csv`

**Step 3 — ML Pipeline**
```bash
python python/02_ml_pipeline.py
```
Trains logistic regression, random forest, and XGBoost models under spatially blocked cross-validation (4-fold, federal state level). Reports ROC-AUC, PR-AUC, and F1.

**Step 4 — SHAP Analysis**
```bash
python python/03_shap_analysis.py
python python/04_shap_figures.py
```
Computes global and land-cover-stratified SHAP values and generates feature importance figures.

**Step 5 — Land Cover Models**
```bash
python python/05_lc_models_figures.py
```
Trains separate models for cropland, forest, and grassland subsets.

**Step 6 — Robustness Check**
```bash
python python/06_robustness_check.py
```
Repeats evaluation using a 20th percentile stress threshold as an alternative to the primary −1 SD definition.

---

## Requirements

```
python >= 3.9
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
```

Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib
```

---

## Key Results

| Model | ROC-AUC (spatial CV) | PR-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 0.826 | 0.506 | 0.494 |
| Random Forest | 0.917 | 0.716 | 0.592 |
| XGBoost | 0.905 | 0.701 | 0.646 |
| XGBoost (random CV) | 0.957 | 0.849 | — |

Dominant predictors (global SHAP): NDVI anomaly (t-1) > Precipitation anomaly (t-1) > Temperature anomaly (t-2) > Soil moisture anomaly (t-1)

---

## Citation

If you use this code or data, please cite the manuscript once published.

---

## License

MIT License
