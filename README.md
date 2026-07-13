# Data Mania 2024 — What Makes Traffic Accidents Worse?

🏆 **Winner — Furman University Data Mania, November 2024**

Given a 250,000-row sample of the [US Accidents dataset](https://smoosavi.org/datasets/us_accidents) and a day to work, the prompt asked three questions: what factors contribute most to the **severity** of crashes, to the **duration** of the traffic delays they cause, and to the **length of road** they affect?

## Approach

My modeling focused on the two continuous outcomes. From the raw start/end timestamps and distance fields I engineered two targets — `Affected_Time` (delay duration) and `Affected_Distance` (length of road affected) — and trained a tuned XGBoost regressor for each, then read the answers out of the models' feature importances and targeted comparisons.

**Cleaning & features** (`cleaning.py`, `cleaner.py`)
- Dropped IDs, free-text, and leakage-prone columns
- Consolidated the dataset's dozens of raw weather and wind categories into a manageable set via mapping dictionaries (`weather_dict.py`, `wind_dict.py`)
- One-hot encoded categoricals (state, weather, wind, etc.) into a model-ready matrix (`us_accidents_sample_cleaned.csv`)

**Models** (`train_model.py`, `search_model.py`)
- Two XGBoost regressors (delay duration; affected distance), tuned with **Optuna** — 50 trials over depth, learning rate, subsampling, and regularization, each scored by 5-fold `xgb.cv` with early stopping on RMSE, on GPU
- Best models saved as `best_model_time.xgb` / `best_model_distance.xgb`

**Answering the questions** (`feature_importance.py`, `state_comparison.py`, `vis.py`)
- Gain-based feature importances per target (e.g., `feature_importances_time.csv`) identify the dominant factors
- Slice comparisons across states (e.g., Louisiana vs. South Carolina average delay and distance) and matplotlib visuals support the findings

## Running it

```bash
pip install pandas xgboost optuna scikit-learn matplotlib
python cleaning.py        # raw sample -> cleaned, encoded matrix
python train_model.py     # tune + train + save models and importances
```

Training uses `device='cuda'` by default — switch to `'cpu'` in `train_model.py` if needed.

## Data & attribution

The included CSV is a sample of **US Accidents** by Moosavi et al., used under CC BY-NC-SA 4.0 for non-commercial purposes:

> Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. "A Countrywide Traffic Accident Dataset." arXiv:1906.05409 (2019).

---
*Part of my portfolio — [jackhroberts.dev](https://jackhroberts.dev)*
