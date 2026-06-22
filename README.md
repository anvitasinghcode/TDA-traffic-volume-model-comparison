#This README is AI-Generated because I wanted to convey the terminal commands for the project 
# Traffic Volume — Model Comparison (Random Forest vs AdaBoost vs XGBoost)

## Dataset
Download `Metro_Interstate_Traffic_Volume.csv` from:
https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume

Place it in the same folder as `traffic_volume_model_comparison.py`.

## Setup
```
pip3 install -r requirements.txt
```

**Note for Mac users:** if you get an `XGBoost Library could not be loaded` error,
it's because XGBoost needs the `libomp` runtime, which isn't installed by default.
Fix with:
```
brew install libomp
```
(If you don't have Homebrew: https://brew.sh)

## Run
```
python3 traffic_volume_model_comparison.py
```

## Output
- Console: dataset shape, train/test split sizes, MAE/RMSE/R² for each model,
  and a final comparison table sorted by RMSE.
- `model_comparison.png`: bar chart comparing all 3 models across the 3 metrics.
- `feature_importance.png`: feature importance plot for the best-performing model.

## Result Summary (from our run)
Random Forest and XGBoost performed similarly well (R² ≈ 0.96), both
substantially outperforming AdaBoost (R² ≈ 0.83). Random Forest had a
slight edge in MAE/RMSE. AdaBoost's weaker performance is consistent with
its known sensitivity to noisy/outlier-heavy data traffic volume has
irregular spikes around holidays and weather events.
