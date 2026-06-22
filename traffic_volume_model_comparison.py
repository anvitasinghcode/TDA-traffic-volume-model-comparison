

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

# Option B: load directly via the UCI ML repo package (uncomment if you prefer)
# pip install ucimlrepo
# from ucimlrepo import fetch_ucirepo
# dataset = fetch_ucirepo(id=492)
# df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

print("Shape:", df.shape)
print(df.head())
print(df.dtypes)


df = df.drop_duplicates()
df = df[df["temp"] > 0]          # remove sensor-error rows (temp in Kelvin, 0K is impossible)
df = df[df["rain_1h"] < 1000]    # remove an extreme outlier spike in rain_1h

df["date_time"] = pd.to_datetime(df["date_time"])


df["hour"] = df["date_time"].dt.hour
df["day_of_week"] = df["date_time"].dt.dayofweek      # 0 = Monday
df["month"] = df["date_time"].dt.month
df["year"] = df["date_time"].dt.year
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Encode categorical columns
cat_cols = ["holiday", "weather_main"]
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# Drop columns we won't feed into the model
df_model = df.drop(columns=["date_time", "holiday", "weather_main", "weather_description"])

feature_cols = [c for c in df_model.columns if c != "traffic_volume"]
X = df_model[feature_cols]
y = df_model["traffic_volume"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")


models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    ),
    "AdaBoost": AdaBoostRegressor(
        n_estimators=200, learning_rate=0.1, random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1, objective="reg:squarederror"
    ),
}


results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"\n{name}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R2   : {r2:.4f}")

results_df = pd.DataFrame(results).sort_values("RMSE")
print("\n=== Final Comparison (sorted by RMSE, lower is better) ===")
print(results_df.to_string(index=False))


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
    ax.bar(results_df["Model"], results_df[metric], color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_title(metric)
    ax.set_ylabel(metric)
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)

plt.suptitle("Model Comparison: Random Forest vs AdaBoost vs XGBoost")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()


best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    importances.plot(kind="bar")
    plt.title(f"Feature Importance ({best_model_name})")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()

    print(f"\nTop features for {best_model_name}:")
    print(importances.head(10))
