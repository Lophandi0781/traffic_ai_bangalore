import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib

from feature_build import build_features

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bangalore_traffic.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA_PATH}\n"
            "Put your CSV at: data/bangalore_traffic.csv"
        )

    df = pd.read_csv(DATA_PATH)

    # ✅ IMPORTANT: Update these keys to match your CSV header names
    colmap = {
        "DateTime": "timestamp",     # example
        "Location": "location_id",   # example
        "Speed": "speed",            # example
    }

    for src, dst in colmap.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    required = ["timestamp", "location_id", "speed"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset missing columns: {missing}\n"
            "Fix: Update colmap in training/train.py to match your CSV headers.\n"
            f"Your CSV columns are: {list(df.columns)}"
        )

    df = build_features(df)

    feature_cols = [
        "hour","dayofweek","month","is_weekend",
        "hour_sin","hour_cos","dow_sin","dow_cos",
        "speed_lag_1","speed_lag_2","speed_lag_4","speed_lag_8",
        "speed_roll_mean_4","speed_roll_std_4",
    ]

    # Simple location encoding (MVP)
    df["loc_hash"] = df["location_id"].astype(str).apply(lambda x: hash(x) % 1_000_000)
    feature_cols = ["loc_hash"] + feature_cols

    X = df[feature_cols]
    y = df["speed"]

    # Time-series split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    print(f"✅ MAE (km/h): {mae:.3f}")

    joblib.dump(model, ARTIFACTS / "model.joblib")
    (ARTIFACTS / "feature_meta.json").write_text(
        json.dumps({"feature_cols": feature_cols, "colmap": colmap}, indent=2),
        encoding="utf-8"
    )

    print("✅ Saved:", ARTIFACTS / "model.joblib")
    print("✅ Saved:", ARTIFACTS / "feature_meta.json")

if __name__ == "__main__":
    main()
