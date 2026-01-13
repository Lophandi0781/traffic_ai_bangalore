import pandas as pd
import numpy as np

def _safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Required columns:
      - timestamp (datetime-like string)
      - location_id (segment/road/junction id)
      - speed (target, km/h)
    """
    df = df.copy()

    if "timestamp" not in df.columns:
        raise ValueError("Missing required column: timestamp")

    df["timestamp"] = _safe_to_datetime(df["timestamp"])
    df = df.dropna(subset=["timestamp"])

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek  # Mon=0
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Ensure location exists
    if "location_id" not in df.columns:
        df["location_id"] = "default"

    df = df.sort_values(["location_id", "timestamp"])

    # Lag features (tune these based on sampling frequency)
    for lag in [1, 2, 4, 8]:
        df[f"speed_lag_{lag}"] = df.groupby("location_id")["speed"].shift(lag)

    # Rolling stats (shifted to avoid leakage)
    shifted = df.groupby("location_id")["speed"].shift(1)
    df["speed_roll_mean_4"] = (
        shifted.groupby(df["location_id"])
        .rolling(4)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["speed_roll_std_4"] = (
        shifted.groupby(df["location_id"])
        .rolling(4)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Fill missing lag/rolling values
    lag_cols = [c for c in df.columns if "lag" in c or "roll_" in c]
    for c in lag_cols:
        df[c] = df[c].fillna(df.groupby("location_id")[c].transform("median"))
        df[c] = df[c].fillna(df[c].median())

    return df
