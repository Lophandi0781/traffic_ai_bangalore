import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts"

class TrafficModel:
    def __init__(self):
        model_path = ARTIFACTS / "model.joblib"
        meta_path = ARTIFACTS / "feature_meta.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError("Run training first: python training/train.py")

        self.model = joblib.load(model_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # meta may contain feature_cols or feature_cols name depending on your earlier version
        self.feature_cols = meta.get("feature_cols") or meta.get("feature_cols") or meta.get("feature_cols")
        if self.feature_cols is None:
            # fallback key from some earlier variants
            self.feature_cols = meta.get("feature_cols") or meta.get("feature_cols") or meta.get("feature_cols")

        # MVP defaults (for demo). Production should pull recent history per location.
        self.default_lags = {
            "speed_lag_1": 22.0, "speed_lag_2": 23.0, "speed_lag_4": 24.0, "speed_lag_8": 26.0,
            "speed_roll_mean_4": 23.0, "speed_roll_std_4": 3.0
        }

    def _time_features(self, ts: datetime) -> dict:
        hour = ts.hour
        dow = ts.weekday()
        month = ts.month
        return {
            "hour": hour,
            "dayofweek": dow,
            "month": month,
            "is_weekend": 1 if dow >= 5 else 0,
            "hour_sin": float(np.sin(2*np.pi*hour/24)),
            "hour_cos": float(np.cos(2*np.pi*hour/24)),
            "dow_sin": float(np.sin(2*np.pi*dow/7)),
            "dow_cos": float(np.cos(2*np.pi*dow/7)),
        }

    def predict_speed(self, location_id: str, ts: datetime, horizon_minutes: int, is_rain: int, is_event: int) -> float:
        feats = {"loc_hash": hash(str(location_id)) % 1_000_000}
        feats.update(self._time_features(ts))
        feats.update(self.default_lags)

        # simple what-if knobs
        if is_rain:
            feats["speed_lag_1"] = max(5.0, feats["speed_lag_1"] - 3.0)
        if is_event:
            feats["speed_lag_1"] = max(5.0, feats["speed_lag_1"] - 2.0)

        # horizon penalty (demo)
        horizon_penalty = (horizon_minutes - 15) / 165 * 2.0
        feats["speed_lag_1"] = max(5.0, feats["speed_lag_1"] - horizon_penalty)

        X = pd.DataFrame([feats])[self.feature_cols]
        pred = float(self.model.predict(X)[0])
        return max(1.0, pred)

    @staticmethod
    def congestion_label(speed: float) -> str:
        if speed >= 28:
            return "LOW"
        if speed >= 18:
            return "MEDIUM"
        return "HIGH"
