"""
predict.py — Load best model and predict on new data.

Uses the Random Forest model trained in Phase 1.
"""

import pandas as pd
import numpy as np
import yaml
import joblib
import json
from pathlib import Path


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_best_model(config):
    """Load the best model (Random Forest) from disk."""
    models_dir = Path(config["paths"]["models_dir"])
    path = models_dir / "random_forest.joblib"
    return joblib.load(path)


def load_encoders(config):
    """Load label encoding mappings."""
    enc_path = Path(config["paths"]["reports_dir"]) / "label_encodings.json"
    with open(enc_path) as f:
        return json.load(f)


def predict_single(model, movement_activity, location, hour, day_of_week, encoders):
    """
    Predict fall probability for a single observation.

    Args:
        movement_activity: str — "Walking", "Sitting", "Lying", "No Movement"
        location: str — "Kitchen", "Living Room", "Bedroom", "Bathroom"
        hour: int — 0-23
        day_of_week: int — 0=Monday, 6=Sunday
    """
    # Encode categoricals
    mv_encoded = encoders["Movement Activity"][movement_activity]
    loc_encoded = encoders["Location"][location]

    X = pd.DataFrame([{
        "Movement Activity": mv_encoded,
        "Location": loc_encoded,
        "Hour": hour,
        "Day_of_Week": day_of_week,
    }])

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    return {"prediction": int(pred), "fall_probability": round(float(prob), 4)}


def main():
    """CLI demo: predict on example inputs."""
    config = load_config()
    model = load_best_model(config)
    encoders = load_encoders(config)

    print("=" * 50)
    print("  Fall Detection — Prediction Demo")
    print("  Model: Random Forest (Phase 1)")
    print("=" * 50)

    examples = [
        ("Walking", "Bathroom", 2, 0),
        ("No Movement", "Bedroom", 23, 5),
        ("Lying", "Kitchen", 14, 3),
        ("Sitting", "Living Room", 10, 1),
    ]

    for mv, loc, hr, dow in examples:
        result = predict_single(model, mv, loc, hr, dow, encoders)
        label = "⚠️ FALL" if result["prediction"] == 1 else "✅ No Fall"
        print(f"\n  Input:  {mv}, {loc}, Hour={hr}, Day={dow}")
        print(f"  Result: {label}  (prob={result['fall_probability']:.4f})")


if __name__ == "__main__":
    main()
