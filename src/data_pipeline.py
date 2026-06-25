"""
data_pipeline.py — Data cleaning, feature engineering, and train/test split.

Reads raw safety_monitoring.csv, removes post-event features (data leakage),
engineers time-based features, and produces a clean dataset + reproducible split.
"""

import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_data(path):
    """Load raw CSV and drop the empty trailing column."""
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 9"], errors="ignore")
    return df


def document_leakage(df):
    """Print leakage evidence for each post-event feature."""
    target = "Fall Detected (Yes/No)"
    leaked = [
        "Post-Fall Inactivity Duration (Seconds)",
        "Alert Triggered (Yes/No)",
        "Caregiver Notified (Yes/No)",
        "Impact Force Level",
    ]

    print("=" * 60)
    print("DATA LEAKAGE ANALYSIS")
    print("=" * 60)

    for col in leaked:
        if col not in df.columns:
            continue
        ct = pd.crosstab(df[target], df[col])
        print(f"\n--- {col} vs {target} ---")
        print(ct)

    print("\n" + "=" * 60)


def clean_data(df, config):
    """Remove post-event + ID columns, engineer time features."""
    # 1. Document what we're removing
    post_event = config["post_event_features"]
    drop_cols = config["drop_columns"]

    print(f"Removing post-event features: {post_event}")
    print(f"Removing non-predictive columns: {drop_cols}")

    # 2. Engineer time features BEFORE dropping Timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Hour"] = df["Timestamp"].dt.hour
    df["Day_of_Week"] = df["Timestamp"].dt.dayofweek

    # 3. Drop leaked + non-predictive columns
    all_drop = post_event + drop_cols
    existing = [c for c in all_drop if c in df.columns]
    df = df.drop(columns=existing)

    print(f"\nClean columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")

    return df


def encode_categoricals(df):
    """Label-encode categorical columns. Returns df and encoder mapping."""
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        if col == "Fall Detected (Yes/No)":
            # Target: Yes=1, No=0
            df[col] = df[col].map({"Yes": 1, "No": 0})
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    return df, encoders


def create_split(df, config):
    """Stratified 80/20 train-test split."""
    target = config["target"]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    return train, test


def main():
    config = load_config()

    # Load raw data
    print("Loading raw data...")
    df = load_raw_data(config["paths"]["raw_data"])
    print(f"Raw shape: {df.shape}")
    print(f"Target distribution:\n{df['Fall Detected (Yes/No)'].value_counts()}\n")

    # Document leakage
    document_leakage(df)

    # Clean
    print("\nCleaning data...")
    df_clean = clean_data(df, config)

    # Save clean CSV (before encoding, human-readable)
    clean_path = config["paths"]["clean_data"]
    Path(clean_path).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(clean_path, index=False)
    print(f"\nSaved clean data to {clean_path}")

    # Encode
    df_encoded, encoders = encode_categoricals(df_clean)
    print(f"\nLabel encodings: {encoders}")

    # Split
    print("\nCreating train/test split...")
    train, test = create_split(df_encoded, config)
    print(f"Train: {train.shape} (Fall={train[config['target']].sum()})")
    print(f"Test:  {test.shape} (Fall={test[config['target']].sum()})")

    # Save splits
    train.to_csv(config["paths"]["train_data"], index=False)
    test.to_csv(config["paths"]["test_data"], index=False)
    print(f"Saved train → {config['paths']['train_data']}")
    print(f"Saved test  → {config['paths']['test_data']}")

    # Save encoding map for reproducibility
    import json
    enc_path = Path(config["paths"]["reports_dir"]) / "label_encodings.json"
    enc_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy int values to plain int for JSON
    encoders_serializable = {
        col: {k: int(v) for k, v in mapping.items()}
        for col, mapping in encoders.items()
    }
    with open(enc_path, "w") as f:
        json.dump(encoders_serializable, f, indent=2)
    print(f"Saved encodings → {enc_path}")


if __name__ == "__main__":
    main()
