"""
model_training.py — Train 5 models with SMOTE-within-fold cross-validation.

Critical fix: SMOTE is applied ONLY inside each CV fold's training set,
preventing synthetic-sample leakage into the validation set.
"""

import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from imblearn.over_sampling import SMOTE


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_models():
    """Return dict of model name → model instance."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def evaluate_fold(y_true, y_pred, y_prob):
    """Compute metrics for a single fold."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob),
    }


def cross_validate_with_smote(X, y, models, config):
    """
    Stratified K-Fold CV with SMOTE applied INSIDE each fold.

    For each fold:
      1. Split into fold-train and fold-val
      2. Apply SMOTE to fold-train ONLY
      3. Train on SMOTE'd fold-train
      4. Evaluate on original fold-val (no SMOTE)
    """
    n_folds = config["cv_folds"]
    rs = config["random_state"]
    smote_k = config["smote"]["k_neighbors"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs)

    all_results = []

    for model_name, model_template in models.items():
        print(f"\n{'='*50}")
        print(f"  {model_name}")
        print(f"{'='*50}")

        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

            # SMOTE on training fold ONLY
            smote = SMOTE(
                random_state=rs,
                k_neighbors=min(smote_k, y_fold_train.sum() - 1),
            )
            X_resampled, y_resampled = smote.fit_resample(X_fold_train, y_fold_train)

            # Train
            from sklearn.base import clone
            model = clone(model_template)
            model.fit(X_resampled, y_resampled)

            # Predict on ORIGINAL validation fold
            y_pred = model.predict(X_fold_val)
            y_prob = model.predict_proba(X_fold_val)[:, 1]

            metrics = evaluate_fold(y_fold_val, y_pred, y_prob)
            metrics["Fold"] = fold_idx + 1
            metrics["Model"] = model_name
            fold_metrics.append(metrics)

            print(f"  Fold {fold_idx+1}: F1={metrics['F1']:.4f}  AUC={metrics['AUC']:.4f}")

        # Average across folds
        fold_df = pd.DataFrame(fold_metrics)
        means = fold_df[["Accuracy", "Precision", "Recall", "F1", "AUC"]].mean()
        stds = fold_df[["Accuracy", "Precision", "Recall", "F1", "AUC"]].std()
        print(f"  Mean:  F1={means['F1']:.4f}±{stds['F1']:.4f}  AUC={means['AUC']:.4f}±{stds['AUC']:.4f}")

        all_results.extend(fold_metrics)

    return pd.DataFrame(all_results)


def train_final_models(X_train, y_train, models, config):
    """Train final models on full training set with SMOTE. Save to disk."""
    rs = config["random_state"]
    smote_k = config["smote"]["k_neighbors"]
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    smote = SMOTE(
        random_state=rs,
        k_neighbors=min(smote_k, y_train.sum() - 1),
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"\nSMOTE on full train set: {len(X_train)} → {len(X_resampled)} samples")

    trained = {}
    for name, model_template in models.items():
        from sklearn.base import clone
        model = clone(model_template)
        model.fit(X_resampled, y_resampled)
        trained[name] = model

        safe_name = name.lower().replace(" ", "_")
        path = models_dir / f"{safe_name}.joblib"
        joblib.dump(model, path)
        print(f"  Saved {name} → {path}")

    return trained


def main():
    config = load_config()
    target = config["target"]

    # Load train data
    train = pd.read_csv(config["paths"]["train_data"])
    print(f"Train set: {train.shape}")
    print(f"Target distribution:\n{train[target].value_counts()}\n")

    features = config["features"]
    X_train = train[features]
    y_train = train[target]

    models = get_models()

    # Cross-validation with SMOTE inside fold
    print("=" * 60)
    print("CROSS-VALIDATION WITH SMOTE-WITHIN-FOLD")
    print("=" * 60)
    cv_results = cross_validate_with_smote(X_train, y_train, models, config)

    # Save CV results
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Per-fold results
    cv_results.to_csv(reports_dir / "cv_results_per_fold.csv", index=False)

    # Summary (mean ± std)
    summary = cv_results.groupby("Model")[
        ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    ].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.sort_values("F1_mean", ascending=False)
    summary.to_csv(reports_dir / "cv_results.csv")

    print("\n" + "=" * 60)
    print("CV SUMMARY (sorted by F1)")
    print("=" * 60)
    print(summary[["F1_mean", "F1_std", "AUC_mean", "AUC_std"]].to_string())

    # Train final models on full train set
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODELS")
    print("=" * 60)
    train_final_models(X_train, y_train, models, config)

    print("\nDone. CV results saved to reports/")


if __name__ == "__main__":
    main()
