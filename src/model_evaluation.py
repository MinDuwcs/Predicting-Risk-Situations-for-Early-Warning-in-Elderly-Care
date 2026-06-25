"""
model_evaluation.py — Evaluate trained models on held-out test set.

Produces test metrics, confusion matrices, feature importances,
and exports all results as CSVs for PowerBI.
"""

import pandas as pd
import numpy as np
import yaml
# pyrefly: ignore [missing-import]
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_models(models_dir):
    """Load all saved models from disk."""
    models = {}
    for path in sorted(Path(models_dir).glob("*.joblib")):
        # Convert filename back to display name
        name = path.stem.replace("_", " ").title()
        models[name] = joblib.load(path)
    return models


def evaluate_on_test(models, X_test, y_test):
    """Evaluate all models on the held-out test set (no SMOTE)."""
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "AUC": roc_auc_score(y_test, y_prob),
        }
        results.append(metrics)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1:        {metrics['F1']:.4f}")
        print(f"  AUC:       {metrics['AUC']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
        print(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['No Fall', 'Fall'])}")

    return pd.DataFrame(results)


def get_feature_importances(models, feature_names):
    """Extract feature importances from tree-based models."""
    rows = []
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for feat, imp in zip(feature_names, importances):
                rows.append({"Model": name, "Feature": feat, "Importance": imp})
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_[0])
            for feat, coef in zip(feature_names, coefs):
                rows.append({"Model": name, "Feature": feat, "Importance": coef})

    return pd.DataFrame(rows)


def export_predictions(models, X_test, y_test, features):
    """Export test set predictions for PowerBI analysis."""
    pred_df = X_test.copy()
    pred_df.columns = features
    pred_df["Actual"] = y_test.values

    for name, model in models.items():
        safe = name.lower().replace(" ", "_")
        pred_df[f"Pred_{safe}"] = model.predict(X_test)
        pred_df[f"Prob_{safe}"] = model.predict_proba(X_test)[:, 1]

    return pred_df


def export_confusion_matrices(models, X_test, y_test):
    """Export confusion matrix data for PowerBI visualization."""
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        rows.append({"Model": name, "TN": cm[0][0], "FP": cm[0][1],
                      "FN": cm[1][0], "TP": cm[1][1]})
    return pd.DataFrame(rows)


def main():
    config = load_config()
    target = config["target"]
    features = config["features"]
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    test = pd.read_csv(config["paths"]["test_data"])
    X_test = test[features]
    y_test = test[target]
    print(f"Test set: {test.shape}")
    print(f"Test target distribution:\n{y_test.value_counts()}\n")

    # Load models
    models = load_models(config["paths"]["models_dir"])
    print(f"Loaded models: {list(models.keys())}\n")

    # Evaluate on test set
    print("=" * 60)
    print("TEST SET EVALUATION (no SMOTE on test)")
    print("=" * 60)
    test_results = evaluate_on_test(models, X_test, y_test)
    test_results = test_results.sort_values("F1", ascending=False)
    test_results.to_csv(reports_dir / "test_results.csv", index=False)

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY (sorted by F1)")
    print("=" * 60)
    print(test_results.to_string(index=False))

    # Feature importances
    fi = get_feature_importances(models, features)
    if not fi.empty:
        fi.to_csv(reports_dir / "feature_importances.csv", index=False)
        print(f"\nFeature importances saved → {reports_dir / 'feature_importances.csv'}")

    # Predictions for PowerBI
    preds = export_predictions(models, X_test, y_test, features)
    preds.to_csv(reports_dir / "predictions.csv", index=False)
    print(f"Predictions saved → {reports_dir / 'predictions.csv'}")

    # Confusion matrices for PowerBI
    cm_df = export_confusion_matrices(models, X_test, y_test)
    cm_df.to_csv(reports_dir / "confusion_matrices.csv", index=False)
    print(f"Confusion matrices saved → {reports_dir / 'confusion_matrices.csv'}")

    # Compare with original notebook metrics
    print("\n" + "=" * 60)
    print("COMPARISON: Original Notebook vs Phase 1 (Honest Metrics)")
    print("=" * 60)
    original = {
        "Random Forest": {"F1": 0.9275, "AUC": 0.9530},
        "Decision Tree": {"F1": 0.8973, "AUC": 0.9202},
        "Gradient Boosting": {"F1": 0.8953, "AUC": 0.9255},
        "Naive Bayes": {"F1": 0.8937, "AUC": 0.9053},
        "Logistic Regression": {"F1": 0.6998, "AUC": 0.6439},
    }
    comparison_rows = []
    for _, row in test_results.iterrows():
        name = row["Model"]
        orig = original.get(name, {})
        comparison_rows.append({
            "Model": name,
            "Original_F1": orig.get("F1", "N/A"),
            "Phase1_F1": round(row["F1"], 4),
            "F1_Change": round(row["F1"] - orig.get("F1", 0), 4) if orig.get("F1") else "N/A",
            "Original_AUC": orig.get("AUC", "N/A"),
            "Phase1_AUC": round(row["AUC"], 4),
            "AUC_Change": round(row["AUC"] - orig.get("AUC", 0), 4) if orig.get("AUC") else "N/A",
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(reports_dir / "metric_comparison.csv", index=False)
    print(comp_df.to_string(index=False))
    print(f"\nComparison saved → {reports_dir / 'metric_comparison.csv'}")


if __name__ == "__main__":
    main()
