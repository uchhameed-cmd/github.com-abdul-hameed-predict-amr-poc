#!/usr/bin/env python3
"""
train.py

Train logistic regression model for AMR prediction from k-mer features.

Usage:
    python train.py --features features/kmer_matrix.npz --labels data/ast_data.csv --output results/
    python train.py --features features/kmer_matrix.npz --labels data/ast_data.csv --model random_forest --cv 10
"""

import argparse
import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")


def load_data(features_path: str, labels_path: str, target_antimicrobial: str = None):
    """
    Load k-mer matrix and corresponding AST labels.

    Args:
        features_path: Path to NPZ file with k-mer matrix
        labels_path: Path to CSV with AST data
        target_antimicrobial: Optional filter by antimicrobial name

    Returns:
        X: Feature matrix
        y: Labels
        genome_ids: List of genome IDs
    """
    # Load k-mer matrix
    data = np.load(features_path, allow_pickle=True)
    X = data["X"]
    genome_ids = data["genome_ids"].tolist()

    print(f"Loaded feature matrix: {X.shape}")

    # Load AST labels
    df = pd.read_csv(labels_path)

    # Filter to genomes in our matrix
    df_filtered = df[df["genome_id"].isin(genome_ids)].copy()
    print(f"Matched {len(df_filtered)} genomes with labels")

    # Filter by antimicrobial if specified
    if target_antimicrobial:
        df_filtered = df_filtered[df_filtered["antimicrobial"] == target_antimicrobial]
        print(f"Filtered to {target_antimicrobial}: {len(df_filtered)} records")

    # Keep only one record per genome (take first or most common)
    df_filtered = df_filtered.drop_duplicates(subset="genome_id", keep="first")

    # Align features with labels
    genome_to_label = dict(zip(df_filtered["genome_id"], df_filtered["is_resistant"]))

    y = []
    aligned_genome_ids = []

    for i, gid in enumerate(genome_ids):
        if gid in genome_to_label:
            y.append(genome_to_label[gid])
            aligned_genome_ids.append(gid)

    # Filter X to aligned genomes
    idx = [i for i, gid in enumerate(genome_ids) if gid in genome_to_label]
    X = X[idx]

    y = np.array(y)

    print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: Resistant={y.sum()}, Susceptible={len(y) - y.sum()}")

    return X, y, aligned_genome_ids


def train_model(X, y, model_name="logistic", cv_folds=5):
    """
    Train model with cross-validation.

    Args:
        X: Feature matrix
        y: Labels
        model_name: "logistic", "random_forest", or "svm"
        cv_folds: Number of cross-validation folds

    Returns:
        y_pred_proba: Predicted probabilities
        y_pred: Predicted classes
        model: Trained model
        cv_scores: Dictionary of cross-validation scores
    """
    # Select model
    if model_name == "logistic":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    elif model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    elif model_name == "svm":
        model = SVC(probability=True, class_weight="balanced", random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"\nTraining {model_name} with {cv_folds}-fold cross-validation...")

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Get cross-validated predictions
    y_pred_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")

    # Calculate per-fold scores
    fold_scores = []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model_fold = model.__class__(**model.get_params())
        model_fold.fit(X_train, y_train)
        y_pred_fold = model_fold.predict_proba(X_test)[:, 1]

        fold_auc = roc_auc_score(y_test, y_pred_fold)
        fold_scores.append(fold_auc)

    # Train final model on all data
    model.fit(X, y)

    # Calculate metrics
    metrics = {
        "auroc": roc_auc_score(y, y_pred_proba),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "cv_auroc_mean": np.mean(fold_scores),
        "cv_auroc_std": np.std(fold_scores),
        "cv_folds": cv_folds,
    }

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    print(f"\nResults:")
    print(f"  AUROC: {metrics['auroc']:.3f}")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1: {metrics['f1']:.3f}")
    print(f"  CV AUROC: {metrics['cv_auroc_mean']:.3f} ± {metrics['cv_auroc_std']:.3f}")

    return y_pred_proba, y_pred, model, metrics


def plot_roc_curve(y_true, y_pred_proba, output_path: str, model_name: str = None):
    """
    Generate and save ROC curve plot.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save plot
        model_name: Optional model name for title
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auroc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUROC = {auroc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUROC = 0.5)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve: {model_name or 'AMR Predictor'}", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"ROC curve saved to: {output_path}")
    plt.close()


def save_results(metrics: dict, y_true, y_pred, y_pred_proba, output_dir: str, model_name: str):
    """
    Save all results to files.

    Args:
        metrics: Dictionary of metrics
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        output_dir: Output directory
        model_name: Model name for file naming
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics as JSON
    metrics_json = {
        **metrics,
        "model_name": model_name,
        "n_samples": len(y_true),
        "n_resistant": int(y_true.sum()),
        "n_susceptible": int(len(y_true) - y_true.sum()),
    }

    # Convert numpy types to Python types for JSON serialization
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2, cls=NumpyEncoder)

    print(f"Metrics saved to: {output_dir / 'metrics.json'}")

    # Save predictions
    predictions_df = pd.DataFrame(
        {
            "true_label": y_true,
            "predicted_label": y_pred,
            "predicted_probability": y_pred_proba,
        }
    )
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    print(f"Predictions saved to: {output_dir / 'predictions.csv'}")

    # Save classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Classification report saved to: {output_dir / 'classification_report.json'}")


def main():
    parser = argparse.ArgumentParser(description="Train AMR prediction model")
    parser.add_argument("--features", type=str, required=True, help="Path to k-mer matrix NPZ file")
    parser.add_argument("--labels", type=str, required=True, help="Path to AST labels CSV file")
    parser.add_argument("--output", type=str, default="results/", help="Output directory")
    parser.add_argument("--antimicrobial", type=str, default=None, help="Filter by antimicrobial name")
    parser.add_argument("--model", type=str, default="logistic", choices=["logistic", "random_forest", "svm"])
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plot from existing predictions")

    args = parser.parse_args()

    if args.plot_only:
        # Load existing predictions and plot
        predictions_df = pd.read_csv(Path(args.output) / "predictions.csv")
        y_true = predictions_df["true_label"].values
        y_pred_proba = predictions_df["predicted_probability"].values

        plot_roc_curve(
            y_true,
            y_pred_proba,
            Path(args.output) / "roc_curve.png",
            model_name=args.model,
        )
        print("✓ Plot generated!")
        return

    # Load data
    X, y, genome_ids = load_data(args.features, args.labels, target_antimicrobial=args.antimicrobial)

    if len(y) < 10:
        print(f"Error: Only {len(y)} samples found. Need at least 10 samples for meaningful training.")
        sys.exit(1)

    # Train model
    y_pred_proba, y_pred, model, metrics = train_model(X, y, model_name=args.model, cv_folds=args.cv)

    # Generate ROC curve
    plot_roc_curve(y, y_pred_proba, Path(args.output) / "roc_curve.png", model_name=args.model)

    # Save results
    save_results(metrics, y, y_pred, y_pred_proba, args.output, args.model)

    # Print feature importance for logistic regression (top 10)
    if args.model == "logistic" and hasattr(model, "coef_"):
        coefs = model.coef_[0]
        top_features_idx = np.argsort(np.abs(coefs))[-10:][::-1]

        # Note: We don't have feature names easily here, but can be added
        print("\nTop 10 feature coefficients (absolute):")
        for i, idx in enumerate(top_features_idx):
            print(f"  {i+1}. Feature {idx}: {coefs[idx]:.4f}")

    print(f"\n✓ All results saved to {args.output}/")


if __name__ == "__main__":
    main()
