"""
CircuitGuard AI - Model Evaluation Pipeline
Evaluates all retained models on the untouched holdout test set (750 samples).
Computes:
- Confusion Matrix (TN, FP, FN, TP) with class support
- Accuracy, Precision (Class 1), Recall (Class 1), F1-Score (Class 1)
- ROC-AUC and PR-AUC
- Detailed classification reports with macro/weighted averages
- Feature importance analysis and publication-grade evaluation plots
"""

import os
import sys

# Prevent OpenMP runtime collision on macOS ARM64
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def run_evaluation(
    processed_dir: str = "dataset/processed",
    models_dir: str = "models",
    reports_dir: str = "reports"
):
    os.makedirs(reports_dir, exist_ok=True)
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("Loading untouched holdout test dataset...")
    X_test = np.load(os.path.join(processed_dir, "X_test_trans.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    with open(os.path.join(models_dir, "feature_metadata.json"), "r") as f:
        metadata = json.load(f)
    feature_names = metadata["transformed_feature_names"]

    n_normal = int((y_test == 0).sum())
    n_maint = int((y_test == 1).sum())
    total_test = len(y_test)

    print(f"Untouched test set: {total_test} samples")
    print(f" - Class 0 (No Maintenance Required): {n_normal} samples ({n_normal/total_test*100:.1f}%)")
    print(f" - Class 1 (Maintenance Required - Positive Class): {n_maint} samples ({n_maint/total_test*100:.1f}%)")

    # Load all retained models
    model_definitions = {
        "XGBoost (Fine-Tuned)": "xgboost_tuned.joblib",
        "XGBoost (Baseline)": "xgboost_baseline.joblib",
        "Random Forest": "random_forest.joblib",
        "Gradient Boosting (GBM)": "gradient_boosting.joblib",
        "Logistic Regression": "logistic_regression.joblib"
    }

    models = {}
    for name, filename in model_definitions.items():
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            models[name] = joblib.load(filepath)
            print(f"Loaded: {name}")
        else:
            print(f"Warning: {filename} not found.")

    evaluation_metrics = {}
    roc_data = {}
    pr_data = {}
    conf_matrices = {}

    print("\n" + "="*80)
    print("UNTOUCHED TEST SET EVALUATION BENCHMARK")
    print("="*80)

    for name, model in models.items():
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds.astype(float)

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, pos_label=1, zero_division=0))
        rec = float(recall_score(y_test, preds, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_test, preds, pos_label=1, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, probs))
        pr_auc = float(average_precision_score(y_test, probs))
        cm = confusion_matrix(y_test, preds)

        tn, fp, fn, tp = cm.ravel()
        clf_rep = classification_report(
            y_test,
            preds,
            target_names=["No Maintenance Required (0)", "Maintenance Required (1)"],
            output_dict=True,
            zero_division=0
        )

        fpr, tpr, _ = roc_curve(y_test, probs)
        precision_pts, recall_pts, _ = precision_recall_curve(y_test, probs)

        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
        pr_data[name] = {"precision": precision_pts, "recall": recall_pts, "pr_auc": pr_auc}
        conf_matrices[name] = cm

        evaluation_metrics[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "positive_class": "1 (Maintenance Required)",
            "averaging_method": "Binary (Positive label = 1)",
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "raw_matrix": cm.tolist()
            },
            "class_support": {
                "Class 0 (No Maintenance Required)": n_normal,
                "Class 1 (Maintenance Required)": n_maint,
                "total": total_test
            },
            "classification_report": clf_rep
        }

        print(f"{name:25s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f} | FP (False Alarms): {fp:2d} | FN (Missed Faults): {fn:2d}")

    # Save metrics JSON
    metrics_path = os.path.join(reports_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(evaluation_metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # -------------------------------------------------------------------------
    # VISUALIZATION 1: CONFUSION MATRICES PLOT
    # -------------------------------------------------------------------------
    num_models = len(models)
    fig, axes = plt.subplots(1, num_models, figsize=(4.2 * num_models, 4.0), dpi=200)
    if num_models == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, conf_matrices.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Normal (0)", "Maint (1)"],
            yticklabels=["Normal (0)", "Maint (1)"],
            annot_kws={"size": 11, "weight": "bold"}
        )
        ax.set_title(f"{name}\nAcc: {evaluation_metrics[name]['accuracy']:.3f} | F1: {evaluation_metrics[name]['f1_score']:.3f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=9)
        ax.set_ylabel("True Class", fontsize=9)

    plt.tight_layout()
    cm_path = os.path.join(figures_dir, "confusion_matrices.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrices figure to {cm_path}")

    # -------------------------------------------------------------------------
    # VISUALIZATION 2: ROC & PR CURVES PLOT
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=200)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # ROC Curves
    for (name, data), color in zip(roc_data.items(), colors):
        ax1.plot(data["fpr"], data["tpr"], label=f"{name} (AUC = {data['auc']:.4f})", color=color, lw=2)
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance (AUC = 0.5000)")
    ax1.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=11, fontweight="bold")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=10)
    ax1.legend(loc="lower right", fontsize=8.5)
    ax1.grid(True, alpha=0.25)

    # PR Curves
    for (name, data), color in zip(pr_data.items(), colors):
        ax2.plot(data["recall"], data["precision"], label=f"{name} (AP = {data['pr_auc']:.4f})", color=color, lw=2)
    ax2.set_title("Precision-Recall (PR) Curves (Positive = Maintenance Required)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Recall (Coverage of Real Maintenance Cases)", fontsize=10)
    ax2.set_ylabel("Precision (Accuracy of Maintenance Flags)", fontsize=10)
    ax2.legend(loc="lower left", fontsize=8.5)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    roc_path = os.path.join(figures_dir, "roc_pr_curves.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC/PR curves to {roc_path}")

    # -------------------------------------------------------------------------
    # VISUALIZATION 3: FEATURE IMPORTANCE (TUNED XGBOOST)
    # -------------------------------------------------------------------------
    if "XGBoost (Fine-Tuned)" in models:
        xgb_model = models["XGBoost (Fine-Tuned)"]
        importances = xgb_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]

        plt.figure(figsize=(9, 6), dpi=200)
        plt.barh(range(15), importances[indices][::-1], align="center", color="#1E88E5", edgecolor="#0D47A1")
        plt.yticks(range(15), [feature_names[i] for i in indices][::-1], fontsize=9)
        plt.xlabel("XGBoost Gain Feature Importance", fontsize=10, fontweight="bold")
        plt.title("Top 15 Influential Telemetry Features — XGBoost (Fine-Tuned)", fontsize=11, fontweight="bold")
        plt.grid(True, axis="x", alpha=0.25)
        plt.tight_layout()
        feat_path = os.path.join(figures_dir, "feature_importance.png")
        plt.savefig(feat_path)
        plt.close()
        print(f"Saved feature importance figure to {feat_path}")

    return evaluation_metrics

if __name__ == "__main__":
    run_evaluation()
