"""
CircuitGuard AI - Model Training Pipeline
Implements:
1. Baseline models (Untuned XGBoost, Random Forest, Gradient Boosting, Logistic Regression)
2. Hyperparameter optimization of XGBoost using 5-Fold Stratified Cross-Validation
3. Model artifact persistence for production inference
Note: TabNet has been completely removed in accordance with project requirements.
"""

import os
import sys

# Thread isolation for macOS ARM64 and reproducible execution
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import json
import time
import warnings
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def train_and_tune_models(
    processed_dir: str = "dataset/processed",
    models_dir: str = "models",
    random_state: int = 42
):
    os.makedirs(models_dir, exist_ok=True)
    np.random.seed(random_state)

    print(f"Loading preprocessed arrays from {processed_dir}...")
    X_train = np.load(os.path.join(processed_dir, "X_train_trans.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val_trans.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test_trans.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    with open(os.path.join(models_dir, "feature_metadata.json"), "r") as f:
        metadata = json.load(f)
    feature_names = metadata["transformed_feature_names"]

    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"Features count: {len(feature_names)}")
    print(f"Test Class 0 (Normal): {(y_test == 0).sum()}, Class 1 (Maintenance Required): {(y_test == 1).sum()}")

    # Training + Validation pooled for cross-validation tuning (keeping test set strictly untouched)
    X_tune = np.vstack([X_train, X_val])
    y_tune = np.concatenate([y_train, y_val])

    training_records = {}
    start_total = time.time()

    # -------------------------------------------------------------------------
    # 1. XGBOOST BASELINE (UNTUNED / DEFAULT SETTINGS)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 1: Training XGBoost Baseline (Default Parameters)")
    print("="*70)
    xgb_base = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=1,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=1
    )
    t0 = time.time()
    xgb_base.fit(X_train, y_train)
    xgb_base_time = time.time() - t0
    joblib.dump(xgb_base, os.path.join(models_dir, "xgboost_baseline.joblib"))
    print(f"XGBoost Baseline fitted in {xgb_base_time:.2f}s")

    # -------------------------------------------------------------------------
    # 2. XGBOOST FINE-TUNING (STRATIFIED 5-FOLD CROSS-VALIDATION)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 2: Fine-Tuning XGBoost via 5-Fold Stratified Cross-Validation")
    print("="*70)
    param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 3]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=1),
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    t0 = time.time()
    grid_search.fit(X_tune, y_tune)
    tuning_time = time.time() - t0

    best_params = grid_search.best_params_
    best_cv_f1 = float(grid_search.best_score_)
    print(f"Grid search completed in {tuning_time:.2f}s across {len(grid_search.cv_results_['params'])} candidates (5 folds each).")
    print(f"Selected Best Parameters: {best_params}")
    print(f"Best 5-Fold CV F1 Score: {best_cv_f1:.4f}")

    best_xgb = grid_search.best_estimator_
    joblib.dump(best_xgb, os.path.join(models_dir, "xgboost_tuned.joblib"))
    # Also save as primary xgboost.joblib for seamless inference engine loading
    joblib.dump(best_xgb, os.path.join(models_dir, "xgboost.joblib"))
    print("Saved fine-tuned XGBoost model to models/xgboost_tuned.joblib and models/xgboost.joblib")

    # -------------------------------------------------------------------------
    # 3. BASELINE ENSEMBLES & COMPARISON MODELS
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 3: Training Standard Baseline Models for Fair Comparison")
    print("="*70)

    baselines = {
        "Random Forest": (
            RandomForestClassifier(n_estimators=150, max_depth=12, random_state=random_state, n_jobs=1),
            "random_forest.joblib",
            "Bagging ensemble of 150 decorrelated decision trees; evaluates variance reduction"
        ),
        "Gradient Boosting (GBM)": (
            GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=random_state),
            "gradient_boosting.joblib",
            "Sequential residual gradient boosting; evaluates standard boosting without second-order Taylor expansion"
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=random_state),
            "logistic_regression.joblib",
            "Linear classification baseline with L2 regularization; tests linear separability"
        )
    }

    for name, (model, filename, purpose) in baselines.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - t0
        joblib.dump(model, os.path.join(models_dir, filename))
        print(f"Trained {name:25s} in {duration:.2f}s | Purpose: {purpose}")

    # -------------------------------------------------------------------------
    # 4. RECORD TUNING METADATA
    # -------------------------------------------------------------------------
    tuning_metadata = {
        "project_name": "CircuitGuard AI",
        "primary_model": "XGBoost (Fine-Tuned)",
        "tuning_strategy": "5-Fold Stratified Cross-Validation on Training + Validation Data",
        "scoring_metric": "f1 (Positive class = 1: Maintenance Required)",
        "random_state": random_state,
        "search_space": param_grid,
        "total_candidates": len(grid_search.cv_results_["params"]),
        "total_fits": len(grid_search.cv_results_["params"]) * 5,
        "selected_best_parameters": best_params,
        "best_cv_f1_score": best_cv_f1,
        "tuning_walltime_seconds": round(tuning_time, 2),
        "training_walltime_total_seconds": round(time.time() - start_total, 2)
    }

    with open(os.path.join(models_dir, "model_metadata.json"), "w") as f:
        json.dump(tuning_metadata, f, indent=2)
    print(f"\nSaved tuning metadata to {os.path.join(models_dir, 'model_metadata.json')}")

    # Clean up any legacy TabNet files from models directory
    for legacy_file in ["tabnet_finetuned.zip", "tabnet_pretrainer.zip", "tabnet_scratch.zip"]:
        path = os.path.join(models_dir, legacy_file)
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed legacy artifact: {legacy_file}")

    print("\nModel training and tuning pipeline completed successfully!")

if __name__ == "__main__":
    train_and_tune_models()
