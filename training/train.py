"""
CircuitGuard AI - Model Training Pipeline
Implements self-supervised TabNet pretraining, weight transfer, fine-tuning,
and comparison against baseline models (Random Forest, XGBoost, LightGBM, MLP, Logistic Regression).
"""

import os
import sys

# Prevent OpenMP runtime collision on macOS ARM64
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import json
import warnings
import time
import numpy as np
import torch
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def train_all_models(
    processed_dir: str = "dataset/processed",
    models_dir: str = "models",
    random_state: int = 42
):
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    os.makedirs(models_dir, exist_ok=True)

    print("Loading preprocessed training and validation arrays...")
    X_train = np.load(os.path.join(processed_dir, "X_train_trans.npy")).astype(np.float32)
    y_train = np.load(os.path.join(processed_dir, "y_train.npy")).astype(np.int64)
    X_val = np.load(os.path.join(processed_dir, "X_val_trans.npy")).astype(np.float32)
    y_val = np.load(os.path.join(processed_dir, "y_val.npy")).astype(np.int64)

    with open(os.path.join(models_dir, "feature_metadata.json"), "r") as f:
        metadata = json.load(f)
    feature_names = metadata["transformed_feature_names"]

    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    print(f"Number of input features: {len(feature_names)}")

    val_results = {}
    training_histories = {}
    start_all = time.time()

    # -------------------------------------------------------------
    # 1. TABNET TRANSFER LEARNING: SELF-SUPERVISED PRETRAINING
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 1: TabNet Self-Supervised Pretraining (Transfer Learning)")
    print("="*60)
    t0 = time.time()
    pretrainer = TabNetPretrainer(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax",
        n_d=16,
        n_a=16,
        n_steps=3,
        gamma=1.3,
        seed=random_state,
        verbose=1
    )
    pretrainer.fit(
        X_train=X_train,
        eval_set=[X_val],
        max_epochs=30,
        patience=10,
        batch_size=128,
        virtual_batch_size=64,
        pretraining_ratio=0.5
    )
    pretrain_time = time.time() - t0
    pretrainer_path = os.path.join(models_dir, "tabnet_pretrainer")
    pretrainer.save_model(pretrainer_path)
    print(f"Unsupervised pretraining complete in {pretrain_time:.2f}s. Saved to {pretrainer_path}.zip")

    # -------------------------------------------------------------
    # 2. TABNET SUPERVISED FINE-TUNING (WEIGHT TRANSFER)
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 2: Supervised TabNet Fine-Tuning (with Pretrained Weights)")
    print("="*60)
    t0 = time.time()
    tabnet_finetuned = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        n_d=16,
        n_a=16,
        n_steps=3,
        gamma=1.3,
        seed=random_state,
        verbose=1
    )
    tabnet_finetuned.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=["val"],
        eval_metric=["accuracy", "auc"],
        max_epochs=40,
        patience=12,
        batch_size=128,
        virtual_batch_size=64,
        from_unsupervised=pretrainer
    )
    finetune_time = time.time() - t0
    tabnet_ft_path = os.path.join(models_dir, "tabnet_finetuned")
    tabnet_finetuned.save_model(tabnet_ft_path)
    print(f"Fine-tuning complete in {finetune_time:.2f}s. Saved to {tabnet_ft_path}.zip")

    ft_preds = tabnet_finetuned.predict(X_val)
    ft_probs = tabnet_finetuned.predict_proba(X_val)[:, 1]
    val_results["TabNet (Pretrained + Fine-Tuned)"] = {
        "accuracy": float(accuracy_score(y_val, ft_preds)),
        "precision": float(precision_score(y_val, ft_preds)),
        "recall": float(recall_score(y_val, ft_preds)),
        "f1": float(f1_score(y_val, ft_preds)),
        "roc_auc": float(roc_auc_score(y_val, ft_probs)),
        "training_time_seconds": round(pretrain_time + finetune_time, 2),
        "approach": "Transfer Learning (Self-supervised pretraining + Fine-tuning)"
    }
    training_histories["TabNet_FineTuned"] = {
        "loss": [float(x) for x in tabnet_finetuned.history["loss"]],
        "val_accuracy": [float(x) for x in tabnet_finetuned.history["val_accuracy"]],
        "val_auc": [float(x) for x in tabnet_finetuned.history["val_auc"]]
    }

    # -------------------------------------------------------------
    # 3. BASELINE MODELS FOR COMPARISON (4 MODELS TOTAL IN SUITE)
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 3: Baseline Models Training")
    print("="*60)

    baselines = {
        "Random Forest": (
            RandomForestClassifier(n_estimators=150, max_depth=12, random_state=random_state, n_jobs=1),
            "random_forest.joblib"
        ),
        "XGBoost": (
            XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=random_state, eval_metric="logloss", n_jobs=1),
            "xgboost.joblib"
        ),
        "Gradient Boosting (GBM)": (
            GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=random_state),
            "gradient_boosting.joblib"
        )
    }

    for name, (model, filename) in baselines.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - t0

        save_path = os.path.join(models_dir, filename)
        joblib.dump(model, save_path)

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

        val_results[name] = {
            "accuracy": float(accuracy_score(y_val, preds)),
            "precision": float(precision_score(y_val, preds)),
            "recall": float(recall_score(y_val, preds)),
            "f1": float(f1_score(y_val, preds)),
            "roc_auc": float(roc_auc_score(y_val, probs)) if probs is not None else 0.0,
            "training_time_seconds": round(duration, 2),
            "approach": "Baseline Supervised"
        }
        print(f"Trained {name:22s} in {duration:.2f}s | Val Acc: {val_results[name]['accuracy']:.4f} | Val F1: {val_results[name]['f1']:.4f} | Val AUC: {val_results[name]['roc_auc']:.4f}")

    # -------------------------------------------------------------
    # 5. SUMMARY AND METADATA SAVING
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("VALIDATION SUMMARY (All Models)")
    print("="*60)
    for model_name, res in sorted(val_results.items(), key=lambda x: x[1]["f1"], reverse=True):
        print(f"{model_name:32s} | Acc: {res['accuracy']:.4f} | F1: {res['f1']:.4f} | AUC: {res['roc_auc']:.4f} | Time: {res['training_time_seconds']}s")

    # Save training history and metadata
    with open(os.path.join(models_dir, "training_history.json"), "w") as f:
        json.dump(training_histories, f, indent=2)

    model_metadata = {
        "project_name": "CircuitGuard AI",
        "primary_model": "TabNet (Pretrained + Fine-Tuned)",
        "approach": "Self-supervised feature reconstruction pretraining followed by supervised classification fine-tuning",
        "random_state": random_state,
        "input_dim": int(X_train.shape[1]),
        "feature_names": feature_names,
        "class_labels": {0: "No Maintenance Required", 1: "Maintenance Required"},
        "validation_benchmarks": val_results,
        "total_training_walltime_seconds": round(time.time() - start_all, 2),
        "device": "macOS M4 (Accelerate / MPS / CPU)"
    }

    with open(os.path.join(models_dir, "model_metadata.json"), "w") as f:
        json.dump(model_metadata, f, indent=2)
    print(f"\nSaved metadata to {os.path.join(models_dir, 'model_metadata.json')}")

    return val_results

if __name__ == "__main__":
    train_all_models()
