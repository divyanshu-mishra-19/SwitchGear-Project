"""
CircuitGuard AI - Inference Engine
Loads trained models (XGBoost Fine-Tuned, XGBoost Baseline, Random Forest, Gradient Boosting)
and preprocessing pipeline once at startup.
Generates predictions, calibrated class probabilities, and local feature importance rankings.
TabNet has been completely removed in accordance with project requirements.
"""

import os
import sys

# Prevent OpenMP runtime conflict on macOS ARM64
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import json
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class ModelInferenceEngine:
    def __init__(self, models_dir: str = "models", reports_dir: str = "reports"):
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.pipeline = None
        self.metadata = None
        self.metrics = None
        self.models = {}
        self.default_model_name = "XGBoost (Fine-Tuned)"
        self.load_all_artifacts()

    def load_all_artifacts(self):
        print(f"Initializing Inference Engine from {self.models_dir}...")

        # 1. Load Preprocessing Pipeline
        pipeline_path = os.path.join(self.models_dir, "preprocessing_pipeline.joblib")
        if os.path.exists(pipeline_path):
            self.pipeline = joblib.load(pipeline_path)
            print("Loaded fitted scikit-learn preprocessing pipeline.")
        else:
            raise FileNotFoundError(f"Preprocessing pipeline not found at {pipeline_path}")

        # 2. Load Feature Metadata
        metadata_path = os.path.join(self.models_dir, "feature_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            print("Loaded feature metadata.")
        else:
            raise FileNotFoundError(f"Feature metadata not found at {metadata_path}")

        # 3. Load Evaluation Metrics
        metrics_path = os.path.join(self.reports_dir, "model_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)
            print("Loaded evaluation metrics.")
        else:
            self.metrics = {}

        # 4. Load Models Suite
        models_to_load = {
            "XGBoost (Fine-Tuned)": "xgboost_tuned.joblib",
            "XGBoost (Baseline)": "xgboost_baseline.joblib",
            "Random Forest": "random_forest.joblib",
            "Gradient Boosting (GBM)": "gradient_boosting.joblib"
        }

        # Fallback if xgboost_tuned is named xgboost.joblib
        for name, filename in models_to_load.items():
            filepath = os.path.join(self.models_dir, filename)
            if not os.path.exists(filepath) and name == "XGBoost (Fine-Tuned)":
                filepath = os.path.join(self.models_dir, "xgboost.joblib")

            if os.path.exists(filepath):
                try:
                    self.models[name] = joblib.load(filepath)
                    print(f"Loaded model: {name}")
                except Exception as e:
                    print(f"Warning: Failed to load {name}: {e}")

        if not self.models:
            raise RuntimeError("No models were successfully loaded!")

    def get_available_models(self):
        return list(self.models.keys())

    def _prepare_dataframe(self, input_dict: dict) -> pd.DataFrame:
        raw_order = self.metadata["raw_features_order"]
        row_data = {}
        for col in raw_order:
            val = input_dict.get(col, None)
            if val is None or val == "" or (isinstance(val, float) and np.isnan(val)):
                row_data[col] = np.nan
            else:
                row_data[col] = val
        return pd.DataFrame([row_data])

    def predict_single(self, input_data: dict, model_name: str = None) -> dict:
        if not model_name or model_name not in self.models:
            # Handle aliases
            if model_name in ["XGBoost", "xgboost"]:
                model_name = "XGBoost (Fine-Tuned)"
            else:
                model_name = self.default_model_name if self.default_model_name in self.models else list(self.models.keys())[0]

        model = self.models[model_name]
        df_raw = self._prepare_dataframe(input_data)

        # Transform using fitted pipeline
        X_trans = self.pipeline.transform(df_raw)

        preds = model.predict(X_trans)
        pred_class = int(preds[0])
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_trans)[0]
            prob_0 = float(probs[0])
            prob_1 = float(probs[1])
        else:
            prob_1 = 1.0 if pred_class == 1 else 0.0
            prob_0 = 1.0 - prob_1

        # Local feature attribution
        if hasattr(model, "feature_importances_"):
            feature_scores = model.feature_importances_ * np.abs(X_trans[0])
        elif hasattr(model, "coef_"):
            feature_scores = np.abs(model.coef_[0]) * np.abs(X_trans[0])
        else:
            feature_scores = np.abs(X_trans[0])

        confidence = prob_1 if pred_class == 1 else prob_0
        pred_label = self.metadata["target_classes"].get(str(pred_class), "Maintenance Required" if pred_class == 1 else "No Maintenance Required")

        # Rank Top Influential Features
        feature_names = self.metadata["transformed_feature_names"]
        top_indices = np.argsort(feature_scores)[-5:][::-1]

        units_map = {k: v.get("unit", "") for k, v in self.metadata["numerical_features"].items()}
        top_features_list = []
        for idx in top_indices:
            raw_feat_name = feature_names[idx].replace("num__", "").replace("cat__", "")
            parent_feat = raw_feat_name.split("_")[0] if "_" in raw_feat_name else raw_feat_name
            actual_val = input_data.get(raw_feat_name, input_data.get(parent_feat, "N/A"))
            unit = units_map.get(raw_feat_name, units_map.get(parent_feat, ""))
            top_features_list.append({
                "feature": raw_feat_name,
                "importance_score": round(float(feature_scores[idx]), 4),
                "reading_value": actual_val,
                "unit": unit
            })

        # Factual explanation (explicitly clarifying estimated probability vs guaranteed accuracy)
        top_f1 = top_features_list[0]["feature"]
        top_f2 = top_features_list[1]["feature"]
        if pred_class == 1:
            explanation = f"Primary indicators triggering maintenance: '{top_f1}' and '{top_f2}'."
        else:
            explanation = f"Key operational stability indicators: '{top_f1}' and '{top_f2}'."

        key_summary = {
            "Product Variant": input_data.get("Product_variant", "N/A"),
            "SF6 Density": f"{input_data.get('SF6_density', 'N/A')} kg/m³",
            "Cabinet Temperature": f"{input_data.get('Ambient_cabinet_temperature', 'N/A')} °C",
            "Motor Run Time": f"{input_data.get('Motor_run_time', 'N/A')} s",
            "Total Operations": f"{input_data.get('Total_operation_count', 'N/A')} cycles",
            "Breaker Status": input_data.get("Breaker_status", "N/A")
        }

        return {
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
            "estimated_class_probability": round(confidence, 4),
            "probabilities": {
                "No Maintenance Required": round(prob_0, 4),
                "Maintenance Required": round(prob_1, 4)
            },
            "model_used": model_name,
            "key_input_summary": key_summary,
            "top_influential_features": top_features_list,
            "explanation": explanation,
            "disclaimer": (
                "Prediction probabilities reflect statistical classification estimates based on training telemetry. "
                "They are intended for decision-support and do not replace certified electrical inspection or safety guidelines."
            )
        }

    def compare_all_models(self, input_data: dict) -> dict:
        results = []
        votes = {0: 0, 1: 0}

        # Focus comparison on the 4 primary models
        primary_models = [m for m in ["XGBoost (Fine-Tuned)", "XGBoost (Baseline)", "Random Forest", "Gradient Boosting (GBM)"] if m in self.models]
        if not primary_models:
            primary_models = self.get_available_models()

        for model_name in primary_models:
            res = self.predict_single(input_data, model_name=model_name)
            pred_cls = res["predicted_class"]
            votes[pred_cls] += 1

            test_metric = self.metrics.get(model_name, {})
            cm_dict = test_metric.get("confusion_matrix", {})

            results.append({
                "model_name": model_name,
                "predicted_class": pred_cls,
                "predicted_label": res["predicted_label"],
                "confidence": res["confidence"],
                "estimated_class_probability": res["confidence"],
                "probabilities": res["probabilities"],
                "test_accuracy": test_metric.get("accuracy", None),
                "test_f1": test_metric.get("f1_score", None),
                "test_precision": test_metric.get("precision", None),
                "test_recall": test_metric.get("recall", None),
                "test_roc_auc": test_metric.get("roc_auc", None),
                "confusion_matrix": cm_dict
            })

        majority_class = 1 if votes[1] > votes[0] else 0
        total_models = len(results)
        agreement = (max(votes.values()) / total_models) * 100 if total_models > 0 else 0

        consensus_label = "Maintenance Required" if majority_class == 1 else "No Maintenance Required"

        return {
            "models_evaluated": total_models,
            "consensus_label": consensus_label,
            "agreement_percentage": round(agreement, 1),
            "vote_summary": {
                "Maintenance Required": votes[1],
                "No Maintenance Required": votes[0]
            },
            "comparisons": results
        }

    def predict_batch(self, df_input: pd.DataFrame, model_name: str = None) -> dict:
        if not model_name or model_name not in self.models:
            model_name = self.default_model_name if self.default_model_name in self.models else list(self.models.keys())[0]

        model = self.models[model_name]
        raw_order = self.metadata["raw_features_order"]

        for col in raw_order:
            if col not in df_input.columns:
                df_input[col] = np.nan
        df_ordered = df_input[raw_order]

        X_trans = self.pipeline.transform(df_ordered)
        preds = model.predict(X_trans)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_trans)
        else:
            probs = np.zeros((len(preds), 2))
            for i, p in enumerate(preds):
                probs[i, p] = 1.0

        items = []
        maint_count = 0
        no_maint_count = 0

        for i in range(len(preds)):
            cls = int(preds[i])
            conf = float(probs[i, cls])
            lbl = "Maintenance Required" if cls == 1 else "No Maintenance Required"
            if cls == 1:
                maint_count += 1
            else:
                no_maint_count += 1

            items.append({
                "row_index": i,
                "predicted_class": cls,
                "predicted_label": lbl,
                "confidence": round(conf, 4),
                "estimated_class_probability": round(conf, 4),
                "probabilities": {
                    "No Maintenance Required": round(float(probs[i, 0]), 4),
                    "Maintenance Required": round(float(probs[i, 1]), 4)
                }
            })

        return {
            "total_records": len(preds),
            "model_used": model_name,
            "predictions": items,
            "maintenance_required_count": maint_count,
            "no_maintenance_count": no_maint_count
        }

# Global singleton engine instance
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = ModelInferenceEngine()
    return engine
