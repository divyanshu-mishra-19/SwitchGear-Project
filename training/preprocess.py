"""
CircuitGuard AI - Data Preprocessing Pipeline
Splits data into train/val/test, fits preprocessing strictly on training data,
and persists splits and fitted pipeline.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_and_run_preprocessing(
    raw_path: str = "dataset/raw/HV Circuit Breaker Maintenance Data.xlsx",
    output_dir: str = "dataset/processed",
    models_dir: str = "models",
    random_state: int = 42
):
    print(f"Reading raw dataset from {raw_path}...")
    df = pd.read_excel(raw_path)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    target_col = "Maintenance_required"
    cat_cols = ["Product_variant", "Breaker_status", "Heater_status", "Last_trip_coil_energized"]
    num_cols = [c for c in df.columns if c not in cat_cols and c != target_col]

    print(f"Total samples: {len(df)}")
    print(f"Numerical features: {len(num_cols)}")
    print(f"Categorical features: {len(cat_cols)}")

    # Stratified splitting: 70% Train, 15% Validation, 15% Test
    # Using composite stratification of Product_variant and Maintenance_required
    strat_key = df["Product_variant"].astype(str) + "_" + df[target_col].astype(str)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=strat_key
    )

    temp_strat = temp_df["Product_variant"].astype(str) + "_" + temp_df[target_col].astype(str)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_strat
    )

    print(f"Split sizes -> Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%), "
          f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%), "
          f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    # Save raw split CSVs
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(f"Saved train.csv, val.csv, test.csv to {output_dir}/")

    # Separate features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col].values
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].values

    # Preprocessing pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ]
    )

    # Fit ONLY on training data
    print("Fitting preprocessor strictly on training data...")
    preprocessor.fit(X_train)

    # Save fitted preprocessor
    pipeline_path = os.path.join(models_dir, "preprocessing_pipeline.joblib")
    joblib.dump(preprocessor, pipeline_path)
    print(f"Saved fitted pipeline to {pipeline_path}")

    # Transform splits
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    feature_names = list(preprocessor.get_feature_names_out())
    print(f"Transformed feature dimension: {X_train_trans.shape[1]}")

    # Build rich feature metadata for UI / validation / API
    units_map = {
        "SF6_density": "kg/m³",
        "SF6_temperature": "°C",
        "SF6_dew_point": "°C",
        "Motor_run_time": "s",
        "Motor_runs_per_day": "runs/day",
        "Motor_I2T": "A²·s",
        "Line_voltage": "V",
        "Ambient_cabinet_temperature": "°C",
        "Days_since_last_operation": "days",
        "Contact_life": "points",
        "Clearing_time": "s",
        "Open_latch_time": "s",
        "Close_latch_time": "s",
        "Opening_travel_time": "s",
        "Closing_travel_time": "s",
        "Total_operation_count": "cycles",
        "Non_fault_operation_count": "cycles",
        "Fault_operations_count": "cycles",
        "Accumulated_motor_run_time": "s",
        "Heater_current": "A",
        "Open_average_velocity": "m/s",
        "Close_average_velocity": "m/s",
        "Restrike_occurrence": "flag (0/1)",
        "Open_operation_arcing_time": "s"
    }

    feature_metadata = {
        "raw_features_order": [c for c in df.columns if c != target_col],
        "target_col": target_col,
        "target_classes": {0: "No Maintenance Required", 1: "Maintenance Required"},
        "transformed_feature_names": feature_names,
        "numerical_features": {},
        "categorical_features": {}
    }

    for col in num_cols:
        series = train_df[col].dropna()
        feature_metadata["numerical_features"][col] = {
            "type": "float" if series.dtype == "float64" else "int",
            "unit": units_map.get(col, ""),
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "mean": float(series.mean()),
            "has_nulls": bool(train_df[col].isnull().any())
        }

    for col in cat_cols:
        series = train_df[col].dropna()
        feature_metadata["categorical_features"][col] = {
            "type": "category",
            "options": sorted([str(x) for x in series.unique()]),
            "mode": str(series.mode()[0])
        }

    metadata_path = os.path.join(models_dir, "feature_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(feature_metadata, f, indent=2)
    print(f"Saved feature metadata to {metadata_path}")

    # Also save transformed numpy arrays for fast training loading
    np.save(os.path.join(output_dir, "X_train_trans.npy"), X_train_trans)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_val_trans.npy"), X_val_trans)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "X_test_trans.npy"), X_test_trans)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    print(f"Saved transformed matrices to {output_dir}/")

    return preprocessor, feature_metadata

if __name__ == "__main__":
    build_and_run_preprocessing()

