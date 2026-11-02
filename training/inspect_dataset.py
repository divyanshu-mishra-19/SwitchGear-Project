"""
CircuitGuard AI - Dataset Inspection & Exploratory Data Analysis Script
Analyzes the HV Circuit Breaker Maintenance Data and generates comprehensive reports.
"""

import os
import json
import numpy as np
import pandas as pd

def inspect_dataset(file_path: str = "dataset/raw/HV Circuit Breaker Maintenance Data.xlsx"):
    print(f"Loading raw dataset from {file_path}...")
    df = pd.read_excel(file_path)

    os.makedirs("reports", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    num_rows, num_cols = df.shape
    columns = list(df.columns)
    target_col = "Maintenance_required"
    feature_cols = [c for c in columns if c != target_col]

    # Types and missing values
    dtypes = {col: str(df[col].dtype) for col in columns}
    missing_counts = df.isnull().sum().to_dict()
    missing_pcts = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    duplicate_count = int(df.duplicated().sum())

    # Categorical and numerical columns
    cat_cols = list(df.select_dtypes(include=["object"]).columns)
    num_cols_list = list(df.select_dtypes(include=["number"]).columns)
    if target_col in num_cols_list:
        num_cols_list.remove(target_col)

    cat_stats = {}
    for col in cat_cols:
        cat_stats[col] = {
            "unique_count": int(df[col].nunique()),
            "unique_values": [str(v) for v in df[col].unique()],
            "value_counts": {str(k): int(v) for k, v in df[col].value_counts().items()}
        }

    # Special binary/float column check
    if "Restrike_occurrence" in df.columns:
        cat_stats["Restrike_occurrence"] = {
            "unique_count": int(df["Restrike_occurrence"].nunique()),
            "unique_values": [str(v) for v in df["Restrike_occurrence"].dropna().unique()],
            "value_counts": {str(k): int(v) for k, v in df["Restrike_occurrence"].value_counts(dropna=False).items()}
        }

    # Numerical statistics and outlier detection (IQR method)
    num_stats = {}
    outlier_counts = {}
    for col in num_cols_list:
        series = df[col].dropna()
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = int(((series < lower_bound) | (series > upper_bound)).sum())
        outlier_counts[col] = outliers

        num_stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "q25": q1,
            "median": float(series.median()),
            "q75": q3,
            "max": float(series.max()),
            "outliers_iqr": outliers
        }

    # Target class distribution
    target_counts = {int(k): int(v) for k, v in df[target_col].value_counts().items()}
    target_pcts = {int(k): float(v) for k, v in df[target_col].value_counts(normalize=True).round(4).items()}

    # Correlation with target
    num_target_df = df[num_cols_list + [target_col]].dropna()
    correlations = num_target_df.corr()[target_col].drop(target_col).sort_values(ascending=False).to_dict()

    # Product variant breakdown of missing values
    missing_by_variant = {}
    for variant, group in df.groupby("Product_variant"):
        missing_by_variant[str(variant)] = {
            col: int(group[col].isnull().sum())
            for col in feature_cols if group[col].isnull().sum() > 0
        }

    # Prepare structured JSON report
    quality_report = {
        "dataset_metadata": {
            "file_name": os.path.basename(file_path),
            "rows": num_rows,
            "columns": num_cols,
            "target_column": target_col,
            "feature_count": len(feature_cols),
            "duplicates": duplicate_count,
            "total_missing_cells": int(df.isnull().sum().sum()),
            "total_missing_percentage": round(float(df.isnull().sum().sum() / (num_rows * num_cols) * 100), 2)
        },
        "target_distribution": {
            "counts": target_counts,
            "proportions": target_pcts,
            "labels": {0: "No Maintenance Required", 1: "Maintenance Required"}
        },
        "categorical_features": cat_stats,
        "numerical_features": num_stats,
        "missing_values": {col: {"count": missing_counts[col], "percentage": missing_pcts[col]} for col in columns if missing_counts[col] > 0},
        "missing_by_product_variant": missing_by_variant,
        "target_correlations": {k: round(v, 4) for k, v in correlations.items()},
        "domain_assessment": {
            "data_nature": "Semi-synthetically derived by utility engineers for critical electrical infrastructure maintenance research",
            "independent_measurements": True,
            "grouping_variable": "Product_variant (represents 5 utility operating fleets A through E)",
            "timestamp_present": False,
            "identifier_present": False
        }
    }

    with open("reports/data_quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2)

    # Generate Markdown dataset report
    md_content = f"""# CircuitGuard AI - Dataset Inspection and Quality Report

## 1. Executive Summary
- **Dataset Source**: High-Voltage Circuit Breaker Maintenance Data (`HV Circuit Breaker Maintenance Data.xlsx`)
- **Total Records**: {num_rows:,}
- **Total Columns**: {num_cols} ({len(feature_cols)} features + 1 target)
- **Target Variable**: `{target_col}` (0 = No Maintenance Required, 1 = Maintenance Required)
- **Duplicates**: {duplicate_count}
- **Total Missing Cells**: {quality_report['dataset_metadata']['total_missing_cells']:,} ({quality_report['dataset_metadata']['total_missing_percentage']}%)
- **Data Provenance**: Semi-synthetically derived based on substation utility operational monitoring standards, augmented with maintenance failure triggers to evaluate federated/tabular models.

---

## 2. Target Class Distribution
| Target Code | Label | Count | Percentage |
| :--- | :--- | :--- | :--- |
| `0` | No Maintenance Required | {target_counts[0]:,} | {target_pcts[0]*100:.1f}% |
| `1` | Maintenance Required | {target_counts[1]:,} | {target_pcts[1]*100:.1f}% |
| **Total** | | **{num_rows:,}** | **100.0%** |

*Observation*: Mild class imbalance (59.3% vs 40.7%), representative of operational settings with augmented failure examples.

---

## 3. Structural Missingness by Product Variant
The dataset contains 5 product variants / utility fleets (`A`, `B`, `C`, `D`, `E`) with 1,000 samples each. Missing values are **strictly structural** (Missing Not At Random / MNAR), reflecting sensor equipment variation across utilities:

| Feature Name | Missing Count | Missing Pct | Variants Affected |
| :--- | :--- | :--- | :--- |
| `Days_since_last_operation` | 1,000 | 20.0% | Variant C (1,000) |
| `Clearing_time` | 1,000 | 20.0% | Variant C (1,000) |
| `Accumulated_motor_run_time` | 2,000 | 40.0% | Variant A (1,000), Variant C (1,000) |
| `Heater_current` | 2,000 | 40.0% | Variant A (1,000), Variant C (1,000) |
| `Open_average_velocity` | 2,000 | 40.0% | Variant A (1,000), Variant C (1,000) |
| `Close_average_velocity` | 2,000 | 40.0% | Variant A (1,000), Variant C (1,000) |
| `Restrike_occurrence` | 2,000 | 40.0% | Variant A (1,000), Variant B (1,000) |
| `Open_operation_arcing_time` | 3,000 | 60.0% | Variant A (1,000), Variant B (1,000), Variant C (1,000) |

*Note*: Variants `D` and `E` have complete 28-sensor telemetry (0 missing values).

---

## 4. Feature Dictionary and Physical Units

| # | Feature Name | Data Type | Physical Unit / Domain | Range [Min - Max] | Mean ± Std |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Product_variant` | Categorical | Utility Fleet ID | [A, B, C, D, E] | 5 fleets |
| 2 | `Breaker_status` | Categorical | Switch State | [Closed, Open] | 79.7% Closed |
| 3 | `SF6_density` | Float | Gas Density (kg/m³) | [{num_stats['SF6_density']['min']:.2f} - {num_stats['SF6_density']['max']:.2f}] | {num_stats['SF6_density']['mean']:.2f} ± {num_stats['SF6_density']['std']:.2f} |
| 4 | `SF6_temperature` | Float | Gas Temperature (°C) | [{num_stats['SF6_temperature']['min']:.2f} - {num_stats['SF6_temperature']['max']:.2f}] | {num_stats['SF6_temperature']['mean']:.2f} ± {num_stats['SF6_temperature']['std']:.2f} |
| 5 | `SF6_dew_point` | Float | Dew Point Moisture (°C) | [{num_stats['SF6_dew_point']['min']:.2f} - {num_stats['SF6_dew_point']['max']:.2f}] | {num_stats['SF6_dew_point']['mean']:.2f} ± {num_stats['SF6_dew_point']['std']:.2f} |
| 6 | `Motor_run_time` | Float | Charging Motor Time (s) | [{num_stats['Motor_run_time']['min']:.2f} - {num_stats['Motor_run_time']['max']:.2f}] | {num_stats['Motor_run_time']['mean']:.2f} ± {num_stats['Motor_run_time']['std']:.2f} |
| 7 | `Motor_runs_per_day` | Integer | Daily Cycles (count/day) | [{num_stats['Motor_runs_per_day']['min']:.0f} - {num_stats['Motor_runs_per_day']['max']:.0f}] | {num_stats['Motor_runs_per_day']['mean']:.2f} ± {num_stats['Motor_runs_per_day']['std']:.2f} |
| 8 | `Motor_I2T` | Float | Thermal Stress (A²·s) | [{num_stats['Motor_I2T']['min']:.2f} - {num_stats['Motor_I2T']['max']:.2f}] | {num_stats['Motor_I2T']['mean']:.2f} ± {num_stats['Motor_I2T']['std']:.2f} |
| 9 | `Heater_status` | Categorical | Cabinet Heater | [On, Off] | 69.8% Off |
| 10 | `Line_voltage` | Float | Grid Line Voltage (V) | [{num_stats['Line_voltage']['min']:,.0f} - {num_stats['Line_voltage']['max']:,.0f}] | {num_stats['Line_voltage']['mean']:,.0f} ± {num_stats['Line_voltage']['std']:,.0f} |
| 11 | `Ambient_cabinet_temperature` | Float | Cabinet Temperature (°C) | [{num_stats['Ambient_cabinet_temperature']['min']:.1f} - {num_stats['Ambient_cabinet_temperature']['max']:.1f}] | {num_stats['Ambient_cabinet_temperature']['mean']:.1f} ± {num_stats['Ambient_cabinet_temperature']['std']:.1f} |
| 12 | `Days_since_last_operation` | Float | Idle Elapsed Days (days) | [{num_stats['Days_since_last_operation']['min']:.0f} - {num_stats['Days_since_last_operation']['max']:.0f}] | {num_stats['Days_since_last_operation']['mean']:.1f} ± {num_stats['Days_since_last_operation']['std']:.1f} |
| 13 | `Contact_life` | Float | Wear Resistance Rating | [{num_stats['Contact_life']['min']:,.0f} - {num_stats['Contact_life']['max']:,.0f}] | {num_stats['Contact_life']['mean']:,.0f} ± {num_stats['Contact_life']['std']:,.0f} |
| 14 | `Last_trip_coil_energized` | Categorical | Tripping Circuit Coil | [TC1, TC2] | 50.4% TC1 |
| 15 | `Clearing_time` | Float | Fault Interrupt Time (s) | [{num_stats['Clearing_time']['min']:.3f} - {num_stats['Clearing_time']['max']:.3f}] | {num_stats['Clearing_time']['mean']:.3f} ± {num_stats['Clearing_time']['std']:.3f} |
| 16 | `Open_latch_time` | Float | Open Mechanism Latch (s) | [{num_stats['Open_latch_time']['min']:.3f} - {num_stats['Open_latch_time']['max']:.3f}] | {num_stats['Open_latch_time']['mean']:.3f} ± {num_stats['Open_latch_time']['std']:.3f} |
| 17 | `Close_latch_time` | Float | Close Mechanism Latch (s) | [{num_stats['Close_latch_time']['min']:.3f} - {num_stats['Close_latch_time']['max']:.3f}] | {num_stats['Close_latch_time']['mean']:.3f} ± {num_stats['Close_latch_time']['std']:.3f} |
| 18 | `Opening_travel_time` | Float | Travel Duration Open (s) | [{num_stats['Opening_travel_time']['min']:.3f} - {num_stats['Opening_travel_time']['max']:.3f}] | {num_stats['Opening_travel_time']['mean']:.3f} ± {num_stats['Opening_travel_time']['std']:.3f} |
| 19 | `Closing_travel_time` | Float | Travel Duration Close (s) | [{num_stats['Closing_travel_time']['min']:.3f} - {num_stats['Closing_travel_time']['max']:.3f}] | {num_stats['Closing_travel_time']['mean']:.3f} ± {num_stats['Closing_travel_time']['std']:.3f} |
| 20 | `Total_operation_count` | Integer | Total Operations (cycles) | [{num_stats['Total_operation_count']['min']:,} - {num_stats['Total_operation_count']['max']:,}] | {num_stats['Total_operation_count']['mean']:,.0f} ± {num_stats['Total_operation_count']['std']:,.0f} |
| 21 | `Non_fault_operation_count` | Integer | Normal Switching Cycles | [{num_stats['Non_fault_operation_count']['min']:,} - {num_stats['Non_fault_operation_count']['max']:,}] | {num_stats['Non_fault_operation_count']['mean']:,.0f} ± {num_stats['Non_fault_operation_count']['std']:,.0f} |
| 22 | `Fault_operations_count` | Integer | Fault Interruption Count | [{num_stats['Fault_operations_count']['min']:,} - {num_stats['Fault_operations_count']['max']:,}] | {num_stats['Fault_operations_count']['mean']:,.0f} ± {num_stats['Fault_operations_count']['std']:,.0f} |
| 23 | `Accumulated_motor_run_time`| Float | Cumulative Motor Run (s) | [{num_stats['Accumulated_motor_run_time']['min']:.1f} - {num_stats['Accumulated_motor_run_time']['max']:.1f}] | {num_stats['Accumulated_motor_run_time']['mean']:.1f} ± {num_stats['Accumulated_motor_run_time']['std']:.1f} |
| 24 | `Heater_current` | Float | Current Drawn by Heater (A)| [{num_stats['Heater_current']['min']:.2f} - {num_stats['Heater_current']['max']:.2f}] | {num_stats['Heater_current']['mean']:.2f} ± {num_stats['Heater_current']['std']:.2f} |
| 25 | `Open_average_velocity` | Float | Contact Velocity Open (m/s)| [{num_stats['Open_average_velocity']['min']:.2f} - {num_stats['Open_average_velocity']['max']:.2f}] | {num_stats['Open_average_velocity']['mean']:.2f} ± {num_stats['Open_average_velocity']['std']:.2f} |
| 26 | `Close_average_velocity` | Float | Contact Velocity Close (m/s)|[{num_stats['Close_average_velocity']['min']:.2f} - {num_stats['Close_average_velocity']['max']:.2f}] | {num_stats['Close_average_velocity']['mean']:.2f} ± {num_stats['Close_average_velocity']['std']:.2f} |
| 27 | `Restrike_occurrence` | Float/Binary | Arc Restrike Flag | [0.0 (No), 1.0 (Yes)] | 9.6% restrike when logged |
| 28 | `Open_operation_arcing_time`| Float | Arc Duration (s) | [{num_stats['Open_operation_arcing_time']['min']:.3f} - {num_stats['Open_operation_arcing_time']['max']:.3f}] | {num_stats['Open_operation_arcing_time']['mean']:.3f} ± {num_stats['Open_operation_arcing_time']['std']:.3f} |

---

## 5. Correlation with Maintenance Required
The top linear correlations with maintenance requirement:
- Positive correlations (longer times / higher stress increase maintenance probability):
  - `Motor_run_time`: +{correlations['Motor_run_time']:.4f}
  - `Open_operation_arcing_time`: +{correlations['Open_operation_arcing_time']:.4f}
  - `Close_latch_time`: +{correlations['Close_latch_time']:.4f}
  - `Open_latch_time`: +{correlations['Open_latch_time']:.4f}
  - `Motor_I2T`: +{correlations['Motor_I2T']:.4f}
- Negative correlations (higher health / contact life decrease maintenance need):
  - `Non_fault_operation_count`: {correlations['Non_fault_operation_count']:.4f}
  - `Total_operation_count`: {correlations['Total_operation_count']:.4f}
  - `Contact_life`: {correlations['Contact_life']:.4f}
  - `Open_average_velocity`: {correlations['Open_average_velocity']:.4f}
  - `Close_average_velocity`: {correlations['Close_average_velocity']:.4f}
"""

    with open("reports/dataset_report.md", "w") as f:
        f.write(md_content)

    print("Inspection complete! Reports saved to reports/dataset_report.md and reports/data_quality_report.json")
    return quality_report

if __name__ == "__main__":
    inspect_dataset()

