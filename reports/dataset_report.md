# CircuitGuard AI - Dataset Inspection and Quality Report

## 1. Executive Summary
- **Dataset Source**: High-Voltage Circuit Breaker Maintenance Data (`HV Circuit Breaker Maintenance Data.xlsx`)
- **Total Records**: 5,000
- **Total Columns**: 29 (28 features + 1 target)
- **Target Variable**: `Maintenance_required` (0 = No Maintenance Required, 1 = Maintenance Required)
- **Duplicates**: 0
- **Total Missing Cells**: 15,000 (10.34%)
- **Data Provenance**: Semi-synthetically derived based on substation utility operational monitoring standards, augmented with maintenance failure triggers to evaluate federated/tabular models.

---

## 2. Target Class Distribution
| Target Code | Label | Count | Percentage |
| :--- | :--- | :--- | :--- |
| `0` | No Maintenance Required | 2,965 | 59.3% |
| `1` | Maintenance Required | 2,035 | 40.7% |
| **Total** | | **5,000** | **100.0%** |

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
| 3 | `SF6_density` | Float | Gas Density (kg/m³) | [5.50 - 6.96] | 6.01 ± 0.28 |
| 4 | `SF6_temperature` | Float | Gas Temperature (°C) | [0.01 - 50.00] | 20.29 ± 9.65 |
| 5 | `SF6_dew_point` | Float | Dew Point Moisture (°C) | [0.09 - 35.00] | 19.89 ± 7.70 |
| 6 | `Motor_run_time` | Float | Charging Motor Time (s) | [0.50 - 15.00] | 7.07 ± 4.29 |
| 7 | `Motor_runs_per_day` | Integer | Daily Cycles (count/day) | [0 - 7] | 0.97 ± 0.98 |
| 8 | `Motor_I2T` | Float | Thermal Stress (A²·s) | [0.10 - 25.00] | 8.90 ± 6.77 |
| 9 | `Heater_status` | Categorical | Cabinet Heater | [On, Off] | 69.8% Off |
| 10 | `Line_voltage` | Float | Grid Line Voltage (V) | [72,500 - 150,000] | 119,747 ± 14,851 |
| 11 | `Ambient_cabinet_temperature` | Float | Cabinet Temperature (°C) | [32.0 - 122.0] | 68.1 ± 14.4 |
| 12 | `Days_since_last_operation` | Float | Idle Elapsed Days (days) | [13 - 50] | 29.9 ± 5.5 |
| 13 | `Contact_life` | Float | Wear Resistance Rating | [10,000 - 84,575] | 33,801 ± 12,721 |
| 14 | `Last_trip_coil_energized` | Categorical | Tripping Circuit Coil | [TC1, TC2] | 50.4% TC1 |
| 15 | `Clearing_time` | Float | Fault Interrupt Time (s) | [0.020 - 0.100] | 0.046 ± 0.025 |
| 16 | `Open_latch_time` | Float | Open Mechanism Latch (s) | [0.050 - 0.150] | 0.112 ± 0.024 |
| 17 | `Close_latch_time` | Float | Close Mechanism Latch (s) | [0.050 - 0.150] | 0.112 ± 0.024 |
| 18 | `Opening_travel_time` | Float | Travel Duration Open (s) | [0.050 - 0.150] | 0.081 ± 0.019 |
| 19 | `Closing_travel_time` | Float | Travel Duration Close (s) | [0.050 - 0.150] | 0.081 ± 0.019 |
| 20 | `Total_operation_count` | Integer | Total Operations (cycles) | [3,894.0 - 6,105.0] | 4,400 ± 801 |
| 21 | `Non_fault_operation_count` | Integer | Normal Switching Cycles | [2,776.0 - 5,764.0] | 3,737 ± 718 |
| 22 | `Fault_operations_count` | Integer | Fault Interruption Count | [196.0 - 1,797.0] | 662 ± 246 |
| 23 | `Accumulated_motor_run_time`| Float | Cumulative Motor Run (s) | [10.0 - 1800.2] | 438.9 ± 304.1 |
| 24 | `Heater_current` | Float | Current Drawn by Heater (A)| [0.00 - 3.00] | 0.47 ± 0.76 |
| 25 | `Open_average_velocity` | Float | Contact Velocity Open (m/s)| [1.50 - 5.50] | 2.99 ± 0.85 |
| 26 | `Close_average_velocity` | Float | Contact Velocity Close (m/s)|[1.50 - 5.55] | 3.01 ± 0.86 |
| 27 | `Restrike_occurrence` | Float/Binary | Arc Restrike Flag | [0.0 (No), 1.0 (Yes)] | 9.6% restrike when logged |
| 28 | `Open_operation_arcing_time`| Float | Arc Duration (s) | [0.010 - 0.080] | 0.028 ± 0.017 |

---

## 5. Correlation with Maintenance Required
The top linear correlations with maintenance requirement:
- Positive correlations (longer times / higher stress increase maintenance probability):
  - `Motor_run_time`: +-0.0308
  - `Open_operation_arcing_time`: +0.1593
  - `Close_latch_time`: +0.0441
  - `Open_latch_time`: +0.0266
  - `Motor_I2T`: +0.1314
- Negative correlations (higher health / contact life decrease maintenance need):
  - `Non_fault_operation_count`: -0.3432
  - `Total_operation_count`: -0.2726
  - `Contact_life`: -0.1923
  - `Open_average_velocity`: 0.0000
  - `Close_average_velocity`: 0.0150
