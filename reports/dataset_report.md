# CircuitGuard AI: Dataset Documentation & Data Quality Report

---

## 1. Dataset Name and Concise Description
* **Dataset Name:** High-Voltage Circuit Breaker Maintenance Data (`HV Circuit Breaker Maintenance Data.xlsx`).
* **Concise Description:** A condition-monitoring dataset for high-voltage (HV) substation circuit breakers (ranging from 72.5 kV to 150 kV ratings across dead-tank, live-tank, and gas-insulated switchgear architectures). The dataset aggregates physical sensor telemetry spanning SF6 insulating gas condition, mechanical operating latch and travel timings, spring charging motor thermal stress, contact kinematics, switching operation counters, and an inspection-driven binary maintenance requirement target.
* **Operational Context:** In utility electrical substations, circuit breakers are critical switching devices that must interrupt high fault currents within tens of milliseconds. Utilities monitor sensor telemetry to transition from fixed-interval time-based servicing to condition-based predictive maintenance.

---

## 2. Official Source, Provenance, and Citation
* **Repository:** [sghosh27/Federated-Learning-for-Critical-Electrical-Infrastructure-Maintenance](https://github.com/sghosh27/Federated-Learning-for-Critical-Electrical-Infrastructure-Maintenance)
* **Dataset File:** `Datasets/HV Circuit Breaker Maintenance Data.xlsx`
* **License:** Apache License 2.0 (Open Source, Reproducible)
* **Official Research Paper Citation:** 
  > Ghosh S, Mittal G (2026). *Federated learning for critical electrical infrastructure - handling data heterogeneity for predictive maintenance of substation equipment*. Frontiers in Artificial Intelligence, 8:1697175.
* **Author Disclosure on Data Provenance:** As explicitly stated by the dataset authors in `SOURCE_README.md`:
  > *"This dataset is semi-synthetically derived for research and educational purposes and may not fully represent real-world operational data... The dataset comprises 5,000 samples of breaker readings, aggregated from five representative utilities... To enhance the training of machine learning models, the dataset has been augmented with an increased proportion of negative examples to ensure sufficient representation of failure cases."*

---

## 3. Number of Records and Features Used (How Counts Were Established)
* **Total Records:** **5,000 observations**.
  * Verified programmatically by inspecting the Excel spreadsheet via `pandas.read_excel()` (`df.shape[0] = 5000`).
  * 0 duplicate rows exist in the raw dataset (`df.duplicated().sum() == 0`).
  * The 5,000 records are equally distributed across five utility fleets (`Product_variant` A, B, C, D, and E with exactly 1,000 samples each).
* **Total Columns:** **29 columns**.
  * 28 input features + 1 target variable (`Maintenance_required`).
  * In the machine learning pipeline, all 28 raw features were retained (24 numerical features and 4 categorical features).
  * After one-hot encoding the 4 categorical features, the transformed input feature space consists of **35 columns**.

---

## 4. Input Features: Descriptions, Physical Units, and Value Ranges
The 28 input features are categorized below across physical breaker subsystems:

| # | Feature Name | Data Type | Physical Unit | Observed Range [Min – Max] | Mean ± Std | Physical Description |
| :-: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Fleet Identity & Switch State** |
| 1 | `Product_variant` | Categorical | Categorical ID | [A, B, C, D, E] | 5 fleets (1,000 ea) | Identifier for participating utility fleet / breaker design family. |
| 2 | `Breaker_status` | Categorical | Switch State | [Closed, Open] | 79.7% Closed | Operational contact status at time of logging (Closed = in-service, Open = isolated). |
| 3 | `Heater_status` | Categorical | Binary State | [Off, On] | 69.8% Off | Operating state of internal cabinet condensation prevention heater. |
| 4 | `Last_trip_coil_energized`| Categorical | Circuit ID | [TC1, TC2] | 50.4% TC1, 49.6% TC2 | Indicates whether primary trip coil 1 or redundant trip coil 2 actuated the last operation. |
| **Dielectric & SF6 Gas Condition** |
| 5 | `SF6_density` | Float | kg/m³ | [5.50 – 6.96] | 6.01 ± 0.28 | Density of Sulfur Hexafluoride dielectric insulating gas inside the interrupter chamber. |
| 6 | `SF6_temperature` | Float | °C | [0.01 – 50.00] | 20.29 ± 9.65 | Measured temperature of the SF6 gas chamber. |
| 7 | `SF6_dew_point` | Float | °C | [0.09 – 35.00] | 19.89 ± 7.70 | Dew point moisture temperature; higher dew points indicate moisture ingress. |
| **Operating Environment & Grid Conditions** |
| 8 | `Line_voltage` | Float | Volts (V) | [72,500 – 150,000] | 119,747 ± 14,851 | Substation transmission bus line-to-line operating voltage. |
| 9 | `Ambient_cabinet_temperature`| Float | °C | [32.0 – 122.0] | 68.1 ± 14.4 | Internal control and operating mechanism cabinet temperature. |
| **Spring Charging Motor Subsystem** |
| 10 | `Motor_run_time` | Float | Seconds (s) | [0.50 – 15.00] | 7.07 ± 4.29 | Duration required for the motor to recharge the operating mechanism springs. |
| 11 | `Motor_runs_per_day` | Integer | Cycles/day | [0 – 7] | 0.97 ± 0.98 | Frequency of spring charging motor actuations per day. |
| 12 | `Motor_I2T` | Float | A²·s | [0.10 – 25.00] | 8.90 ± 6.77 | Motor thermal energy stress index ($I^2t$) during charging. |
| 13 | `Accumulated_motor_run_time`| Float | Seconds (s) | [10.0 – 1800.2] | 438.9 ± 304.1 | Cumulative motor runtime across breaker operating history. |
| 14 | `Heater_current` | Float | Amperes (A) | [0.00 – 3.00] | 0.47 ± 0.76 | Electrical current drawn by the cabinet anti-condensation heater. |
| **Operating Mechanism Latches & Travel Timings** |
| 15 | `Open_latch_time` | Float | Seconds (s) | [0.050 – 0.150] | 0.112 ± 0.024 | Time elapsed from trip signal energization until mechanical latch release. |
| 16 | `Close_latch_time` | Float | Seconds (s) | [0.050 – 0.150] | 0.112 ± 0.024 | Time elapsed from close signal energization until close latch release. |
| 17 | `Opening_travel_time` | Float | Seconds (s) | [0.050 – 0.150] | 0.081 ± 0.019 | Contact travel stroke duration during an opening operation. |
| 18 | `Closing_travel_time` | Float | Seconds (s) | [0.050 – 0.150] | 0.081 ± 0.019 | Contact travel stroke duration during a closing operation. |
| 19 | `Clearing_time` | Float | Seconds (s) | [0.020 – 0.100] | 0.046 ± 0.025 | Total fault current interruption duration from trip inception to arc extinction. |
| **Contact Kinematics, Arc Stress & Lifetime Wear** |
| 20 | `Open_average_velocity` | Float | m/s | [1.50 – 5.50] | 2.99 ± 0.85 | Mean mechanical velocity of interrupter contacts during opening. |
| 21 | `Close_average_velocity`| Float | m/s | [1.50 – 5.55] | 3.01 ± 0.86 | Mean mechanical velocity of interrupter contacts during closing. |
| 22 | `Open_operation_arcing_time`| Float | Seconds (s) | [0.010 – 0.080] | 0.028 ± 0.017 | Measured duration of high-energy electrical arcing during contact separation. |
| 23 | `Restrike_occurrence` | Float/Binary | Flag (0/1) | [0.0, 1.0] | 9.6% restrike rate | Arc restrike occurrence flag (failure of dielectric medium to withstand recovery voltage). |
| 24 | `Contact_life` | Float | Points / Rating | [10,000 – 84,575] | 33,801 ± 12,721 | Estimated residual electrical contact endurance life rating points. |
| 25 | `Days_since_last_operation` | Float | Days | [13 – 50] | 29.9 ± 5.5 | Number of idle elapsed days since the breaker last switched. |
| 26 | `Total_operation_count` | Integer | Cycles | [3,894 – 6,105] | 4,400 ± 801 | Cumulative switching operations executed over the lifetime of the breaker. |
| 27 | `Non_fault_operation_count`| Integer | Cycles | [2,776 – 5,764] | 3,737 ± 718 | Normal switching operations (load balancing, bus sectionalizing). |
| 28 | `Fault_operations_count` | Integer | Cycles | [196 – 1,797] | 662 ± 246 | High-current fault switching operations (short circuit clearing events). |

---

## 5. Target Variable, Meaning, and Class Encoding
* **Target Column:** `Maintenance_required`
* **Domain Meaning:** A binary classification label assigned by utility engineers and maintenance triggers indicating whether physical service intervention, overhaul, or contact replacement is required.
* **Class Encoding:**
  * **`0` = No Maintenance Required** (Breaker operates within safe electrical and mechanical tolerances).
    * Count: **2,965 samples** (59.30% of total dataset).
  * **`1` = Maintenance Required** (Breaker requires workshop maintenance, contact reconditioning, or gas replenishment).
    * Count: **2,035 samples** (40.70% of total dataset).
* **Positive Class Definition:** In all evaluation metrics throughout this project, **Class `1` (Maintenance Required)** is formally defined as the positive class.

---

## 6. Missing-Value Audit: Affected Variables and Structural Distribution
* **Total Missing Cells:** Exactly 15,000 missing values out of 145,000 total data cells (10.34% overall missingness rate).
* **Missingness Mechanism:** Missing Not At Random (MNAR). Missingness is **strictly structural by utility variant**, reflecting differences in sensor instrument packages installed across the five participating utilities:

| Feature Name | Missing Count | Proportion | Utility Variants Affected |
| :--- | :---: | :---: | :--- |
| `Open_operation_arcing_time` | 3,000 | 60.0% | Variant A (1,000), Variant B (1,000), Variant C (1,000) |
| `Accumulated_motor_run_time` | 2,000 | 40.0% | Variant A (1,000), Variant C (1,000) |
| `Heater_current` | 2,000 | 40.0% | Variant A (1,000), Variant C (1,000) |
| `Open_average_velocity` | 2,000 | 40.0% | Variant A (1,000), Variant C (1,000) |
| `Close_average_velocity` | 2,000 | 40.0% | Variant A (1,000), Variant C (1,000) |
| `Restrike_occurrence` | 2,000 | 40.0% | Variant A (1,000), Variant B (1,000) |
| `Days_since_last_operation` | 1,000 | 20.0% | Variant C (1,000) |
| `Clearing_time` | 1,000 | 20.0% | Variant C (1,000) |

* **Variants D and E:** Variants D (1,000 records) and E (1,000 records) contain zero missing values across all 28 features (100% sensor telemetry coverage).

---

## 7. How Missing Values Were Handled in the Pipeline
To ensure strict avoidance of data leakage:
1. **Fit Preprocessing on Training Data Only:**
   * The dataset was split into **Train (70%, 3,500 samples)**, **Validation (15%, 750 samples)**, and **Holdout Test (15%, 750 samples)** before any imputation or scaling.
   * `SimpleImputer(strategy="median")` was fitted strictly on `X_train` numerical features.
   * `SimpleImputer(strategy="most_frequent")` was fitted strictly on `X_train` categorical features.
2. **Applied via Scikit-Learn ColumnTransformer:**
   * Numerical columns: `SimpleImputer(median)` followed by `StandardScaler()`.
   * Categorical columns: `SimpleImputer(most_frequent)` followed by `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
   * The validation and holdout test sets were transformed using the frozen parameters calculated exclusively on `X_train`.
   * The complete fitted transformer was serialized to `models/preprocessing_pipeline.joblib`.

---

## 8. Excluded Rows and Variables (Zero Exclusion Policy)
* **Excluded Rows:** **0 rows were excluded**. All 5,000 original observations were retained to preserve the natural heterogeneous distribution of utility breaker records.
* **Excluded Variables:** **0 original features were excluded**. All 28 sensor-monitoring features were preserved and utilized in the modeling and inference pipeline.
