# CircuitGuard AI: AI-Based Predictive Maintenance of High-Voltage Circuit Breakers Using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyTorch TabNet](https://img.shields.io/badge/Model-PyTorch%20TabNet-red.svg)](https://github.com/dreamquark-ai/tabnet)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io)

---

## 1. Project Overview & Problem Statement

High-voltage (HV) circuit breakers (typically rated 72.5 kV to 150 kV+) are critical protection devices installed across utility substations. Their core mission is to interrupt fault currents within milliseconds (20 ms to 100 ms) and protect multi-million-dollar power transformers and transmission grids from cataclysmic damage.

Traditional substation maintenance strategies rely heavily on:
1. **Time-based calendar intervals**: Breakers are disassembled and overhauled every 3 to 5 years regardless of actual asset health, incurring high operational expenditure.
2. **Post-failure emergency replacements**: Catastrophic breaker failure (e.g., contact welding, SF6 gas degradation, spring charging motor failure) causes extensive power outages, fire hazards, and environmental hazards.

**CircuitGuard AI** replaces reactive approaches with an intelligent **condition-based predictive maintenance (PdM)** platform. By analyzing 28 multi-modal sensor telemetry features—including SF6 gas density and moisture, mechanism latch timings, contact velocity, and motor charging thermal stress ($I^2t$)—the system predicts whether maintenance is required before an in-service failure occurs.

---

## 2. Dataset Acquisition, Source, and Verification

- **Public Repository**: [sghosh27/Federated-Learning-for-Critical-Electrical-Infrastructure-Maintenance](https://github.com/sghosh27/Federated-Learning-for-Critical-Electrical-Infrastructure-Maintenance)
- **Paper Citation**: *Ghosh S and Mittal G (2026) Federated learning for critical electrical infrastructure - handling data heterogeneity for predictive maintenance of substation equipment. Front. Artif. Intell 8:1697175*
- **License**: Apache License 2.0 (preserved in `dataset/raw/LICENSE`)
- **Raw File**: `dataset/raw/HV Circuit Breaker Maintenance Data.xlsx` (1.37 MB)
- **Total Records**: Exactly **5,000 samples**
- **Feature Space**: **28 condition features** + **1 target column** (`Maintenance_required`)
- **Target Distribution**:
  - `0 (No Maintenance Required)`: 2,965 samples (59.3%)
  - `1 (Maintenance Required)`: 2,035 samples (40.7%)

### Data Provenance & Structure
- The dataset represents condition monitoring records aggregated across **5 representative utility fleets** (coded as `Product_variant` A through E, each with 1,000 samples).
- **Semi-Synthetic Nature**: As documented by the original authors, the dataset is semi-synthetically derived based on real utility substation monitoring standards, augmented with maintenance failure examples to train and evaluate tabular and federated ML models.
- **Structural Missingness (MNAR)**: The 8 features with missing values reflect sensor suite variations between utility fleets:
  - Fleets `D` and `E` have complete telemetry across all 28 features (0 missing values).
  - Fleets `A` and `C` lack contact velocity and motor run telemetry.
  - Fleet `C` also lacks idle days and clearing time.
  - No missing values occur randomly within a variant.

---

## 3. Telemetry Features & Physical Units

| # | Feature Name | Type | Physical Unit | Description | Range [Min - Max] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Product_variant` | Categorical | Utility Fleet | Asset utility fleet identifier | [A, B, C, D, E] |
| 2 | `Breaker_status` | Categorical | State | Primary switch state | [Closed, Open] |
| 3 | `SF6_density` | Numerical | kg/m³ | Dielectric insulating gas density | 5.50 - 6.96 kg/m³ |
| 4 | `SF6_temperature` | Numerical | °C | SF6 chamber gas temperature | 0.01 - 50.00 °C |
| 5 | `SF6_dew_point` | Numerical | °C | Moisture condensation dew point | 0.09 - 35.00 °C |
| 6 | `Motor_run_time` | Numerical | s | Spring charging motor run duration | 0.50 - 15.00 s |
| 7 | `Motor_runs_per_day` | Numerical | runs/day | Daily motor recharge operations | 0 - 7 cycles/day |
| 8 | `Motor_I2T` | Numerical | A²·s | Motor charging thermal dissipation stress | 0.10 - 25.00 A²·s |
| 9 | `Heater_status` | Categorical | State | Anti-condensation cabinet heater | [Off, On] |
| 10 | `Line_voltage` | Numerical | V | Operating transmission line voltage | 72,500 - 150,000 V |
| 11 | `Ambient_cabinet_temperature`| Numerical| °C | Substation mechanism cabinet temp | 32.0 - 122.0 °C |
| 12 | `Days_since_last_operation` | Numerical | days | Elapsed days since prior switching | 13 - 50 days |
| 13 | `Contact_life` | Numerical | points | Remaining electrical contact wear rating | 10,000 - 84,575 |
| 14 | `Last_trip_coil_energized` | Categorical | Circuit | Dual tripping circuit designation | [TC1, TC2] |
| 15 | `Clearing_time` | Numerical | s | Fault current interruption duration | 0.020 - 0.100 s |
| 16 | `Open_latch_time` | Numerical | s | Mechanical latch release time (opening)| 0.050 - 0.150 s |
| 17 | `Close_latch_time` | Numerical | s | Mechanical latch release time (closing)| 0.050 - 0.150 s |
| 18 | `Opening_travel_time` | Numerical | s | Contact travel time during opening | 0.050 - 0.150 s |
| 19 | `Closing_travel_time` | Numerical | s | Contact travel time during closing | 0.050 - 0.150 s |
| 20 | `Total_operation_count` | Numerical | cycles | Cumulative switching operations | 3,894 - 6,105 cycles |
| 21 | `Non_fault_operation_count` | Numerical | cycles | Routine load switching operations | 2,776 - 5,764 cycles |
| 22 | `Fault_operations_count` | Numerical | cycles | Short-circuit fault clearance count | 196 - 1,797 cycles |
| 23 | `Accumulated_motor_run_time`| Numerical| s | Lifetime cumulative motor run duration | 10.0 - 1,800.2 s |
| 24 | `Heater_current` | Numerical | A | Electrical current drawn by heater | 0.00 - 3.00 A |
| 25 | `Open_average_velocity` | Numerical | m/s | Contact separation speed | 1.50 - 5.50 m/s |
| 26 | `Close_average_velocity` | Numerical | m/s | Contact mating speed | 1.50 - 5.55 m/s |
| 27 | `Restrike_occurrence` | Numerical | flag | High-frequency arc restrike flag | 0.0 (No) or 1.0 (Yes) |
| 28 | `Open_operation_arcing_time`| Numerical | s | Plasma arc duration during contact open | 0.010 - 0.080 s |

---

## 4. Preprocessing & Data Splitting

- **Stratified Split**:
  - Training Set: **3,500 samples (70.0%)**
  - Validation Set: **750 samples (15.0%)**
  - Untouched Test Set: **750 samples (15.0%)**
  - Splitting uses composite stratification on `(Product_variant, Maintenance_required)` to ensure exact representation of all 5 utility fleets and both target classes across splits.
- **Data Leakage Prevention**:
  - Imputation statistics (median for numerical features, mode for categorical features) and standard scaling are fitted **strictly on the training data**.
  - Categorical variables (`Product_variant`, `Breaker_status`, `Heater_status`, `Last_trip_coil_energized`) are one-hot encoded into 11 binary dimensions, resulting in a **35-dimensional preprocessed input vector**.
  - Pipeline is saved to `models/preprocessing_pipeline.joblib`.

---

## 5. Machine Learning Architecture & Transfer Learning

### Approach: PyTorch TabNet Self-Supervised Pretraining & Fine-Tuning
For tabular predictive maintenance telemetry, standard neural networks often struggle with feature sparsity and non-linear interactions. We implement **TabNet** (Arik & Pfister, Google Cloud AI):

```
Unlabeled Telemetry X ──► [ TabNet Pretrainer (Entmax Masking) ] ──► Masked Reconstruction Loss
                                        │
                                        ▼ (Weight Transfer)
Supervised Telemetry (X, y) ──► [ TabNet Classifier ] ──► Supervised Fine-Tuning Loss ──► Maintenance Prediction
```

1. **Unsupervised Pretraining Phase**:
   - Model: `TabNetPretrainer(mask_type='entmax', n_d=16, n_a=16, n_steps=3, gamma=1.3)`.
   - Objective: 50% of the 35 transformed features are masked at each step; the network reconstructs the masked values via self-supervised learning.
   - Benefit: The feature transformers learn physical correlations between SF6 density, mechanical latch speeds, and charging motor wear without using target labels.
2. **Supervised Fine-Tuning Phase**:
   - The pretrained weights are transferred to `TabNetClassifier`.
   - The model is fine-tuned on maintenance labels using Adam optimizer ($lr = 0.02$) with early stopping on validation AUC.
3. **Ablation Comparison**:
   - An identical TabNet model is trained strictly from scratch (without pretraining) to empirically measure the benefit of transfer learning.
4. **Supervised Baselines**:
   - Random Forest (150 trees, max depth 12)
   - XGBoost (150 trees, max depth 5, lr 0.05)
   - LightGBM (150 trees, max depth 5, lr 0.05)
   - Multi-Layer Perceptron (64-32 hidden neurons)
   - Logistic Regression (L2 regularized)

---

## 6. Real Test-Set Evaluation Results

The models were evaluated on the untouched 750-sample test set. **All metrics are genuine and reproducible**:

| Model Architecture | Paradigm | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | Supervised Baseline | **92.00%** | **97.67%** | 82.30% | **89.32%** | **0.9794** | **0.9732** |
| **Gradient Boosting (GBM)** | Supervised Baseline | 91.60% | 97.27% | 81.64% | 88.77% | 0.9794 | 0.9733 |
| **TabNet (Scratch)** | Supervised Scratch | 91.60% | 94.49% | 84.26% | 89.08% | 0.9809 | 0.9747 |
| **XGBoost** | Supervised Baseline | 90.93% | 94.38% | 82.62% | 88.11% | 0.9800 | 0.9737 |
| **TabNet (Pretrained + FT)** | Transfer Learning | **90.27%** | **99.15%** | 76.72% | **86.51%** | **0.9777** | **0.9703** |

### Key Findings
- **High Precision on TabNet**: TabNet (Pretrained + Fine-Tuned) achieved an exceptional **99.15% precision** (only 2 false alarms out of 238 predicted maintenance flags), which is critical for electrical utilities to avoid costly unnecessary truck dispatches.
- **Non-Linear Tree and Attention Superiority**: Tree models and TabNet significantly outperformed linear models (Logistic Regression F1: 72.83%), proving that circuit-breaker degradation involves non-linear thermal-mechanical interactions.

---

## 7. Demonstration Scenarios (Genuine Dataset Records)

The platform provides 5 genuine demonstration profiles extracted directly from actual dataset rows:

1. **Normal Operating Condition** (Row #0, Target: `0`):
   - SF6 density: 6.05 kg/m³, Cabinet Temp: 52.6 °C, Motor Run: 1.87 s, Latch Time: 0.128 s.
   - Model Prediction: **No Maintenance Required** (Confidence: 99.96%).
2. **High Temperature Condition** (Row #16, Target: `0`):
   - Extreme thermal stress: Cabinet Temp 102.4 °C, SF6 Temp 41.2 °C.
   - Demonstrates model robustness under severe ambient environmental heating.
3. **Low Gas Density Condition** (Row #15, Target: `0`):
   - Depleted SF6 density: 5.503 kg/m³ (approaching critical lockout threshold).
4. **High Operation Count Condition** (Row #1002, Target: `0`):
   - Heavy switching asset: 5,618 total switching cycles, 1,024 fault interruptions.
5. **Maintenance Required Example** (Row #1, Target: `1`):
   - Severe mechanical sluggishness: Motor recharge 15.0 s, Latch delay 0.117 s.
   - Model Prediction: **Maintenance Required** (Confidence: 99.95%).

---

## 8. Quickstart Guide (macOS Terminal Commands)

### Step 1: Navigate to Project Directory
```bash
cd "/Users/divyanshu1911/Desktop/Switch Gear Project"
```

### Step 2: Launch the Complete React Application
Simply run:
```bash
bash run.sh
```
This automatically starts the FastAPI backend and serves the **React Dashboard**:
* **Interactive React Dashboard**: [http://localhost:8000](http://localhost:8000)
* **API Documentation & Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)

*(For Vite hot-reloading dev mode: `bash run.sh react` on [http://localhost:3000](http://localhost:3000))*  
*(For the legacy Streamlit interface: `bash run.sh streamlit` on [http://localhost:8501](http://localhost:8501))*

---

## 9. API Reference

- `GET /health`: Health check and loaded model registry.
- `GET /model-info`: Model metadata, architecture, and test set metrics.
- `GET /features`: Input feature schema, valid ranges, units, and categories.
- `GET /scenarios`: 5 genuine demonstration profiles.
- `POST /predict`: Single circuit breaker predictive assessment with confidence score and local explainability.
- `POST /predict-batch`: High-throughput CSV batch evaluation.

### Example Single Prediction Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Product_variant": "A",
      "Breaker_status": "Closed",
      "SF6_density": 6.05,
      "SF6_temperature": 20.0,
      "SF6_dew_point": 19.5,
      "Motor_run_time": 6.5,
      "Motor_runs_per_day": 1,
      "Motor_I2T": 8.5,
      "Heater_status": "Off",
      "Line_voltage": 120000.0,
      "Ambient_cabinet_temperature": 65.0,
      "Days_since_last_operation": 30.0,
      "Contact_life": 33000.0,
      "Last_trip_coil_energized": "TC1",
      "Clearing_time": 0.045,
      "Open_latch_time": 0.11,
      "Close_latch_time": 0.11,
      "Opening_travel_time": 0.08,
      "Closing_travel_time": 0.08,
      "Total_operation_count": 4200,
      "Non_fault_operation_count": 3600,
      "Fault_operations_count": 600,
      "Accumulated_motor_run_time": 400.0,
      "Heater_current": 0.5,
      "Open_average_velocity": 3.0,
      "Close_average_velocity": 3.0,
      "Restrike_occurrence": 0.0,
      "Open_operation_arcing_time": 0.025
    },
    "model_name": "TabNet (Pretrained + Fine-Tuned)"
  }'
```

---

## 10. Disclaimer
*This machine learning prediction system is intended for educational, technical demonstration, and decision-support purposes. It is not a replacement for certified utility circuit-breaker maintenance procedures, manual timing test sets, or safety standards.*

