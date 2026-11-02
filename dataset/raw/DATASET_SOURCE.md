# Dataset Source Documentation: CircuitGuard AI

## Origin and Repository
- **Repository**: [sghosh27/Federated-Learning-for-Critical-Electrical-Infrastructure-Maintenance](https://github.com/sghosh27/Federated-Learning-for-Critical-Electrical-Infrastructure-Maintenance)
- **Paper Citation**: Ghosh S and Mittal G (2026) Federated learning for critical electrical infrastructure - handling data heterogeneity for predictive maintenance of substation equipment. Front. Artif. Intell 8:1697175.
- **License**: Apache License 2.0 (see `LICENSE` in this directory)
- **File Name**: `HV Circuit Breaker Maintenance Data.xlsx`
- **File Size**: ~1.37 MB (1,404,324 bytes)

## Dataset Description
- **Domain**: High-Voltage (HV) Circuit Breaker condition monitoring and predictive maintenance at electrical utility substations.
- **Total Records**: 5,000 samples.
- **Total Columns**: 29 columns (28 condition/sensor monitoring features + 1 binary maintenance target).
- **Source Aggregation**: Derived from 5 representative utility fleets, labeled as `Product_variant` A through E (1,000 samples each).
- **Target Label**: `Maintenance_required` (0 = No Maintenance Required [2,965 samples, 59.3%], 1 = Maintenance Required [2,035 samples, 40.7%]).
- **Nature of Data**: Semi-synthetically derived for research and educational purposes based on utility substation monitoring guidelines, standard maintenance triggers (such as SF6 density/dew point thresholds, trip coil energization times, contact wear, operation counts), and non-IID data heterogeneity across utility operating zones.

## Download Instructions (Manual / Reproducible)
To acquire the raw dataset directly:
```bash
git clone https://github.com/sghosh27/Federated-Learning-for-Critical-Electrical-Infrastructure-Maintenance.git /tmp/source_repo
cp "/tmp/source_repo/Datasets/HV Circuit Breaker Maintenance Data.xlsx" dataset/raw/
rm -rf /tmp/source_repo
```

