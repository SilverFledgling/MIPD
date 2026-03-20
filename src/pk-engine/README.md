# PK Engine — Bayesian MIPD cho Vancomycin

Công cụ định liều chính xác (Model-Informed Precision Dosing) dựa trên dược động học quần thể và thuật toán Bayesian thích nghi cho bệnh nhân Việt Nam.

## Kiến trúc

```
pk-engine/
├── pk/                     # PK model core
│   ├── models.py           # PKParams, PopPKModel, DoseEvent, Observation
│   ├── solver.py           # ODE/Analytical PK solver
│   ├── population.py       # VANCOMYCIN_VN params (Goti 2018), apply_iiv
│   └── analytical.py       # 2-compartment analytical superposition
├── bayesian/               # Bayesian estimation algorithms
│   ├── engine.py           # Unified engine + adaptive_pipeline()
│   ├── map_estimator.py    # MAP (Maximum A Posteriori)
│   ├── laplace.py          # Laplace Approximation (MAP + Hessian → CI)
│   ├── smc.py              # Sequential Monte Carlo (Particle Filter)
│   ├── ep.py               # Expectation Propagation
│   ├── mcmc_mh.py          # MCMC Metropolis-Hastings (analytical solver)
│   ├── mcmc.py             # MCMC-NUTS (JAX + NumPyro)
│   ├── advi.py             # Automatic Differentiation VI (NumPyro)
│   ├── bma.py              # Bayesian Model Averaging
│   ├── hierarchical.py     # Hierarchical Bayesian (3-tier)
│   └── population_store.py # Vietnam Population Store (SAEM update)
└── benchmarks/             # Benchmark & validation
    ├── run_benchmark.py          # Single-process benchmark
    ├── run_benchmark_parallel.py # Multi-process parallel benchmark
    ├── merge_results.py          # Merge CSV results from parallel runs
    ├── benchmark_plots.py        # Bland-Altman, VPC, NPDE plots
    └── plots/                    # Generated validation plots
```

## Cài đặt

```bash
cd src/pk-engine
pip install -r requirements.txt
```

Yêu cầu: Python 3.11+, NumPy, SciPy, JAX, NumPyro, matplotlib

## Chạy Benchmark

### Cách 1: Chạy đơn (1 CMD)

```bash
cd src/pk-engine
python benchmarks/run_benchmark.py --n 50 --methods map,laplace,smc,ep,mcmc_mh,mcmc_nuts,advi,adaptive_ind,adaptive_cum
```

### Cách 2: Chạy song song (2 CMD, khuyến nghị)

**CMD 1** — Các method nhanh:
```bash
cd src/pk-engine
python benchmarks/run_benchmark_parallel.py --methods map,laplace,smc,ep,mcmc_mh --n 50 --workers 3
```

**CMD 2** — Các method nặng + adaptive:
```bash
cd src/pk-engine
python benchmarks/run_benchmark_parallel.py --methods mcmc_nuts,advi,adaptive_ind,adaptive_cum --n 50 --workers 3
```

### Merge kết quả

Sau khi cả 2 CMD chạy xong, merge CSV:

```bash
cd src/pk-engine
python benchmarks/merge_results.py "benchmarks/results_CMD1_*.csv" "benchmarks/results_CMD2_*.csv"
```

Kết quả:
- `benchmark_results_merged.csv` — Bảng Core + Clinical metrics đầy đủ
- `benchmarks/plots/` — 27 plots (3 loại × 9 methods)

### Tải lại bệnh nhân đã lưu (reproducible)

```bash
# Lần đầu: tự động lưu patients_50.pkl
python benchmarks/run_benchmark_parallel.py --methods map --n 50

# Lần sau: load lại cùng bộ BN
python benchmarks/run_benchmark_parallel.py --methods smc --n 50 --load-patients patients_50.pkl
```

## Các Methods

| Method | Tốc độ | CCC | Mô tả |
|:--|:--|:--|:--|
| `map` | 4.9s/BN | 0.938 | Maximum A Posteriori — tiêu chuẩn FDA |
| `laplace` | 8.5s/BN | 0.938 | MAP + Hessian → CI 95% |
| `smc` | 7.4s/BN | 0.967 | Sequential Monte Carlo (500 particles) |
| `ep` | 3.5s/BN | 0.937 | Expectation Propagation |
| `mcmc_mh` | **0.2s/BN** | 0.967 | Metropolis-Hastings (analytical solver) |
| `mcmc_nuts` | 6.0s/BN | 0.966 | NUTS — gold standard MCMC (JAX) |
| `advi` | 34.4s/BN | 0.638 | Variational Inference (đang cải thiện) |
| `adaptive_ind` | 17.7s/BN | **0.972** | Pipeline MAP→SMC (reset mỗi BN) |
| `adaptive_cum` | 24.7s/BN | **0.972** | Pipeline MAP→SMC + Population Store (tích lũy) |

## Metrics đầu ra

### Core Metrics
- **MPE** — Mean Prediction Error (bias)
- **MAPE** — Mean Absolute Prediction Error (precision)
- **RMSE** — Root Mean Squared Error
- **CCC** — Concordance Correlation Coefficient

### Clinical Metrics
- **BA Bias / LoA** — Bland-Altman analysis
- **Cov95%** — Coverage xác suất 95%
- **TA%** — Target Attainment (AUC 400–600)
- **Shrink%** — η-Shrinkage
- **TOST** — Two One-Sided Tests tương đương

### Plots
- `bland_altman_{method}.png` — Bias ± Limits of Agreement
- `vpc_{method}.png` — Visual Predictive Check
- `npde_{method}.png` — Normalized Prediction Distribution Errors

## Tham khảo chính

- Goti et al. (2018), *Clin Pharmacokinet* 57(6):735–748
- Rybak et al. (2020), *AJHP* 77(11):835–864 (IDSA/ASHP Guidelines)
- Hien et al. (2021), *Kidney Int Rep* — CKD prevalence VN
- Broeker et al. (2019), *CPT:PSP* 8(6) — Validation framework
