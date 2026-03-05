# MIPD – Model-Informed Precision Dosing

**Hệ thống tối ưu liều lượng thuốc dựa trên Pharmacokinetics/Pharmacodynamics (PK/PD) cho bệnh viện Việt Nam.**

## 🎯 Mục tiêu

Xây dựng hệ thống MIPD (Model-Informed Precision Dosing) hỗ trợ dược sĩ lâm sàng tối ưu liều vancomycin dựa trên:
- Mô hình PK dân số (Population PK)
- Bayesian estimation cá thể hóa tham số
- Thuật toán tối ưu liều AUC₂₄/MIC-guided
- Hệ thống an toàn 5 lớp bảo vệ bệnh nhân

## 🏗️ Kiến trúc

```
MIPD/src/pk-engine/
├── pk/                # Core PK models, ODE solver, clinical calculations
│   ├── models.py      # Data models (PatientData, DoseEvent, PKParams)
│   ├── analytical.py  # Analytical PK solutions
│   ├── solver.py      # ODE solver (RK45)
│   ├── population.py  # Population PK model (Vancomycin VN)
│   └── clinical.py    # CrCL, eGFR, IBW, ABW, dosing weight
│
├── bayesian/          # Bayesian inference methods
│   ├── map_estimator.py  # MAP (Maximum A Posteriori)
│   ├── laplace.py        # Laplace approximation
│   ├── advi.py           # ADVI (Automatic Differentiation VI)
│   ├── ep.py             # Expectation Propagation
│   ├── smc.py            # Sequential Monte Carlo
│   └── mcmc.py           # MCMC/NUTS (NumPyro, requires Linux/WSL)
│
├── dosing/            # Dose optimization
│   └── optimizer.py   # Grid search + Monte Carlo PTA
│
├── ai/                # AI/ML features
│   ├── anomaly_detection.py  # Swift Hydra 4-head TDM quality check
│   ├── ml_screening.py       # RF+NN+SVR covariate screening
│   └── gp_bnn.py             # Gaussian Process + BNN
│
├── validation/        # Validation metrics
│   └── metrics.py     # MPE, MAPE, RMSE, CCC, NPDE, TOST
│
├── api/               # FastAPI REST API
│   ├── main.py        # App entry point
│   ├── safety.py      # 5-layer safety guardrails
│   ├── schemas.py     # Pydantic request/response models
│   ├── routes_pk.py   # /pk/predict, /pk/clinical
│   ├── routes_bayesian.py  # /bayesian/estimate
│   ├── routes_dosing.py    # /dosing/recommend
│   └── routes_ai.py        # /ai/anomaly-check, /ai/validate-metrics
│
├── tests/             # Unit + integration tests
│   ├── test_pk_engine.py
│   ├── test_bayesian.py
│   ├── test_dosing_advanced.py
│   ├── test_validation_ai.py
│   └── test_api_safety.py
│
└── requirements.txt   # Python dependencies
```

## 🚀 Cài đặt & Chạy

### 1. Clone repo
```bash
git clone https://github.com/<your-org>/MIPD.git
cd MIPD/src/pk-engine
```

### 2. Cài đặt dependencies
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 3. Chạy tests
```bash
python -m pytest tests/ -v
```

### 4. Chạy API server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Mở `http://localhost:8000/docs` để xem Swagger UI.

## 🛡️ Hệ thống An toàn (5 Lớp)

| Lớp | Bảo vệ | Ví dụ |
|-----|--------|-------|
| 1. Input Validation | Chặn dữ liệu vô lý | Weight < 0, Age > 120 |
| 2. PK Plausibility | PK params bất thường | CL = 0.01 L/h |
| 3. Dose Limits | Hard cap liều tối đa | Max 3000 mg |
| 4. Confidence Check | Reject CI quá rộng | CI ratio > 3.0 |
| 5. Risk Score | Tổng hợp cảnh báo | 0.0 → 1.0 |

## 📡 API Endpoints

| Method | Endpoint | Chức năng |
|--------|----------|-----------|
| GET | `/` | Health check |
| GET | `/safety-info` | Safety system info |
| POST | `/pk/predict` | Dự đoán nồng độ + AUC |
| POST | `/pk/clinical` | CrCL, eGFR, IBW, ABW |
| POST | `/bayesian/estimate` | MAP/Laplace/ADVI/EP/SMC |
| POST | `/dosing/recommend` | Khuyến nghị liều + PTA |
| POST | `/ai/anomaly-check` | Kiểm tra TDM quality |
| POST | `/ai/screen-covariates` | Sàng lọc covariate |
| POST | `/ai/validate-metrics` | MPE, MAPE, RMSE, CCC |

## 🗄️ Database

**Hiện tại**: PK Engine là stateless API (không có database). Tất cả dữ liệu được truyền qua API request.

**Tương lai (Phase 8+)**: Sẽ cần database để lưu:
- Hồ sơ bệnh nhân (patient records)
- Lịch sử TDM (TDM history)
- Kết quả ước lượng PK (PK estimation results)
- Khuyến nghị liều (dose recommendations)
- Audit trail (lịch sử thao tác)

**Đề xuất**: PostgreSQL + SQLAlchemy ORM

## 📋 Requirements

- Python 3.11+
- FastAPI, Pydantic, uvicorn
- NumPy, SciPy
- JAX, NumPyro (cho MCMC, cần Linux/WSL)
- scikit-learn (cho AI/ML features)

## 👥 Team

- **Backend PK Engine**: Python (FastAPI + NumPy + SciPy)
- **Frontend** (planned): React/Next.js
- **Deployment** (planned): Docker + Cloud

## 📄 License

[Cần xác định]
