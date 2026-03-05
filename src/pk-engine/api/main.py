"""
MIPD PK Engine – FastAPI Application Entry Point.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

API documentation auto-generated at:
    - Swagger UI: http://localhost:8000/docs
    - ReDoc:      http://localhost:8000/redoc
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes_pk import router as pk_router
from api.routes_bayesian import router as bayesian_router
from api.routes_dosing import router as dosing_router
from api.routes_ai import router as ai_router

app = FastAPI(
    title="MIPD PK Engine",
    description=(
        "**Model-Informed Precision Dosing** API for Pharmacokinetic "
        "parameter estimation and dose optimization.\n\n"
        "Features:\n"
        "- 🧪 PK concentration prediction (2-compartment IV)\n"
        "- 📊 Bayesian inference (MAP, Laplace, ADVI, EP, SMC)\n"
        "- 💊 Dose optimization (AUC/MIC target)\n"
        "- 🤖 AI/ML (anomaly detection, covariate screening)\n"
        "- ✅ Validation metrics (MPE, MAPE, CCC, NPDE)\n"
        "- 🛡️ 5-layer clinical safety guardrails"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(pk_router)
app.include_router(bayesian_router)
app.include_router(dosing_router)
app.include_router(ai_router)


@app.get("/", tags=["Health"])
def health_check():
    """API health check endpoint."""
    return {
        "status": "healthy",
        "service": "MIPD PK Engine",
        "version": "1.0.0",
        "endpoints": {
            "pk": "/pk/predict, /pk/clinical",
            "bayesian": "/bayesian/estimate (MAP/Laplace/ADVI/EP/SMC)",
            "dosing": "/dosing/recommend",
            "ai": "/ai/anomaly-check, /ai/screen-covariates, /ai/validate-metrics",
            "docs": "/docs, /redoc",
        },
    }


@app.get("/safety-info", tags=["Safety"])
def safety_info():
    """Describe the 5-layer safety guardrail system."""
    return {
        "layers": [
            {
                "layer": 1,
                "name": "Input Validation",
                "description": (
                    "Patient demographics, dose amounts, and TDM "
                    "concentrations checked against physiological ranges"
                ),
            },
            {
                "layer": 2,
                "name": "PK Parameter Plausibility",
                "description": (
                    "Estimated CL, V parameters validated against "
                    "pharmacologically feasible ranges"
                ),
            },
            {
                "layer": 3,
                "name": "Dose Recommendation Limits",
                "description": (
                    "Hard caps on recommended doses. Never exceeds "
                    "maximum safe dose. Flags >50% dose changes."
                ),
            },
            {
                "layer": 4,
                "name": "Confidence Requirement",
                "description": (
                    "Rejects predictions with too-wide 95% credible "
                    "intervals (CI ratio > 3.0)"
                ),
            },
            {
                "layer": 5,
                "name": "Clinical Decision Support",
                "description": (
                    "Consolidated risk score (0-1), alert levels "
                    "(info/warning/critical/reject), and review triggers"
                ),
            },
        ],
        "risk_score_interpretation": {
            "0.00": "All checks passed",
            "0.25": "Informational alerts only",
            "0.50": "Warnings present – review recommended",
            "0.75": "Critical – pharmacist review REQUIRED",
            "1.00": "REJECTED – do not use recommendation",
        },
    }
