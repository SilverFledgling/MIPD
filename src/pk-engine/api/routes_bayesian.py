"""
Bayesian Inference API Routes – MAP, Laplace, ADVI, EP, SMC estimation.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import (
    BayesianRequest, BayesianResponse,
    PKParamsSchema, SafetyReportSchema, SafetyAlertSchema,
)
from api.safety import (
    validate_patient, validate_doses, validate_observations,
    validate_pk_params, validate_confidence, generate_safety_report,
)
from api.routes_pk import _to_patient_data, _to_dose_events
from pk.models import Observation
from pk.population import get_model, compute_vancomycin_tv
from bayesian.map_estimator import estimate_map
from bayesian.laplace import laplace_approximation


router = APIRouter(prefix="/bayesian", tags=["Bayesian Inference"])


@router.post(
    "/estimate",
    response_model=BayesianResponse,
    summary="Estimate individual PK parameters using Bayesian methods",
)
def bayesian_estimate(req: BayesianRequest) -> BayesianResponse:
    """
    Bayesian estimation of individual PK parameters.

    Supports MAP, Laplace, ADVI, EP, and SMC methods.
    All responses include safety validation.
    """
    # Safety validation
    alerts = []
    patient = _to_patient_data(req.patient)
    alerts.extend(validate_patient(patient))
    dose_events = _to_dose_events(req.doses)
    alerts.extend(validate_doses(dose_events))
    alerts.extend(validate_observations(
        [{"concentration": o.concentration, "time": o.time} for o in req.observations]
    ))

    # Get model and typical values
    model = get_model(req.drug)
    try:
        tv = compute_vancomycin_tv(patient)
    except (ValueError, ZeroDivisionError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot compute PK parameters: {e}. "
                   f"Check patient data (weight, age, SCr must be positive).",
        )

    # Convert observations
    observations = [
        Observation(
            time=o.time,
            concentration=o.concentration,
            sample_type=o.sample_type,
        )
        for o in req.observations
    ]

    method = req.method.value
    confidence = None
    diagnostics = {}

    try:
        if method == "map":
            result = estimate_map(model, tv, dose_events, observations)
            ind_params = result.params
            eta = result.eta_map.tolist()
            diagnostics = {
                "objective": result.objective,
                "converged": result.success,
                "n_iterations": result.n_iterations,
            }

        elif method == "laplace":
            result = laplace_approximation(
                model, tv, dose_events, observations,
            )
            ind_params = result.params
            eta = result.eta_map.tolist()
            import numpy as np
            ci_lower = result.eta_map - 1.96 * np.sqrt(np.diag(result.covariance))
            ci_upper = result.eta_map + 1.96 * np.sqrt(np.diag(result.covariance))
            confidence = {
                "CL": {
                    "ci95_lower": float(tv.CL * np.exp(ci_lower[0])),
                    "ci95_upper": float(tv.CL * np.exp(ci_upper[0])),
                },
                "V1": {
                    "ci95_lower": float(tv.V1 * np.exp(ci_lower[1])),
                    "ci95_upper": float(tv.V1 * np.exp(ci_upper[1])),
                },
            }
            diagnostics = {
                "objective": result.objective,
                "converged": result.success,
                "log_det_H": result.log_det_H,
            }

            # Safety Layer 4: confidence check
            for param_name, ci in confidence.items():
                alerts.extend(validate_confidence(
                    ci["ci95_lower"], ci["ci95_upper"], param_name,
                ))

        elif method == "advi":
            from bayesian.advi import run_advi
            result = run_advi(model, tv, dose_events, observations)
            ind_params = result.params
            eta = result.mu.tolist()
            diagnostics = {
                "elbo": result.elbo,
                "converged": result.converged,
                "sigma": result.sigma.tolist(),
            }

        elif method == "ep":
            from bayesian.ep import run_ep
            result = run_ep(model, tv, dose_events, observations)
            ind_params = result.params
            eta = result.mu.tolist()
            diagnostics = {
                "n_iterations": result.n_iterations,
                "converged": result.converged,
            }

        elif method == "smc":
            from bayesian.smc import run_smc
            result = run_smc(
                model, tv, dose_events, observations,
                n_particles=200,
            )
            ind_params = result.params
            import numpy as np
            eta = np.average(
                result.particles, weights=result.weights, axis=0,
            ).tolist()
            diagnostics = {
                "n_resamples": result.n_resamples,
                "ess_history": result.ess_history,
                "n_particles": result.particles.shape[0],
            }

        elif method == "mcmc":
            from bayesian.mcmc import run_mcmc
            result = run_mcmc(
                model, tv, dose_events, observations,
                n_warmup=500, n_samples=1000, n_chains=2,
            )
            ind_params = result.map_params
            eta = result.posterior_eta.mean(axis=0).tolist()
            confidence = {}
            for pk_name, info in result.posterior_params.items():
                confidence[pk_name] = {
                    "ci95_lower": info["ci95_lower"],
                    "ci95_upper": info["ci95_upper"],
                }
            diagnostics = {
                "rhat": result.rhat,
                "ess": result.ess,
                "n_divergences": result.n_divergences,
                "n_samples": result.n_samples,
                "converged": result.converged,
            }
            # Safety Layer 4: confidence check
            for param_name, ci in confidence.items():
                alerts.extend(validate_confidence(
                    ci["ci95_lower"], ci["ci95_upper"], param_name,
                ))

        elif method == "adaptive":
            from bayesian.engine import adaptive_pipeline
            pipeline_result = adaptive_pipeline(
                model, tv, dose_events, observations,
                run_layer2=True,
                run_layer3=False,
            )
            ind_params = pipeline_result.final_params
            eta = pipeline_result.final_eta
            if pipeline_result.final_confidence:
                confidence = pipeline_result.final_confidence
                for param_name, ci in confidence.items():
                    alerts.extend(validate_confidence(
                        ci["ci95_lower"], ci["ci95_upper"], param_name,
                    ))
            diagnostics = {
                "layers_executed": pipeline_result.layers_executed,
                "pipeline": pipeline_result.diagnostics,
            }

        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown method: {method}",
            )

    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Safety Layer 2: validate estimated params
    alerts.extend(validate_pk_params(ind_params))

    safety = generate_safety_report(alerts)

    return BayesianResponse(
        method=method,
        individual_params=PKParamsSchema(
            CL=round(ind_params.CL, 4),
            V1=round(ind_params.V1, 4),
            Q=round(ind_params.Q, 4),
            V2=round(ind_params.V2, 4),
        ),
        eta=[round(e, 6) for e in eta],
        confidence=confidence,
        diagnostics=diagnostics,
        safety=SafetyReportSchema(**{
            "is_safe": safety.is_safe,
            "risk_score": safety.risk_score,
            "alerts": [SafetyAlertSchema(**a.__dict__) for a in safety.alerts],
            "requires_review": safety.requires_review,
            "recommendation": safety.recommendation,
        }),
    )
