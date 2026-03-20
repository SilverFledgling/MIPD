"""
Population PK Model Definitions – Drug-specific PopPK models.

Currently includes:
    - Vancomycin 2-compartment IV (Vietnamese population, placeholder)
    - Tacrolimus 1-compartment Oral (placeholder for future extension)

Each model defines:
    - Typical values (TV)
    - Omega matrix (IIV)
    - Error model (residual)
    - Covariate relationships

Reference:
    - Rybak et al. (2020), AJHP, 77(11), 835-864
    - Antignac et al. (2007), Clin Pharmacokinet, 46(8)
"""

import math

import numpy as np
from numpy.typing import NDArray

from pk.models import (
    ErrorModel,
    ErrorModelType,
    Gender,
    ModelType,
    PatientData,
    PKParams,
    PopPKModel,
)
from pk.clinical import cockcroft_gault_crcl, compute_weight_for_dosing


# ──────────────────────────────────────────────────────────────────
# Vancomycin 2-comp IV (Vietnamese Population)
# ──────────────────────────────────────────────────────────────────

VANCOMYCIN_VN = PopPKModel(
    name="Vancomycin Vietnamese 2-comp IV",
    drug="vancomycin",
    model_type=ModelType.TWO_COMP_IV,
    typical_values=PKParams(
        CL=2.99,    # L/h  — Goti et al. (2018), Clin Pharmacokinet, 57(6):735-748
        V1=30.4,    # L    — Goti 2018 (central volume)
        Q=5.09,     # L/h  — Goti 2018 (inter-compartmental clearance)
        V2=38.4,    # L    — Goti 2018 (peripheral volume)
    ),
    omega_matrix=[
        # Goti 2018: ω²_CL=0.134, ω²_V1=0.0946 (estimated, RSE 3.6%, 6.5%)
        # Q, V2 omega from Thomson et al. (2009) meta-analysis
        # Off-diagonal: covariance(CL,V1)=0.02 (Goti 2018)
        [0.134,  0.020,  0.000,  0.000],   # ω²_CL = 0.134 (CV≈37%)
        [0.020,  0.0946, 0.000,  0.000],   # ω²_V1 = 0.0946 (CV≈31%)
        [0.000,  0.000,  0.200,  0.000],   # ω²_Q  = 0.200 (Thomson 2009)
        [0.000,  0.000,  0.000,  0.150],   # ω²_V2 = 0.150 (Thomson 2009)
    ],
    error_model=ErrorModel(
        sigma_prop=0.138,    # 13.8% proportional — Goti 2018
        sigma_add=1.62,      # 1.62 mg/L additive — Goti 2018
        model_type=ErrorModelType.COMBINED,
    ),
    reference=(
        "Goti et al. (2018), Clin Pharmacokinet, 57(6):735-748. "
        "N≈500, externally validated. "
        "Omega Q/V2: Thomson et al. (2009), J Clin Pharmacol, 49(10):1195."
    ),
)


# ──────────────────────────────────────────────────────────────────
# Tacrolimus 1-comp Oral (Placeholder for future extension)
# ──────────────────────────────────────────────────────────────────

TACROLIMUS_ORAL = PopPKModel(
    name="Tacrolimus 2-comp Oral",
    drug="tacrolimus",
    model_type=ModelType.TWO_COMP_ORAL,
    typical_values=PKParams(
        CL=23.0,    # CL/F (L/h)
        V1=98.0,    # V1/F (L)
        Q=100.0,    # Q/F  (L/h)
        V2=600.0,   # V2/F (L)
        Ka=4.5,     # h^-1
        F=0.25,     # bioavailability
    ),
    omega_matrix=[
        [0.20, 0.00, 0.00, 0.00],
        [0.00, 0.15, 0.00, 0.00],
        [0.00, 0.00, 0.25, 0.00],
        [0.00, 0.00, 0.00, 0.20],
    ],
    error_model=ErrorModel(
        sigma_prop=0.15,
        sigma_add=0.30,
        model_type=ErrorModelType.COMBINED,
    ),
    reference="Antignac et al. (2007), Clin Pharmacokinet, 46(8)",
)


# ──────────────────────────────────────────────────────────────────
# Vancomycin 1-comp IV — Pediatric (from MCMCcode.R)
# ──────────────────────────────────────────────────────────────────

def maturation_factor(pma: float, tm50: float = 34.8, hill: float = 4.53) -> float:
    """
    Hill sigmoid maturation function for pediatric clearance.

    Models the maturation of renal function/metabolism with age:
        MF = 1 / (1 + (PMA / TM50)^(-Hill))

    When PMA >> TM50 → MF ≈ 1.0 (adult-like clearance)
    When PMA << TM50 → MF ≈ 0.0 (immature clearance)
    When PMA  = TM50 → MF = 0.5 (half-maturation)

    Args:
        pma:    Post-menstrual age in weeks
        tm50:   PMA at 50% maturation (default: 34.8 weeks)
        hill:   Hill coefficient (steepness, default: 4.53)

    Returns:
        Maturation factor [0, 1]

    Reference:
        MCMCcode.R (Vancomycin nhi khoa)
        Anderson & Holford (2008), Annu Rev Pharmacol Toxicol
    """
    if pma <= 0:
        return 0.0
    ratio = pma / tm50
    return 1.0 / (1.0 + ratio ** (-hill))


VANCOMYCIN_PEDI = PopPKModel(
    name="Vancomycin Pediatric 1-comp IV",
    drug="vancomycin",
    model_type=ModelType.ONE_COMP_PEDI_IV,
    typical_values=PKParams(
        CL=0.345,   # L/h (reference: 2.9 kg neonate)
        V1=1.75,    # L   (reference: 2.9 kg neonate)
        Q=0.0,      # 1-comp model
        V2=0.0,     # 1-comp model
    ),
    omega_matrix=[
        [0.0466, 0.000],   # omega^2 for CL (~21% CV)
        [0.000,  0.0119],  # omega^2 for V  (~11% CV)
    ],
    error_model=ErrorModel(
        sigma_prop=0.205,    # 20.5% proportional
        sigma_add=1.14,      # sqrt(1.3) ≈ 1.14 mg/L additive
        model_type=ErrorModelType.COMBINED,
    ),
    reference="MCMCcode.R (Vancomycin nhi khoa, adapted from clinical data)",
)


# ──────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, PopPKModel] = {
    "vancomycin_vn": VANCOMYCIN_VN,
    "vancomycin_pedi": VANCOMYCIN_PEDI,
    "tacrolimus_oral": TACROLIMUS_ORAL,
}


def get_model(name: str) -> PopPKModel:
    """Get a PopPK model by name."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{name}'. Available: {available}"
        )
    return MODEL_REGISTRY[name]


# ──────────────────────────────────────────────────────────────────
# Covariate model: compute individual typical values
# ──────────────────────────────────────────────────────────────────

def compute_vancomycin_tv(patient: PatientData) -> PKParams:
    """
    Compute typical PK values for vancomycin using covariate model.

    Covariate relationships (allometric + renal function):
        CL = TV_CL * (CrCL / 85)^0.75 * (WT / 70)^0.75
        V1 = TV_V1 * (WT / 70)^1.0
        Q  = TV_Q  * (WT / 70)^0.75
        V2 = TV_V2 * (WT / 70)^1.0

    Args:
        patient: Patient demographic and clinical data

    Returns:
        PKParams with covariate-adjusted typical values
    """
    tv = VANCOMYCIN_VN.typical_values
    dosing_weight = compute_weight_for_dosing(patient)

    # CrCL for renal function covariate
    crcl = cockcroft_gault_crcl(
        age=patient.age,
        weight=dosing_weight,
        serum_creatinine=patient.serum_creatinine,
        gender=patient.gender,
    )

    # Reference values for centering
    crcl_ref = 85.0   # mL/min (median)
    wt_ref = 70.0     # kg (reference weight)
    wt_ratio = dosing_weight / wt_ref
    crcl_ratio = crcl / crcl_ref

    # Apply covariate relationships
    cl_tv = tv.CL * (crcl_ratio ** 0.75) * (wt_ratio ** 0.75)
    v1_tv = tv.V1 * (wt_ratio ** 1.0)
    q_tv = tv.Q * (wt_ratio ** 0.75)
    v2_tv = tv.V2 * (wt_ratio ** 1.0)

    return PKParams(CL=cl_tv, V1=v1_tv, Q=q_tv, V2=v2_tv)


def compute_vancomycin_pedi_tv(patient: PatientData) -> PKParams:
    """
    Compute typical PK values for *pediatric* vancomycin using covariate model.

    From MCMCcode.R:
        CL = TVCL × (WT/2.9)^0.75 × (1/Scr)^0.267 × MF(PMA) / (1 + (PMA/TM50)^(-Hill))
        V  = TVV  × (WT/2.9)

    Maturation function (Hill sigmoid):
        MF(PMA) = 1 / (1 + (PMA/34.8)^(-4.53))

    Args:
        patient: Patient data (must include pma in weeks)

    Returns:
        PKParams with covariate-adjusted typical values for pediatric patient
    """
    tv = VANCOMYCIN_PEDI.typical_values
    wt = max(0.5, patient.weight)   # kg
    scr = max(0.1, patient.serum_creatinine)  # mg/dL

    # PMA: postmenstrual age in weeks
    # If not provided, estimate from age: PMA ≈ 40 + age_years × 52
    pma = patient.pma if patient.pma is not None else (40 + patient.age * 52)

    # Reference weight for allometric scaling: 2.9 kg (neonatal reference)
    wt_ref = 2.9
    wt_ratio = wt / wt_ref

    # Maturation factor
    mf = maturation_factor(pma, tm50=34.8, hill=4.53)

    # Covariate-adjusted typical values
    cl_tv = tv.CL * (wt_ratio ** 0.75) * ((1.0 / scr) ** 0.267) * mf
    v_tv = tv.V1 * (wt / wt_ref)

    return PKParams(CL=cl_tv, V1=v_tv, Q=0.0, V2=0.0)


# ──────────────────────────────────────────────────────────────────
# Apply IIV (random effects) to typical values
# ──────────────────────────────────────────────────────────────────

def apply_iiv(
    tv_params: PKParams,
    eta: NDArray[np.float64],
) -> PKParams:
    """
    Apply inter-individual variability (IIV) as log-normal random effects.

    Formula: theta_i = TV * exp(eta_i)

    Args:
        tv_params: Typical (population) values
        eta:       Random effects vector [eta_CL, eta_V1, eta_Q, eta_V2]
                   drawn from N(0, Omega)

    Returns:
        PKParams with individual values
    """
    if len(eta) < 2:
        raise ValueError("eta must have at least 2 elements [CL, V1]")

    cl_i = tv_params.CL * math.exp(eta[0])
    v1_i = tv_params.V1 * math.exp(eta[1])

    q_i = tv_params.Q
    v2_i = tv_params.V2
    if len(eta) >= 4:
        q_i = tv_params.Q * math.exp(eta[2])
        v2_i = tv_params.V2 * math.exp(eta[3])

    return PKParams(
        CL=cl_i, V1=v1_i, Q=q_i, V2=v2_i,
        Ka=tv_params.Ka, F=tv_params.F,
    )


def sample_individual_params(
    tv_params: PKParams,
    omega: list[list[float]],
    rng: np.random.Generator | None = None,
) -> PKParams:
    """
    Sample one set of individual PK parameters from population distribution.

    Draws eta ~ N(0, Omega) and applies as log-normal IIV.

    Args:
        tv_params: Typical values (from covariate model)
        omega:     Omega variance-covariance matrix
        rng:       Random number generator (for reproducibility)

    Returns:
        PKParams for one simulated individual
    """
    if rng is None:
        rng = np.random.default_rng()

    omega_arr = np.array(omega, dtype=np.float64)
    n_params = omega_arr.shape[0]
    mean = np.zeros(n_params)

    eta = rng.multivariate_normal(mean, omega_arr)
    return apply_iiv(tv_params, eta)
