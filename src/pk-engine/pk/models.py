"""
PK Data Models – Shared data structures for the entire PK engine.

All dataclasses are pure Python with no external dependencies.
They define the vocabulary of the system.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────

class Route(str, Enum):
    """Drug administration route."""
    IV_BOLUS = "iv_bolus"
    IV_INFUSION = "iv_infusion"
    ORAL = "oral"


class ModelType(str, Enum):
    """Compartmental PK model type."""
    ONE_COMP_IV = "1comp_iv"
    TWO_COMP_IV = "2comp_iv"
    ONE_COMP_ORAL = "1comp_oral"
    TWO_COMP_ORAL = "2comp_oral"


class Gender(str, Enum):
    """Patient gender for clinical calculations."""
    MALE = "male"
    FEMALE = "female"


class ErrorModelType(str, Enum):
    """Residual error model type."""
    ADDITIVE = "additive"
    PROPORTIONAL = "proportional"
    COMBINED = "combined"


# ──────────────────────────────────────────────────────────────────
# PK Parameters
# ──────────────────────────────────────────────────────────────────

@dataclass
class PKParams:
    """
    Individual pharmacokinetic parameters.

    Attributes:
        CL: Systemic clearance (L/h)
        V1: Central volume of distribution (L)
        Q:  Inter-compartmental clearance (L/h), 0 for 1-comp
        V2: Peripheral volume (L), 0 for 1-comp
        Ka: Absorption rate constant (h-1), 0 for IV
        F:  Oral bioavailability (0-1), 1.0 for IV
    """
    CL: float
    V1: float
    Q: float = 0.0
    V2: float = 0.0
    Ka: float = 0.0
    F: float = 1.0

    @property
    def ke(self) -> float:
        """Elimination rate constant: ke = CL / V1 (h-1)."""
        if self.V1 <= 0:
            raise ValueError("V1 must be positive")
        return self.CL / self.V1

    @property
    def k12(self) -> float:
        """Transfer rate central to peripheral: k12 = Q / V1."""
        if self.V1 <= 0:
            return 0.0
        return self.Q / self.V1

    @property
    def k21(self) -> float:
        """Transfer rate peripheral to central: k21 = Q / V2."""
        if self.V2 <= 0:
            return 0.0
        return self.Q / self.V2

    @property
    def half_life(self) -> float:
        """Terminal half-life: t1/2 = ln(2) / ke (hours)."""
        ke_val = self.ke
        if ke_val <= 0:
            raise ValueError("ke must be positive")
        return 0.693147 / ke_val

    def validate(self) -> list[str]:
        """Return list of validation error messages (empty = valid)."""
        errors: list[str] = []
        if self.CL <= 0:
            errors.append("CL must be positive")
        if self.V1 <= 0:
            errors.append("V1 must be positive")
        if self.Q < 0:
            errors.append("Q must be non-negative")
        if self.V2 < 0:
            errors.append("V2 must be non-negative")
        if self.Ka < 0:
            errors.append("Ka must be non-negative")
        if not (0.0 <= self.F <= 1.0):
            errors.append("F must be between 0 and 1")
        return errors


# ──────────────────────────────────────────────────────────────────
# Dose event
# ──────────────────────────────────────────────────────────────────

@dataclass
class DoseEvent:
    """
    A single drug administration event.

    Attributes:
        time:     Administration start time (hours from time-zero)
        amount:   Dose amount (mg)
        duration: Infusion duration (hours). 0 = bolus or instant oral.
        route:    Route of administration
    """
    time: float
    amount: float
    duration: float = 0.0
    route: Route = Route.IV_INFUSION

    def validate(self) -> list[str]:
        """Return list of validation error messages."""
        errors: list[str] = []
        if self.time < 0:
            errors.append("Dose time must be non-negative")
        if self.amount <= 0:
            errors.append("Dose amount must be positive")
        if self.duration < 0:
            errors.append("Infusion duration must be non-negative")
        return errors


# ──────────────────────────────────────────────────────────────────
# Observation (TDM sample)
# ──────────────────────────────────────────────────────────────────

@dataclass
class Observation:
    """
    A TDM concentration measurement.

    Attributes:
        time:          Sampling time (hours from time-zero)
        concentration: Measured drug concentration (mg/L)
        sample_type:   Type of sample (trough, peak, random)
    """
    time: float
    concentration: float
    sample_type: str = "trough"

    def validate(self) -> list[str]:
        """Return list of validation error messages."""
        errors: list[str] = []
        if self.time < 0:
            errors.append("Observation time must be non-negative")
        if self.concentration < 0:
            errors.append("Concentration must be non-negative")
        return errors


# ──────────────────────────────────────────────────────────────────
# Patient demographics
# ──────────────────────────────────────────────────────────────────

@dataclass
class PatientData:
    """
    Patient demographic and clinical data for covariate modeling.

    Attributes:
        age:                Age in years
        weight:             Total body weight (kg)
        height:             Height (cm)
        gender:             Male or Female
        serum_creatinine:   Serum creatinine (mg/dL)
        albumin:            Serum albumin (g/dL), optional
        is_icu:             Whether patient is in ICU
        is_on_dialysis:     Whether patient is on dialysis
    """
    age: float
    weight: float
    height: float
    gender: Gender
    serum_creatinine: float
    albumin: Optional[float] = None
    is_icu: bool = False
    is_on_dialysis: bool = False


# ──────────────────────────────────────────────────────────────────
# Residual error model
# ──────────────────────────────────────────────────────────────────

@dataclass
class ErrorModel:
    """
    Residual error model parameters.

    Combined: Var(C_obs) = (sigma_prop * C_pred)^2 + sigma_add^2

    Attributes:
        sigma_prop: Proportional error SD (unitless fraction)
        sigma_add:  Additive error SD (mg/L)
        model_type: Type of error model
    """
    sigma_prop: float = 0.0
    sigma_add: float = 0.0
    model_type: ErrorModelType = ErrorModelType.COMBINED

    def variance(self, c_pred: float) -> float:
        """
        Compute residual variance at a predicted concentration.

        Combined: Var = (sigma_prop * C_pred)^2 + sigma_add^2
        """
        if self.model_type == ErrorModelType.ADDITIVE:
            return self.sigma_add ** 2
        if self.model_type == ErrorModelType.PROPORTIONAL:
            return (self.sigma_prop * c_pred) ** 2
        # Combined
        return (self.sigma_prop * c_pred) ** 2 + self.sigma_add ** 2


# ──────────────────────────────────────────────────────────────────
# PopPK model definition
# ──────────────────────────────────────────────────────────────────

@dataclass
class PopPKModel:
    """
    Population PK model definition.

    Attributes:
        name:            Model name (e.g. "Vancomycin VN 2-comp")
        drug:            Drug name
        model_type:      Compartmental model type
        typical_values:  Typical (population) PK parameter values
        omega_matrix:    IIV variance-covariance matrix (list of lists)
        error_model:     Residual error model
        reference:       Literature reference / DOI
    """
    name: str
    drug: str
    model_type: ModelType
    typical_values: PKParams
    omega_matrix: list[list[float]]
    error_model: ErrorModel
    reference: str = ""
