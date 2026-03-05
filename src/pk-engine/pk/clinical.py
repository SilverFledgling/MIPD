"""
Clinical Calculations – CrCL, eGFR, BMI, BSA, IBW, ABW.

Reference:
    - Cockcroft & Gault (1976), Nephron 16(1), 31-41
    - Inker et al. (2021), NEJM 385:1737-1749 (CKD-EPI 2021)
    - DuBois & DuBois (1916), Arch Intern Med 17:863-871

All functions are pure math: input numbers, output numbers.
No external dependencies beyond the models module.
"""

from pk.models import Gender, PatientData


def cockcroft_gault_crcl(
    age: float,
    weight: float,
    serum_creatinine: float,
    gender: Gender,
) -> float:
    """
    Cockcroft-Gault creatinine clearance (mL/min).

    Formula:
        CrCL = [(140 - Age) * Weight] / (72 * SCr)
        If female: multiply by 0.85

    Args:
        age:               Age in years
        weight:            Body weight in kg (use ABW if obese)
        serum_creatinine:  Serum creatinine in mg/dL
        gender:            Male or Female

    Returns:
        CrCL in mL/min

    Raises:
        ValueError: If inputs are non-positive where required
    """
    if age <= 0:
        raise ValueError("Age must be positive")
    if weight <= 0:
        raise ValueError("Weight must be positive")
    if serum_creatinine <= 0:
        raise ValueError("Serum creatinine must be positive")

    crcl = ((140.0 - age) * weight) / (72.0 * serum_creatinine)

    if gender == Gender.FEMALE:
        crcl *= 0.85

    return crcl


def ckd_epi_egfr(
    serum_creatinine: float,
    age: float,
    gender: Gender,
) -> float:
    """
    CKD-EPI 2021 estimated GFR (mL/min/1.73m2).

    Race-free equation (Inker et al., 2021, NEJM).

    Formula:
        eGFR = 142 * min(SCr/kappa, 1)^alpha
                   * max(SCr/kappa, 1)^(-1.200)
                   * 0.9938^Age
                   * (1.012 if female)

    Args:
        serum_creatinine: Serum creatinine in mg/dL
        age:              Age in years
        gender:           Male or Female

    Returns:
        eGFR in mL/min/1.73m2
    """
    if serum_creatinine <= 0:
        raise ValueError("Serum creatinine must be positive")
    if age <= 0:
        raise ValueError("Age must be positive")

    if gender == Gender.FEMALE:
        kappa = 0.7
        alpha = -0.241
        sex_factor = 1.012
    else:
        kappa = 0.9
        alpha = -0.302
        sex_factor = 1.0

    scr_ratio = serum_creatinine / kappa
    min_val = min(scr_ratio, 1.0)
    max_val = max(scr_ratio, 1.0)

    egfr = (
        142.0
        * (min_val ** alpha)
        * (max_val ** (-1.200))
        * (0.9938 ** age)
        * sex_factor
    )

    return egfr


def bmi(weight: float, height_cm: float) -> float:
    """
    Body Mass Index (kg/m2).

    Formula: BMI = Weight / (Height_m)^2

    Args:
        weight:    Body weight in kg
        height_cm: Height in cm

    Returns:
        BMI in kg/m2
    """
    if weight <= 0:
        raise ValueError("Weight must be positive")
    if height_cm <= 0:
        raise ValueError("Height must be positive")

    height_m = height_cm / 100.0
    return weight / (height_m ** 2)


def bsa_dubois(weight: float, height_cm: float) -> float:
    """
    Body Surface Area – DuBois formula (m2).

    Formula: BSA = 0.007184 * Height^0.725 * Weight^0.425

    Args:
        weight:    Body weight in kg
        height_cm: Height in cm

    Returns:
        BSA in m2
    """
    if weight <= 0:
        raise ValueError("Weight must be positive")
    if height_cm <= 0:
        raise ValueError("Height must be positive")

    return 0.007184 * (height_cm ** 0.725) * (weight ** 0.425)


def ideal_body_weight(height_cm: float, gender: Gender) -> float:
    """
    Ideal Body Weight – Devine formula (kg).

    Formula:
        Male:   IBW = 50 + 2.3 * (Height_inches - 60)
        Female: IBW = 45.5 + 2.3 * (Height_inches - 60)

    Args:
        height_cm: Height in cm
        gender:    Male or Female

    Returns:
        IBW in kg
    """
    if height_cm <= 0:
        raise ValueError("Height must be positive")

    height_inches = height_cm / 2.54

    if gender == Gender.MALE:
        return 50.0 + 2.3 * (height_inches - 60.0)
    return 45.5 + 2.3 * (height_inches - 60.0)


def adjusted_body_weight(
    total_body_weight: float,
    height_cm: float,
    gender: Gender,
    correction_factor: float = 0.4,
) -> float:
    """
    Adjusted Body Weight for obese patients (kg).

    Formula: ABW = IBW + CF * (TBW - IBW)

    Args:
        total_body_weight:  Actual body weight (kg)
        height_cm:          Height in cm
        gender:             Male or Female
        correction_factor:  Typically 0.4

    Returns:
        ABW in kg
    """
    ibw = ideal_body_weight(height_cm, gender)
    return ibw + correction_factor * (total_body_weight - ibw)


def compute_weight_for_dosing(patient: PatientData) -> float:
    """
    Determine appropriate weight for PK calculations.

    Uses ABW if BMI > 30, otherwise uses total body weight.

    Args:
        patient: PatientData object

    Returns:
        Weight in kg for dosing calculations
    """
    patient_bmi = bmi(patient.weight, patient.height)
    if patient_bmi > 30.0:
        return adjusted_body_weight(
            patient.weight, patient.height, patient.gender
        )
    return patient.weight


def compute_crcl_for_patient(patient: PatientData) -> float:
    """
    Compute CrCL for a patient using appropriate weight.

    Uses ABW if BMI > 30, per vancomycin dosing guidelines.

    Args:
        patient: PatientData object

    Returns:
        CrCL in mL/min
    """
    dosing_weight = compute_weight_for_dosing(patient)
    return cockcroft_gault_crcl(
        age=patient.age,
        weight=dosing_weight,
        serum_creatinine=patient.serum_creatinine,
        gender=patient.gender,
    )
