"""Synthetic Electronic Health Record (EHR) Cohort Generator for Type 2 Diabetes.

This module generates clinically realistic synthetic patient profiles for Type 2
Diabetes Mellitus (T2DM), capturing physiological correlations, glycemic control,
diabetic complications risk, multi-morbidity burden, and pharmacological management.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    """Min-max normalize a numeric array to [0.0, 1.0].

    Parameters
    ----------
    values : np.ndarray
        Input 1D array of numerical observations.

    Returns
    -------
    np.ndarray
        Normalized values bounded between 0.0 and 1.0.
    """
    v_min: float = float(np.min(values))
    v_max: float = float(np.max(values))
    if v_max == v_min:
        return np.zeros_like(values, dtype=float)
    return (values - v_min) / (v_max - v_min)


def print_cohort_summary(df: pd.DataFrame) -> None:
    """Print descriptive statistics and medication prevalence for the cohort.

    Parameters
    ----------
    df : pd.DataFrame
        Cohort DataFrame containing demographic, clinical, and medication columns.
    """
    print("=" * 65)
    print("        EHR SYNTHETIC COHORT SUMMARY (TYPE 2 DIABETES)          ")
    print("=" * 65)
    print(f"Total Patients: {len(df)}")
    print("-" * 65)
    print("Key Continuous Biomarkers (Mean ± Std, Range):")
    key_fields = ["age", "bmi", "hba1c", "fasting_glucose", "egfr"]
    for field in key_fields:
        mean_val: float = float(df[field].mean())
        std_val: float = float(df[field].std())
        min_val: float = float(df[field].min())
        max_val: float = float(df[field].max())
        print(f"  - {field:<17}: {mean_val:6.2f} ± {std_val:5.2f}  [{min_val:.1f} - {max_val:.1f}]")

    print("-" * 65)
    print("Medication Prevalence Percentages:")
    med_fields = [
        "med_metformin",
        "med_insulin_basal",
        "med_insulin_rapid",
        "med_sglt2i",
        "med_glp1ra",
    ]
    for med in med_fields:
        prev_pct: float = float(df[med].mean() * 100.0)
        patient_count: int = int(df[med].sum())
        print(f"  - {med:<20}: {prev_pct:5.1f}%  ({patient_count}/{len(df)} patients)")
    print("=" * 65)


def generate_cohort(
    n_patients: int = 300,
    output_path: str = "data/synthea/patient_profiles.csv"
) -> pd.DataFrame:
    """Generate a synthetic cohort of Type 2 Diabetes patients with realistic EHR data.

    Clinical Logic & Correlation Hierarchy:
    1. age: Normal(55, 10), clipped to [35, 75]
    2. years_with_t2dm: Normal(8, 5), clipped to [0.5, 25.0], constrained to < (age - 30)
    3. bmi: Normal(30.5, 5.5), correlated with age (+0.1 per year over 50), clipped to [22.0, 45.0]
    4. hba1c: Normal(7.8, 1.5), correlated with BMI (+0.05 per unit over 28) and
       years_with_t2dm (+0.08 per year), clipped to [5.7, 12.0]
    5. fasting_glucose: Derived from HbA1c via ADA eAG formula (28.7 * HbA1c - 46.7)
       with Normal(0, 15) additive noise, clipped to [90.0, 280.0]
    6. egfr: Normal(78, 22), negatively correlated with age (-0.8 per year over 55) and
       years_with_t2dm (-1.5 per year over 10), clipped to [30.0, 120.0]
    7. ldl: Normal(125, 35), clipped to [60.0, 200.0]
    8. hdl: Normal(45, 12), slightly higher for females (+5.0), clipped to [25.0, 80.0]
    9. systolic_bp: Normal(138, 18), correlated with age (+0.5 per year over 50) and
       BMI (+0.3 per unit over 30), clipped to [110.0, 180.0]
    10. comorbidity_count: Poisson(lambda=1.5), higher with aging and uncontrolled HbA1c, [0, 5]
    11. Pharmacotherapy (metformin, basal insulin, rapid insulin, SGLT2i, GLP1-RA)
    12. insulin_total_daily_dose: Weight-based basal dosing + prandial boluses, [0.0, 80.0]
    13. genetic_risk_score: Multigenic risk index correlated with HbA1c and BMI, [0.0, 1.0]

    Parameters
    ----------
    n_patients : int, default=300
        Number of synthetic patients to generate.
    output_path : str, default="data/synthea/patient_profiles.csv"
        Relative or absolute file path for the output CSV.

    Returns
    -------
    pd.DataFrame
        Complete EHR cohort DataFrame with exact 19 columns.
    """
    # Set seed for reproducible clinical simulation
    np.random.seed(42)

    # 1. Patient Identifiers: Zero-padded 3-digit identifiers (P001 to P300)
    patient_ids = [f"P{i:03d}" for i in range(1, n_patients + 1)]

    # 2. Age: Typical onset and prevalence of T2DM occurs in mid-to-older adulthood (35-75 years)
    raw_age = np.random.normal(55.0, 10.0, n_patients)
    age = np.clip(np.round(raw_age), 35, 75).astype(int)

    # 3. Years with T2DM: Generated before HbA1c to allow disease progression modeling.
    # Clinical rationale: T2DM is adult-onset; diagnosis occurs at age >= 30, so duration < (age - 30).
    raw_years = np.random.normal(8.0, 5.0, n_patients)
    max_allowed_years = np.minimum(25.0, age - 30.1)
    years_with_t2dm = np.clip(raw_years, 0.5, max_allowed_years)
    years_with_t2dm = np.round(years_with_t2dm, 1)

    # 4. BMI: Insulin resistance is heavily driven by obesity.
    # Clinical rationale: Sarcopenic body composition shifts and metabolic decline with aging (+0.1 per year over 50).
    base_bmi = np.random.normal(30.5, 5.5, n_patients)
    age_over_50 = np.maximum(0, age - 50)
    bmi = np.clip(np.round(base_bmi + 0.1 * age_over_50, 1), 22.0, 45.0)

    # 5. HbA1c: Core marker of chronic glycemia over preceding 90-120 days.
    # Clinical rationale: Adiposity (BMI > 28) worsens peripheral insulin resistance (+0.05 per unit).
    # Disease chronicity leads to progressive pancreatic beta-cell exhaust (+0.08 per year with T2DM).
    # Base centered so that AFTER adding BMI/duration effects the cohort mean HbA1c
    # lands near ~7.6% (realistic for a monitored T2DM population) rather than skewing
    # severe. Slopes are reduced to keep the right tail clinically plausible.
    base_hba1c = np.random.normal(6.9, 1.1, n_patients)
    bmi_over_28 = np.maximum(0.0, bmi - 28.0)
    hba1c = np.clip(
        np.round(base_hba1c + 0.03 * bmi_over_28 + 0.045 * years_with_t2dm, 1),
        5.7,
        12.0
    )

    # 6. Fasting Glucose: Correlated with glycated hemoglobin via ADA estimated Average Glucose (eAG) formula.
    # Clinical rationale: Biological variability in red blood cell turnover and daily glycemic excursions
    # introduces non-deterministic physiological scatter (additive Normal(0, 15) noise).
    eag = 28.7 * hba1c - 46.7
    glucose_noise = np.random.normal(0.0, 15.0, n_patients)
    fasting_glucose = np.clip(np.round(eag + glucose_noise, 1), 90.0, 280.0)

    # 7. Sex: Epidemiological balance showing slight male preponderance in T2DM prevalence (55% M / 45% F).
    sex = np.random.choice(["M", "F"], size=n_patients, p=[0.55, 0.45])

    # 8. eGFR: Renal clearance metric (estimated Glomerular Filtration Rate in mL/min/1.73m2).
    # Clinical rationale: Nephrosclerosis accelerates with chronological age (-0.8 per year over 55)
    # and diabetic kidney disease (microvascular nephropathy, -1.5 per year of diabetes over 10).
    base_egfr = np.random.normal(78.0, 22.0, n_patients)
    age_over_55 = np.maximum(0, age - 55)
    years_over_10 = np.maximum(0.0, years_with_t2dm - 10.0)
    egfr = np.clip(
        np.round(base_egfr - 0.8 * age_over_55 - 1.5 * years_over_10, 1),
        30.0,
        120.0
    )

    # 9. LDL: Low-density lipoprotein cholesterol (atherogenic lipid fraction in mg/dL).
    base_ldl = np.random.normal(125.0, 35.0, n_patients)
    ldl = np.clip(np.round(base_ldl, 1), 60.0, 200.0)

    # 10. HDL: High-density lipoprotein cholesterol (cardioprotective fraction in mg/dL).
    # Clinical rationale: Estrogen receptor mediated hepatic ApoA-I synthesis confers higher baseline HDL in females (+5 mg/dL).
    base_hdl = np.random.normal(45.0, 12.0, n_patients)
    female_bonus = np.where(sex == "F", 5.0, 0.0)
    hdl = np.clip(np.round(base_hdl + female_bonus, 1), 25.0, 80.0)

    # 11. Systolic Blood Pressure: Vascular stiffness and systemic vascular resistance in mmHg.
    # Clinical rationale: Arterial stiffening with aging (+0.5 mmHg per year over 50) and
    # hyperinsulinemia/fluid retention associated with obesity (+0.3 mmHg per BMI unit over 30).
    base_sbp = np.random.normal(138.0, 18.0, n_patients)
    bmi_over_30 = np.maximum(0.0, bmi - 30.0)
    systolic_bp = np.clip(
        np.round(base_sbp + 0.5 * age_over_50 + 0.3 * bmi_over_30, 1),
        110.0,
        180.0
    )

    # 12. Comorbidity Count: Multi-morbidity burden (CAD, HTN, CKD, neuropathy, retinopathy).
    # Clinical rationale: Older age and chronic glucotoxicity increase cumulative systemic organ damage.
    lambda_comorbidity = np.maximum(
        0.1,
        1.5 + 0.03 * np.maximum(0, age - 50) + 0.15 * np.maximum(0.0, hba1c - 7.0)
    )
    comorbidity_count = np.clip(np.random.poisson(lambda_comorbidity), 0, 5).astype(int)

    # 13. Metformin: First-line oral biguanide therapy.
    # Clinical rationale: Recommended for almost all patients with HbA1c >= 6.5 unless contraindicated.
    prob_metformin = np.where(hba1c >= 6.5, 0.80, 0.20)
    med_metformin = (np.random.uniform(0.0, 1.0, n_patients) < prob_metformin).astype(int)

    # 14. Basal Insulin: Long-acting baseline glycemic suppression.
    # Clinical rationale: Indicated as add-on when oral agents fail to keep HbA1c under target (especially HbA1c >= 8.0).
    prob_basal = np.where(hba1c >= 8.0, 0.50, np.where(hba1c >= 7.0, 0.20, 0.05))
    med_insulin_basal = (np.random.uniform(0.0, 1.0, n_patients) < prob_basal).astype(int)

    # 15. Rapid-acting Insulin: Mealtime prandial boluses.
    # Clinical rationale: Escalated in intensive basal-bolus regimens for severe hyperglycemia (HbA1c >= 9.0) already on basal.
    prob_rapid = np.where((hba1c >= 9.0) & (med_insulin_basal == 1), 0.30, 0.05)
    med_insulin_rapid = (np.random.uniform(0.0, 1.0, n_patients) < prob_rapid).astype(int)

    # 16. SGLT2 Inhibitors: Renal glucosuric agents.
    # Clinical rationale: Second-line oral therapy with cardioprotective and nephroprotective benefits for HbA1c >= 7.0.
    prob_sglt2i = np.where(hba1c >= 7.0, 0.35, 0.10)
    med_sglt2i = (np.random.uniform(0.0, 1.0, n_patients) < prob_sglt2i).astype(int)

    # 17. GLP1 Receptor Agonists: Incretin mimetics.
    # Clinical rationale: Preferred for patients with significant obesity (BMI >= 30) and suboptimal control (HbA1c >= 7.5).
    prob_glp1ra = np.where((bmi >= 30.0) & (hba1c >= 7.5), 0.25, 0.05)
    med_glp1ra = (np.random.uniform(0.0, 1.0, n_patients) < prob_glp1ra).astype(int)

    # 18. Insulin Total Daily Dose (Units/day):
    # Clinical rationale: Clinical dosing titration is weight-adjusted (~0.3 U/kg basal dose) plus fixed mealtime bolus coverage.
    height_m = 1.68
    weight_kg = bmi * (height_m ** 2)
    basal_dose = np.where(med_insulin_basal == 1, np.random.normal(0.3 * weight_kg, 8.0), 0.0)
    rapid_dose = np.where(med_insulin_rapid == 1, np.random.normal(15.0, 5.0, n_patients), 0.0)
    insulin_total_daily_dose = np.clip(np.round(basal_dose + rapid_dose, 1), 0.0, 80.0)

    # 19. Genetic Risk Score: Polygenic liability score normalized to [0.0, 1.0].
    # Clinical rationale: Polygenic susceptibility interacts with adiposity and predisposes to higher glycemic set-points.
    norm_hba1c = _min_max_normalize(hba1c)
    norm_bmi = _min_max_normalize(bmi)
    beta_residual = np.random.beta(2.0, 5.0, n_patients)
    grs = 0.25 * norm_hba1c + 0.15 * norm_bmi + 0.60 * beta_residual
    genetic_risk_score = np.clip(np.round(grs, 3), 0.0, 1.0)

    # Construct DataFrame in the exact 19-column order from the specification
    columns = [
        "patient_id",
        "age",
        "sex",
        "bmi",
        "hba1c",
        "fasting_glucose",
        "egfr",
        "ldl",
        "hdl",
        "systolic_bp",
        "years_with_t2dm",
        "comorbidity_count",
        "med_metformin",
        "med_insulin_basal",
        "med_insulin_rapid",
        "med_sglt2i",
        "med_glp1ra",
        "insulin_total_daily_dose",
        "genetic_risk_score",
    ]

    cohort_dict = {
        "patient_id": patient_ids,
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "hba1c": hba1c,
        "fasting_glucose": fasting_glucose,
        "egfr": egfr,
        "ldl": ldl,
        "hdl": hdl,
        "systolic_bp": systolic_bp,
        "years_with_t2dm": years_with_t2dm,
        "comorbidity_count": comorbidity_count,
        "med_metformin": med_metformin,
        "med_insulin_basal": med_insulin_basal,
        "med_insulin_rapid": med_insulin_rapid,
        "med_sglt2i": med_sglt2i,
        "med_glp1ra": med_glp1ra,
        "insulin_total_daily_dose": insulin_total_daily_dose,
        "genetic_risk_score": genetic_risk_score,
    }

    df = pd.DataFrame(cohort_dict, columns=columns)

    # Ensure output directory exists and save CSV
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    cohort_df = generate_cohort(
        n_patients=300,
        output_path="data/synthea/patient_profiles.csv"
    )
    print_cohort_summary(cohort_df)
