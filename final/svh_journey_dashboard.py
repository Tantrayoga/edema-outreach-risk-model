
import os
import json
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# SVH JOURNEY INTELLIGENCE DASHBOARD
# ============================================================

st.set_page_config(
    page_title="SVH Journey Intelligence Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(".")
OUTPUT_DIR = Path("../outputs")
DATA_DIR = Path("../data")
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
DASHBOARD_DIR = OUTPUT_DIR / "dashboard"

MASTER_PATH = OUTPUT_DIR / "master_encounters.parquet"

DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

PATIENT_COMPLEXITY_PATH = DASHBOARD_DIR / "dashboard_patient_complexity.parquet"
PATIENT_DX_JOURNEYS_PATH = DASHBOARD_DIR / "dashboard_patient_diagnosis_journeys.parquet"
DEPT_TRANSITIONS_PATH = DASHBOARD_DIR / "dashboard_department_transitions.parquet"
DEPT_METRICS_PATH = DASHBOARD_DIR / "dashboard_department_metrics.parquet"
DX_METRICS_PATH = DASHBOARD_DIR / "dashboard_diagnosis_metrics.parquet"
MONTHLY_OVERVIEW_PATH = DASHBOARD_DIR / "dashboard_monthly_overview.parquet"
SDOH_METRICS_PATH = DASHBOARD_DIR / "dashboard_sdoh_metrics.parquet"

# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #172033;
        margin-bottom: 0.2rem;
    }

    .dashboard-subtitle {
        font-size: 1.0rem;
        color: #5f6b7a;
        margin-bottom: 1.2rem;
    }

    .section-header {
        font-size: 1.4rem;
        font-weight: 750;
        color: #172033;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .metric-card {
        background: white;
        padding: 1.0rem 1.1rem;
        border-radius: 18px;
        box-shadow: 0 4px 15px rgba(23, 32, 51, 0.06);
        border: 1px solid rgba(23, 32, 51, 0.06);
        min-height: 105px;
    }

    .metric-label {
        font-size: 0.78rem;
        color: #6b7280;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.75rem;
        color: #111827;
        font-weight: 850;
        line-height: 1.1;
    }

    .metric-help {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 0.35rem;
    }

    .info-box {
        background: #eef6ff;
        border-left: 5px solid #2563eb;
        padding: 1rem;
        border-radius: 12px;
        color: #1e3a8a;
        margin: 0.8rem 0rem;
    }

    .warning-box {
        background: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 1rem;
        border-radius: 12px;
        color: #7c2d12;
        margin: 0.8rem 0rem;
    }

    .success-box {
        background: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 1rem;
        border-radius: 12px;
        color: #064e3b;
        margin: 0.8rem 0rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# Helper functions
# ============================================================

def q(col: str) -> str:
    return '"' + str(col).replace('"', '""') + '"'

def escape_sql_value(x) -> str:
    if x is None:
        return "NULL"
    return "'" + str(x).replace("'", "''") + "'"

def sql_in_list(col, values):
    if not values:
        return "1=1"
    vals = ", ".join(escape_sql_value(v) for v in values)
    return f"{q(col)} IN ({vals})"

def fmt_int(x):
    if pd.isna(x):
        return "—"
    return f"{int(x):,}"

def fmt_pct(x, digits=1):
    if pd.isna(x):
        return "—"
    return f"{float(x):.{digits}f}%"

def fmt_float(x, digits=2):
    if pd.isna(x):
        return "—"
    return f"{float(x):.{digits}f}"

def metric_card(label, value, help_text=None):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text if help_text else ""}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

@st.cache_data(show_spinner=False)
def get_columns(path_str):
    con = duckdb.connect()
    try:
        return con.sql(f"DESCRIBE SELECT * FROM '{path_str}'").fetchdf()["column_name"].tolist()
    finally:
        con.close()

@st.cache_data(show_spinner=False)
def run_query(sql):
    con = duckdb.connect()
    try:
        return con.sql(sql).fetchdf()
    finally:
        con.close()

def has_col(cols, col):
    return col in cols

def expr_col(cols, col, alias, default="NULL"):
    if col in cols:
        return f"{q(col)} AS {q(alias)}"
    return f"{default} AS {q(alias)}"

def bool_flag_expr(cols, col):
    if col not in cols:
        return "0"
    return f"""
    CASE
        WHEN TRY_CAST({q(col)} AS BIGINT) = 1 THEN 1
        WHEN LOWER(CAST({q(col)} AS VARCHAR)) IN ('true', 't', 'yes', 'y', '1') THEN 1
        ELSE 0
    END
    """

def los_expr(cols):
    if "enc_AdmissionInstant" in cols and "enc_DischargeInstant" in cols:
        return """
        CASE
            WHEN enc_AdmissionInstant IS NOT NULL
             AND enc_DischargeInstant IS NOT NULL
            THEN GREATEST(
                DATE_DIFF(
                    'hour',
                    TRY_CAST(enc_AdmissionInstant AS TIMESTAMP),
                    TRY_CAST(enc_DischargeInstant AS TIMESTAMP)
                ),
                0
            )
            ELSE NULL
        END
        """
    return "NULL"

def age_group_expr(cols):
    if "pat_PatientBirthYearBin" not in cols:
        return "'Unknown'"
    return """
    CASE
        WHEN TRY_CAST(pat_PatientBirthYearBin AS INTEGER) IS NULL THEN 'Unknown'
        WHEN TRY_CAST(pat_PatientBirthYearBin AS INTEGER) <= 1935 THEN '90+'
        WHEN TRY_CAST(pat_PatientBirthYearBin AS INTEGER) <= 1950 THEN '75-89'
        ELSE 'Under 75'
    END
    """

def deceased_expr(cols):
    if "pat_VitalStatus" not in cols:
        return "0"
    return """
    CASE
        WHEN UPPER(CAST(pat_VitalStatus AS VARCHAR)) LIKE '%DECEASED%'
          OR UPPER(CAST(pat_VitalStatus AS VARCHAR)) LIKE '%DEAD%'
          OR UPPER(CAST(pat_VitalStatus AS VARCHAR)) LIKE '%DIED%'
        THEN 1 ELSE 0
    END
    """

def safe_remove(path):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass



# ============================================================
# Dashboard category cleaning
# ============================================================
# Important:
# We do NOT delete Unknown from raw data.
# We hide/drop Unknown categories from dashboard visuals so charts are cleaner.
# This prevents big "Unknown" bars from dominating the demo.
# ============================================================

UNKNOWN_VALUES = {
    "",
    "unknown",
    "*unknown",
    "unspecified",
    "*unspecified",
    "not applicable",
    "*not applicable",
    "missing",
    "nan",
    "none",
    "null",
    "not recorded",
    "no information",
    "unavailable"
}

def normalize_unknown_value(x):
    if pd.isna(x):
        return pd.NA

    s = str(x).strip()
    s_lower = s.lower().strip()

    if s_lower in UNKNOWN_VALUES:
        return pd.NA

    # Sometimes the data has strings like "*Unknown" or "*Unspecified"
    if s_lower.startswith("*") and s_lower.replace("*", "").strip() in UNKNOWN_VALUES:
        return pd.NA

    return s

def clean_dashboard_df(df, dataset_name):
    """
    Cleans display/category values for dashboard visuals.

    We keep patient-level rows when possible, but for aggregate category tables
    we drop rows where the main grouping category is missing/unknown.
    """

    if df is None or len(df) == 0:
        return df

    df = df.copy()

    # Normalize text columns
    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in object_cols:
        df[col] = df[col].map(normalize_unknown_value)

    # Drop unknown-heavy rows only for category aggregate tables.
    # This keeps charts from being dominated by Unknown.
    key_cols_by_dataset = {
        "monthly": ["department_type"],
        "dept_metrics": ["department_type"],
        "dept_transitions": ["origin_department_type", "next_department_type"],
        "dx_metrics": ["diagnosis_group", "diagnosis_name"],
        "patient_dx": ["diagnosis_group", "diagnosis_name"],
        "sdoh": ["department_type", "diagnosis_group"],
    }

    key_cols = key_cols_by_dataset.get(dataset_name, [])

    for col in key_cols:
        if col in df.columns:
            df = df[df[col].notna()]

    return df


# ============================================================
# Asset precomputation
# ============================================================

def create_dashboard_assets(rebuild=False):
    if not MASTER_PATH.exists():
        st.error(f"Missing master file: {MASTER_PATH}")
        st.stop()

    required_paths = [
        PATIENT_COMPLEXITY_PATH,
        PATIENT_DX_JOURNEYS_PATH,
        DEPT_TRANSITIONS_PATH,
        DEPT_METRICS_PATH,
        DX_METRICS_PATH,
        MONTHLY_OVERVIEW_PATH,
        SDOH_METRICS_PATH
    ]

    if all(p.exists() for p in required_paths) and not rebuild:
        return

    cols = get_columns(str(MASTER_PATH))

    required_core = ["enc_PatientDurableKey", "enc_EncounterKey", "enc_Date"]
    missing_core = [c for c in required_core if c not in cols]
    if missing_core:
        st.error(f"Missing required master columns: {missing_core}")
        st.stop()

    if rebuild:
        for p in required_paths:
            safe_remove(p)

    is_ed = bool_flag_expr(cols, "enc_IsEdVisit")
    is_hosp = bool_flag_expr(cols, "enc_IsHospitalAdmission")
    is_inpatient = bool_flag_expr(cols, "enc_IsInpatientAdmission")
    is_obs = bool_flag_expr(cols, "enc_IsObservation")
    is_outpatient = bool_flag_expr(cols, "enc_IsOutpatientFaceToFaceVisit")
    is_hod = bool_flag_expr(cols, "enc_IsHospitalOutpatientVisit")
    los = los_expr(cols)
    age_group = age_group_expr(cols)
    deceased = deceased_expr(cols)

    dept_key = q("dept_DepartmentKey") if "dept_DepartmentKey" in cols else q("enc_DepartmentKey") if "enc_DepartmentKey" in cols else "NULL"
    dept_type = q("dept_DepartmentType") if "dept_DepartmentType" in cols else "'Unknown'"
    dept_specialty = q("dept_DepartmentSpecialty") if "dept_DepartmentSpecialty" in cols else "'Unknown'"
    provider_key = q("attending_provider_DurableKey") if "attending_provider_DurableKey" in cols else q("enc_AttendingProviderDurableKey") if "enc_AttendingProviderDurableKey" in cols else "NULL"

    dx_value = q("diag_DiagnosisValue") if "diag_DiagnosisValue" in cols else "NULL"
    dx_name = q("diag_DiagnosisName") if "diag_DiagnosisName" in cols else "NULL"
    dx_group = q("diag_GroupName") if "diag_GroupName" in cols else "NULL"

    enc_type = q("enc_Type") if "enc_Type" in cols else "NULL"

    sex = q("pat_SexAssignedAtBirth") if "pat_SexAssignedAtBirth" in cols else "NULL"
    vital = q("pat_VitalStatus") if "pat_VitalStatus" in cols else "NULL"

    sd_exists = "sd_EncounterKey" in cols
    sd_expr = "CASE WHEN sd_EncounterKey IS NOT NULL THEN 1 ELSE 0 END" if sd_exists else "0"

    con = duckdb.connect()

    try:
        # ------------------------------------------------------------
        # Patient-level complexity
        # ------------------------------------------------------------
        con.sql(f"""
        COPY (
            WITH base AS (
                SELECT
                    enc_PatientDurableKey AS patient_id,
                    enc_EncounterKey AS encounter_key,
                    TRY_CAST(enc_Date AS DATE) AS encounter_date,
                    {dept_key} AS department_key,
                    {dept_type} AS department_type,
                    {dept_specialty} AS department_specialty,
                    {provider_key} AS attending_provider_key,
                    {dx_value} AS diagnosis_value,
                    {dx_name} AS diagnosis_name,
                    {dx_group} AS diagnosis_group,
                    {enc_type} AS encounter_type,
                    {is_ed} AS is_ed_visit,
                    {is_hosp} AS is_hospital_admission,
                    {is_inpatient} AS is_inpatient_admission,
                    {is_obs} AS is_observation,
                    {is_outpatient} AS is_outpatient_face_to_face,
                    {is_hod} AS is_hospital_outpatient,
                    {los} AS los_hours,
                    {age_group} AS age_group,
                    {sex} AS sex_assigned_at_birth,
                    {vital} AS vital_status,
                    {deceased} AS is_deceased,
                    {sd_expr} AS has_sdoh_screening
                FROM '{MASTER_PATH}'
                WHERE enc_PatientDurableKey IS NOT NULL
            ),

            raw AS (
                SELECT
                    patient_id,
                    MIN(encounter_date) AS first_encounter_date,
                    MAX(encounter_date) AS last_encounter_date,
                    COALESCE(DATE_DIFF('day', MIN(encounter_date), MAX(encounter_date)), 0) AS journey_days,

                    COUNT(*) AS num_encounters,
                    COUNT(DISTINCT encounter_key) AS num_unique_encounters,
                    COUNT(DISTINCT diagnosis_value) AS num_diagnosis_values,
                    COUNT(DISTINCT diagnosis_group) AS num_diagnosis_groups,
                    COUNT(DISTINCT department_key) AS num_departments,
                    COUNT(DISTINCT department_type) AS num_department_types,
                    COUNT(DISTINCT department_specialty) AS num_department_specialties,
                    COUNT(DISTINCT attending_provider_key) AS num_attending_providers,

                    SUM(is_ed_visit) AS ed_visits,
                    SUM(is_hospital_admission) AS hospital_admissions,
                    SUM(is_inpatient_admission) AS inpatient_admissions,
                    SUM(is_observation) AS observation_visits,
                    SUM(is_outpatient_face_to_face) AS outpatient_face_to_face_visits,
                    SUM(is_hospital_outpatient) AS hospital_outpatient_visits,

                    AVG(los_hours) AS avg_los_hours,
                    MAX(los_hours) AS max_los_hours,

                    SUM(has_sdoh_screening) AS encounters_with_sdoh_screening,

                    ANY_VALUE(age_group) AS age_group,
                    ANY_VALUE(sex_assigned_at_birth) AS sex_assigned_at_birth,
                    ANY_VALUE(vital_status) AS vital_status,
                    MAX(is_deceased) AS is_deceased

                FROM base
                GROUP BY patient_id
            ),

            scored AS (
                SELECT
                    *,
                    CUME_DIST() OVER (ORDER BY num_encounters) AS volume_pct,
                    CUME_DIST() OVER (
                        ORDER BY num_departments + num_department_types + num_department_specialties
                    ) AS department_pct,
                    CUME_DIST() OVER (
                        ORDER BY num_diagnosis_values + num_diagnosis_groups
                    ) AS diagnosis_pct,
                    CUME_DIST() OVER (
                        ORDER BY ed_visits + 2 * hospital_admissions + 2 * inpatient_admissions + observation_visits
                    ) AS acute_pct,
                    CUME_DIST() OVER (
                        ORDER BY COALESCE(max_los_hours, 0) + COALESCE(avg_los_hours, 0)
                    ) AS los_pct,
                    CUME_DIST() OVER (
                        ORDER BY num_attending_providers
                    ) AS provider_pct
                FROM raw
            ),

            final_score AS (
                SELECT
                    *,
                    ROUND(
                        100 * (
                            0.25 * volume_pct
                            + 0.20 * department_pct
                            + 0.20 * diagnosis_pct
                            + 0.20 * acute_pct
                            + 0.10 * los_pct
                            + 0.05 * provider_pct
                        ),
                        2
                    ) AS journey_complexity_score
                FROM scored
            )

            SELECT
                *,
                CASE
                    WHEN journey_complexity_score >= 90 THEN 'Extreme'
                    WHEN journey_complexity_score >= 70 THEN 'High'
                    WHEN journey_complexity_score >= 40 THEN 'Moderate'
                    ELSE 'Low'
                END AS complexity_bucket
            FROM final_score
        )
        TO '{PATIENT_COMPLEXITY_PATH}'
        (FORMAT PARQUET)
        """)

        # ------------------------------------------------------------
        # Patient-diagnosis journeys
        # ------------------------------------------------------------
        con.sql(f"""
        COPY (
            WITH base AS (
                SELECT
                    enc_PatientDurableKey AS patient_id,
                    enc_EncounterKey AS encounter_key,
                    TRY_CAST(enc_Date AS DATE) AS encounter_date,
                    {dept_key} AS department_key,
                    {dept_type} AS department_type,
                    {dept_specialty} AS department_specialty,
                    {provider_key} AS attending_provider_key,
                    {dx_value} AS diagnosis_value,
                    {dx_name} AS diagnosis_name,
                    {dx_group} AS diagnosis_group,
                    {is_ed} AS is_ed_visit,
                    {is_hosp} AS is_hospital_admission,
                    {is_inpatient} AS is_inpatient_admission,
                    {is_obs} AS is_observation,
                    {los} AS los_hours,
                    {deceased} AS is_deceased
                FROM '{MASTER_PATH}'
                WHERE enc_PatientDurableKey IS NOT NULL
                  AND {dx_value} IS NOT NULL
            ),

            raw AS (
                SELECT
                    patient_id,
                    diagnosis_value,
                    ANY_VALUE(diagnosis_name) AS diagnosis_name,
                    ANY_VALUE(diagnosis_group) AS diagnosis_group,

                    MIN(encounter_date) AS first_dx_date,
                    MAX(encounter_date) AS last_dx_date,
                    COALESCE(DATE_DIFF('day', MIN(encounter_date), MAX(encounter_date)), 0) AS dx_journey_days,

                    COUNT(*) AS dx_encounters,
                    COUNT(DISTINCT encounter_key) AS dx_unique_encounters,
                    COUNT(DISTINCT department_key) AS dx_departments,
                    COUNT(DISTINCT department_type) AS dx_department_types,
                    COUNT(DISTINCT department_specialty) AS dx_department_specialties,
                    COUNT(DISTINCT attending_provider_key) AS dx_attending_providers,

                    SUM(is_ed_visit) AS dx_ed_visits,
                    SUM(is_hospital_admission) AS dx_hospital_admissions,
                    SUM(is_inpatient_admission) AS dx_inpatient_admissions,
                    SUM(is_observation) AS dx_observation_visits,

                    AVG(los_hours) AS dx_avg_los_hours,
                    MAX(los_hours) AS dx_max_los_hours,
                    MAX(is_deceased) AS is_deceased

                FROM base
                GROUP BY patient_id, diagnosis_value
            ),

            scored AS (
                SELECT
                    *,
                    (
                        dx_encounters
                        + 2 * dx_departments
                        + 2 * dx_department_types
                        + 2 * dx_department_specialties
                        + 3 * dx_ed_visits
                        + 4 * dx_hospital_admissions
                        + 4 * dx_inpatient_admissions
                        + 0.03 * dx_journey_days
                        + 0.02 * COALESCE(dx_max_los_hours, 0)
                    ) AS dx_complexity_raw
                FROM raw
            ),

            normalized AS (
                SELECT
                    *,
                    ROUND(
                        100 * CUME_DIST() OVER (
                            PARTITION BY COALESCE(diagnosis_group, 'Unknown')
                            ORDER BY dx_complexity_raw
                        ),
                        2
                    ) AS dx_complexity_percentile_within_group
                FROM scored
            )

            SELECT
                *,
                CASE
                    WHEN dx_complexity_percentile_within_group >= 95 THEN 'Extreme within diagnosis'
                    WHEN dx_complexity_percentile_within_group >= 80 THEN 'High within diagnosis'
                    WHEN dx_complexity_percentile_within_group >= 50 THEN 'Moderate within diagnosis'
                    ELSE 'Lower within diagnosis'
                END AS dx_complexity_bucket
            FROM normalized
        )
        TO '{PATIENT_DX_JOURNEYS_PATH}'
        (FORMAT PARQUET)
        """)

        # ------------------------------------------------------------
        # Diagnosis metrics
        # ------------------------------------------------------------
        con.sql(f"""
        COPY (
            SELECT
                diagnosis_group,
                diagnosis_name,
                diagnosis_value,
                COUNT(DISTINCT patient_id) AS patients,
                SUM(dx_encounters) AS encounters,
                ROUND(AVG(dx_encounters), 2) AS avg_encounters_per_patient,
                ROUND(MEDIAN(dx_journey_days), 2) AS median_dx_journey_days,
                ROUND(AVG(dx_complexity_raw), 2) AS avg_dx_complexity_raw,
                ROUND(AVG(dx_complexity_percentile_within_group), 2) AS avg_dx_complexity_percentile,
                ROUND(AVG(CASE WHEN dx_complexity_percentile_within_group >= 80 THEN 1 ELSE 0 END) * 100, 2) AS pct_high_complexity_within_group,
                ROUND(AVG(CASE WHEN dx_ed_visits > 0 THEN 1 ELSE 0 END) * 100, 2) AS pct_with_ed_visit,
                ROUND(AVG(CASE WHEN dx_hospital_admissions > 0 THEN 1 ELSE 0 END) * 100, 2) AS pct_with_hospital_admission,
                ROUND(AVG(CASE WHEN dx_inpatient_admissions > 0 THEN 1 ELSE 0 END) * 100, 2) AS pct_with_inpatient_admission,
                ROUND(AVG(is_deceased) * 100, 2) AS pct_deceased
            FROM '{PATIENT_DX_JOURNEYS_PATH}'
            GROUP BY diagnosis_group, diagnosis_name, diagnosis_value
        )
        TO '{DX_METRICS_PATH}'
        (FORMAT PARQUET)
        """)

        # ------------------------------------------------------------
        # Department transitions
        # ------------------------------------------------------------
        con.sql(f"""
        COPY (
            WITH ordered AS (
                SELECT
                    enc_PatientDurableKey AS patient_id,
                    enc_EncounterKey AS encounter_key,
                    TRY_CAST(enc_Date AS DATE) AS encounter_date,
                    {dept_key} AS department_key,
                    COALESCE(CAST({dept_type} AS VARCHAR), 'Unknown') AS department_type,
                    COALESCE(CAST({dept_specialty} AS VARCHAR), 'Unknown') AS department_specialty,
                    COALESCE(CAST({dx_value} AS VARCHAR), 'Unknown') AS diagnosis_value,

                    LEAD(TRY_CAST(enc_Date AS DATE)) OVER (
                        PARTITION BY enc_PatientDurableKey
                        ORDER BY TRY_CAST(enc_Date AS DATE), enc_EncounterKey
                    ) AS next_encounter_date,

                    LEAD({dept_key}) OVER (
                        PARTITION BY enc_PatientDurableKey
                        ORDER BY TRY_CAST(enc_Date AS DATE), enc_EncounterKey
                    ) AS next_department_key,

                    LEAD(COALESCE(CAST({dept_type} AS VARCHAR), 'Unknown')) OVER (
                        PARTITION BY enc_PatientDurableKey
                        ORDER BY TRY_CAST(enc_Date AS DATE), enc_EncounterKey
                    ) AS next_department_type,

                    LEAD(COALESCE(CAST({dept_specialty} AS VARCHAR), 'Unknown')) OVER (
                        PARTITION BY enc_PatientDurableKey
                        ORDER BY TRY_CAST(enc_Date AS DATE), enc_EncounterKey
                    ) AS next_department_specialty,

                    LEAD(COALESCE(CAST({dx_value} AS VARCHAR), 'Unknown')) OVER (
                        PARTITION BY enc_PatientDurableKey
                        ORDER BY TRY_CAST(enc_Date AS DATE), enc_EncounterKey
                    ) AS next_diagnosis_value

                FROM '{MASTER_PATH}'
                WHERE enc_PatientDurableKey IS NOT NULL
                  AND TRY_CAST(enc_Date AS DATE) IS NOT NULL
            ),

            transitions AS (
                SELECT
                    *,
                    DATE_DIFF('day', encounter_date, next_encounter_date) AS days_to_next,
                    CASE WHEN diagnosis_value = next_diagnosis_value THEN 1 ELSE 0 END AS same_diagnosis_transition,
                    CASE WHEN department_specialty = next_department_specialty THEN 1 ELSE 0 END AS same_specialty_transition,
                    CASE WHEN department_key != next_department_key THEN 1 ELSE 0 END AS changed_department_key,
                    CASE WHEN department_type != next_department_type THEN 1 ELSE 0 END AS changed_department_type
                FROM ordered
                WHERE next_encounter_date IS NOT NULL
            )

            SELECT
                department_type AS origin_department_type,
                next_department_type,
                department_specialty AS origin_department_specialty,
                next_department_specialty,
                COUNT(*) AS transition_count,
                ROUND(MEDIAN(days_to_next), 2) AS median_days_to_next,
                ROUND(AVG(same_diagnosis_transition) * 100, 2) AS pct_same_diagnosis,
                ROUND(AVG(same_specialty_transition) * 100, 2) AS pct_same_specialty,
                ROUND(AVG(changed_department_key) * 100, 2) AS pct_changed_department_key,
                ROUND(AVG(changed_department_type) * 100, 2) AS pct_changed_department_type
            FROM transitions
            GROUP BY
                origin_department_type,
                next_department_type,
                origin_department_specialty,
                next_department_specialty
        )
        TO '{DEPT_TRANSITIONS_PATH}'
        (FORMAT PARQUET)
        """)

        # ------------------------------------------------------------
        # Department metrics
        # ------------------------------------------------------------
        con.sql(f"""
        COPY (
            WITH dept_base AS (
                SELECT
                    enc_PatientDurableKey AS patient_id,
                    enc_EncounterKey AS encounter_key,
                    COALESCE(CAST({dept_type} AS VARCHAR), 'Unknown') AS department_type,
                    COALESCE(CAST({dept_specialty} AS VARCHAR), 'Unknown') AS department_specialty,
                    {is_ed} AS is_ed_visit,
                    {is_hosp} AS is_hospital_admission,
                    {is_inpatient} AS is_inpatient_admission,
                    {is_obs} AS is_observation,
                    {is_outpatient} AS is_outpatient_face_to_face,
                    {los} AS los_hours
                FROM '{MASTER_PATH}'
                WHERE enc_PatientDurableKey IS NOT NULL
            )

            SELECT
                d.department_type,
                d.department_specialty,
                COUNT(*) AS encounters,
                COUNT(DISTINCT d.patient_id) AS patients,
                ROUND(AVG(pc.journey_complexity_score), 2) AS avg_patient_complexity_score,
                ROUND(MEDIAN(pc.journey_complexity_score), 2) AS median_patient_complexity_score,
                ROUND(AVG(CASE WHEN pc.complexity_bucket IN ('High', 'Extreme') THEN 1 ELSE 0 END) * 100, 2) AS pct_high_or_extreme_complexity,
                SUM(d.is_ed_visit) AS ed_visits,
                SUM(d.is_hospital_admission) AS hospital_admissions,
                SUM(d.is_inpatient_admission) AS inpatient_admissions,
                SUM(d.is_observation) AS observation_visits,
                SUM(d.is_outpatient_face_to_face) AS outpatient_face_to_face_visits,
                ROUND(AVG(d.los_hours), 2) AS avg_los_hours,
                ROUND(MAX(d.los_hours), 2) AS max_los_hours
            FROM dept_base d
            LEFT JOIN '{PATIENT_COMPLEXITY_PATH}' pc
                ON d.patient_id = pc.patient_id
            GROUP BY d.department_type, d.department_specialty
        )
        TO '{DEPT_METRICS_PATH}'
        (FORMAT PARQUET)
        """)

        # ------------------------------------------------------------
        # Monthly overview
        # ------------------------------------------------------------
        con.sql(f"""
        COPY (
            SELECT
                DATE_TRUNC('month', TRY_CAST(enc_Date AS DATE)) AS month,
                COALESCE(CAST({dept_type} AS VARCHAR), 'Unknown') AS department_type,
                COUNT(*) AS encounters,
                COUNT(DISTINCT enc_PatientDurableKey) AS patients,
                SUM({is_ed}) AS ed_visits,
                SUM({is_hosp}) AS hospital_admissions,
                SUM({is_inpatient}) AS inpatient_admissions,
                SUM({is_obs}) AS observation_visits,
                SUM({is_outpatient}) AS outpatient_face_to_face_visits
            FROM '{MASTER_PATH}'
            WHERE TRY_CAST(enc_Date AS DATE) IS NOT NULL
            GROUP BY month, department_type
        )
        TO '{MONTHLY_OVERVIEW_PATH}'
        (FORMAT PARQUET)
        """)

        # ------------------------------------------------------------
        # SDOH metrics
        # ------------------------------------------------------------
        con.sql(f"""
        COPY (
            SELECT
                COALESCE(CAST({dept_type} AS VARCHAR), 'Unknown') AS department_type,
                COALESCE(CAST({dx_group} AS VARCHAR), 'Unknown') AS diagnosis_group,
                COUNT(*) AS encounters,
                COUNT(DISTINCT enc_PatientDurableKey) AS patients,
                SUM({sd_expr}) AS screened_encounters,
                ROUND(AVG({sd_expr}) * 100, 2) AS pct_encounters_screened
            FROM '{MASTER_PATH}'
            WHERE enc_PatientDurableKey IS NOT NULL
            GROUP BY department_type, diagnosis_group
        )
        TO '{SDOH_METRICS_PATH}'
        (FORMAT PARQUET)
        """)

    finally:
        con.close()







# ============================================================
# Choropleth / GeoJSON helper functions
# ============================================================

def find_geojson_file():
    """
    Find Census Block Group GeoJSON if available.
    A true choropleth requires polygon boundaries.
    """
    search_dirs = [
        Path("."),
        Path(".."),
        DATA_DIR,
        OUTPUT_DIR,
        DASHBOARD_DIR,
        Path("../data"),
        Path("../outputs"),
        Path("../../data"),
        Path("../../outputs"),
    ]

    patterns = [
        "*.geojson",
        "*.json",
        "*block*group*.geojson",
        "*block*group*.json",
        "*cbg*.geojson",
        "*cbg*.json",
        "*census*.geojson",
        "*census*.json",
        "*tiger*.geojson",
        "*tiger*.json",
    ]

    candidates = []

    for folder in search_dirs:
        if folder.exists():
            for pat in patterns:
                candidates.extend(list(folder.glob(pat)))
            try:
                candidates.extend(list(folder.rglob("*block*group*.geojson")))
                candidates.extend(list(folder.rglob("*cbg*.geojson")))
                candidates.extend(list(folder.rglob("*census*.geojson")))
                candidates.extend(list(folder.rglob("*tiger*.geojson")))
            except Exception:
                pass

    candidates = list(dict.fromkeys([p for p in candidates if p.exists()]))

    if not candidates:
        return None

    def score_file(p):
        name = p.name.lower()
        score = 0
        if "block" in name and "group" in name:
            score += 80
        if "cbg" in name:
            score += 70
        if "census" in name:
            score += 40
        if "tiger" in name:
            score += 40
        if p.suffix.lower() == ".geojson":
            score += 20
        return score

    candidates = sorted(candidates, key=score_file, reverse=True)
    return candidates[0]


@st.cache_data(show_spinner=False)
def load_geojson_boundaries():
    """
    Load GeoJSON boundaries if available.
    The feature id must be mappable to GEOID.
    """
    geojson_path = find_geojson_file()

    if geojson_path is None:
        return None, None, "No GeoJSON boundary file found."

    try:
        with open(geojson_path, "r") as f:
            gj = json.load(f)
    except Exception as e:
        return None, geojson_path, f"Could not read GeoJSON file: {e}"

    # Try to normalize GEOID property names.
    try:
        for feat in gj.get("features", []):
            props = feat.get("properties", {})
            found = None

            for key in props.keys():
                key_norm = normalize_colname_for_match(key)
                if key_norm in [
                    "geoid",
                    "geoid20",
                    "geoid2020",
                    "censusblockgroupfipscode",
                    "blockgroupgeoid",
                ] or "geoid" in key_norm or "fips" in key_norm:
                    found = key
                    break

            if found is not None:
                props["GEOID_CLEAN"] = clean_geoid_value(props.get(found))
            elif "id" in feat:
                props["GEOID_CLEAN"] = clean_geoid_value(feat.get("id"))
            else:
                props["GEOID_CLEAN"] = None

    except Exception as e:
        return None, geojson_path, f"Could not normalize GeoJSON GEOID properties: {e}"

    return gj, geojson_path, None


def fix_us_longitude_if_needed(df, lon_col="CENTLONG", lat_col="CENTLAT"):
    """
    Kansas longitudes should be negative.
    If lat looks like Kansas/US but longitude is positive, flip it.
    """
    if df is None or df.empty or lon_col not in df.columns or lat_col not in df.columns:
        return df

    df = df.copy()

    median_lat = pd.to_numeric(df[lat_col], errors="coerce").median()
    median_lon = pd.to_numeric(df[lon_col], errors="coerce").median()

    # Kansas-ish latitude, but longitude positive means sign likely got lost.
    if pd.notna(median_lat) and pd.notna(median_lon):
        if 35 <= median_lat <= 42 and median_lon > 0:
            df[lon_col] = -1 * pd.to_numeric(df[lon_col], errors="coerce").abs()

    return df


def get_kansas_center(df=None):
    """
    Return a reasonable Kansas/Topeka center.
    If valid geography data exists, use mean of visible points.
    """
    default_center = {"lat": 39.0473, "lon": -95.6752}  # Topeka-ish

    if df is None or df.empty:
        return default_center

    if "CENTLAT" not in df.columns or "CENTLONG" not in df.columns:
        return default_center

    lat = pd.to_numeric(df["CENTLAT"], errors="coerce")
    lon = pd.to_numeric(df["CENTLONG"], errors="coerce")

    lat_mean = lat.mean()
    lon_mean = lon.mean()

    if pd.isna(lat_mean) or pd.isna(lon_mean):
        return default_center

    # If center is clearly not Kansas / surrounding region, use Topeka fallback.
    if not (35 <= lat_mean <= 42 and -103 <= lon_mean <= -94):
        return default_center

    return {"lat": float(lat_mean), "lon": float(lon_mean)}


# ============================================================
# Geography helper functions
# These MUST be defined before the SDOH & Geography page runs.
# ============================================================

def clean_geoid_value(x):
    """
    Convert GEOID / CensusBlockGroupFipsCode to comparable string.
    Handles values loaded as int, float, or string.
    """
    if pd.isna(x):
        return pd.NA

    s = str(x).strip()

    if s.endswith(".0"):
        s = s[:-2]

    s = s.replace(" ", "").replace("-", "")

    if s.lower() in UNKNOWN_VALUES:
        return pd.NA

    return s


def normalize_colname_for_match(c):
    """
    Make column matching robust:
    'CENT LONG', 'cent_lon', 'CENTLON', 'CENTLONG' all become comparable.
    """
    return (
        str(c)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
    )


def find_tiger_file():
    """
    Find tigercensuscodes file in ../data, ../outputs, current folder,
    parent folders, or dashboard folder.
    Supports csv/parquet with names containing tiger/census/geoid.
    """
    search_dirs = [
        Path("."),
        Path(".."),
        DATA_DIR,
        OUTPUT_DIR,
        DASHBOARD_DIR,
        Path("../outputs"),
        Path("../data"),
        Path("../../data"),
        Path("../../outputs"),
    ]

    candidates = []

    for folder in search_dirs:
        if folder.exists():
            patterns = [
                "*tigercensus*.parquet",
                "*tiger*census*.parquet",
                "*tiger*.parquet",
                "*census*.parquet",
                "*geoid*.parquet",
                "*tigercensus*.csv",
                "*tiger*census*.csv",
                "*tiger*.csv",
                "*census*.csv",
                "*geoid*.csv",
            ]

            for pat in patterns:
                candidates.extend(list(folder.glob(pat)))
                candidates.extend(list(folder.glob(pat.upper())))

    # Also one-level recursive search in data/output if available
    for folder in [DATA_DIR, OUTPUT_DIR, Path("../data"), Path("../outputs")]:
        if folder.exists():
            try:
                candidates.extend(list(folder.rglob("*tiger*.parquet")))
                candidates.extend(list(folder.rglob("*census*.parquet")))
                candidates.extend(list(folder.rglob("*geoid*.parquet")))
                candidates.extend(list(folder.rglob("*tiger*.csv")))
                candidates.extend(list(folder.rglob("*census*.csv")))
                candidates.extend(list(folder.rglob("*geoid*.csv")))
            except Exception:
                pass

    # Deduplicate
    candidates = list(dict.fromkeys([p for p in candidates if p.exists()]))

    if not candidates:
        return None

    # Prefer exact-looking tigercensuscodes files
    def score_file(p):
        name = p.name.lower()
        score = 0
        if "tigercensuscodes" in name:
            score += 100
        if "tiger" in name:
            score += 50
        if "census" in name:
            score += 30
        if "geoid" in name:
            score += 20
        if p.suffix.lower() == ".parquet":
            score += 5
        return score

    candidates = sorted(candidates, key=score_file, reverse=True)
    return candidates[0]


@st.cache_data(show_spinner=False)
def load_tiger_codes():
    tiger_path = find_tiger_file()

    if tiger_path is None:
        return pd.DataFrame(), None

    try:
        if tiger_path.suffix.lower() == ".parquet":
            tiger = pd.read_parquet(tiger_path)
        else:
            # dtype=str preserves GEOID formatting
            tiger = pd.read_csv(tiger_path, dtype=str)
    except Exception:
        return pd.DataFrame(), tiger_path

    # Robust column matching
    col_lookup = {normalize_colname_for_match(c): c for c in tiger.columns}

    geoid_keys = [
        "geoid",
        "geoid20",
        "geoid2020",
        "censusblockgroupfipscode",
        "blockgroup",
        "blockgroupgeoid",
    ]

    lat_keys = [
        "centlat",
        "centerlat",
        "centerlatitude",
        "lat",
        "latitude",
        "centroidlat",
        "centroidlatitude",
    ]

    lon_keys = [
        "centlong",
        "centlon",       # IMPORTANT: your file may use CENTLON
        "centerlong",
        "centerlon",
        "centerlongitude",
        "lon",
        "long",
        "longitude",
        "centroidlong",
        "centroidlon",
        "centroidlongitude",
    ]

    pop_keys = [
        "population",
        "pop",
        "pop2020",
        "population2020",
    ]

    def find_col(keys):
        for k in keys:
            if k in col_lookup:
                return col_lookup[k]
        return None

    geoid_col = find_col(geoid_keys)
    lat_col = find_col(lat_keys)
    lon_col = find_col(lon_keys)
    pop_col = find_col(pop_keys)

    # Fallback: substring matching
    if geoid_col is None:
        for norm, orig in col_lookup.items():
            if "geoid" in norm or "fips" in norm:
                geoid_col = orig
                break

    if lat_col is None:
        for norm, orig in col_lookup.items():
            if "lat" in norm:
                lat_col = orig
                break

    if lon_col is None:
        for norm, orig in col_lookup.items():
            if "lon" in norm or "long" in norm:
                lon_col = orig
                break

    if geoid_col is None or lat_col is None or lon_col is None:
        # Store columns for debugging in Streamlit session if possible
        try:
            st.session_state["tiger_debug_path"] = str(tiger_path)
            st.session_state["tiger_debug_columns"] = list(tiger.columns)
        except Exception:
            pass
        return pd.DataFrame(), tiger_path

    tiger = tiger.rename(columns={
        geoid_col: "GEOID",
        lat_col: "CENTLAT",
        lon_col: "CENTLONG",
    })

    if pop_col is not None:
        tiger = tiger.rename(columns={pop_col: "Population"})
    else:
        tiger["Population"] = np.nan

    tiger["geoid_clean"] = tiger["GEOID"].map(clean_geoid_value)
    tiger["CENTLAT"] = pd.to_numeric(tiger["CENTLAT"], errors="coerce")
    tiger["CENTLONG"] = pd.to_numeric(tiger["CENTLONG"], errors="coerce")
    tiger["Population"] = pd.to_numeric(tiger["Population"], errors="coerce")

    tiger = tiger.dropna(subset=["geoid_clean", "CENTLAT", "CENTLONG"])

    # Drop invalid coordinate rows
    tiger = tiger[
        tiger["CENTLAT"].between(-90, 90)
        & tiger["CENTLONG"].between(-180, 180)
    ].copy()

    tiger = fix_us_longitude_if_needed(tiger, lon_col="CENTLONG", lat_col="CENTLAT")

    return tiger, tiger_path


@st.cache_data(show_spinner=True)
def load_geography_map_data():
    """
    Build patient geography map data by joining patient CensusBlockGroupFipsCode
    to TIGER census centroid coordinates.
    """
    cols = get_columns(str(MASTER_PATH))

    if "pat_CensusBlockGroupFipsCode" not in cols:
        return pd.DataFrame(), "Missing pat_CensusBlockGroupFipsCode in master_encounters.parquet.", None

    tiger, tiger_path = load_tiger_codes()

    if tiger.empty:
        debug_cols = st.session_state.get("tiger_debug_columns", None)
        debug_path = st.session_state.get("tiger_debug_path", None)

        if debug_cols is not None:
            msg = (
                "Found TIGER/census file but could not identify GEOID, latitude, and longitude columns. "
                f"File: {debug_path}. Columns found: {debug_cols}. "
                "Expected something like GEOID, CENTLAT, CENTLON/CENTLONG."
            )
        else:
            msg = (
                "Could not find/read tigercensuscodes file with GEOID, CENTLAT, and CENTLON/CENTLONG. "
                "Place the file in ../data or ../outputs."
            )

        return pd.DataFrame(), msg, tiger_path

    is_ed = bool_flag_expr(cols, "enc_IsEdVisit")
    is_hosp = bool_flag_expr(cols, "enc_IsHospitalAdmission")
    is_inpatient = bool_flag_expr(cols, "enc_IsInpatientAdmission")
    deceased = deceased_expr(cols)

    geo_raw = run_query(f"""
        WITH base AS (
            SELECT
                CAST(pat_CensusBlockGroupFipsCode AS VARCHAR) AS geoid_raw,
                enc_PatientDurableKey AS patient_id,
                enc_EncounterKey AS encounter_key,
                {is_ed} AS is_ed_visit,
                {is_hosp} AS is_hospital_admission,
                {is_inpatient} AS is_inpatient_admission,
                {deceased} AS is_deceased
            FROM '{MASTER_PATH}'
            WHERE pat_CensusBlockGroupFipsCode IS NOT NULL
              AND enc_PatientDurableKey IS NOT NULL
        )

        SELECT
            geoid_raw,
            COUNT(DISTINCT patient_id) AS patients,
            COUNT(DISTINCT encounter_key) AS encounters,
            SUM(is_ed_visit) AS ed_visits,
            SUM(is_hospital_admission) AS hospital_admissions,
            SUM(is_inpatient_admission) AS inpatient_admissions,
            MAX(is_deceased) AS any_deceased_flag
        FROM base
        GROUP BY geoid_raw
    """)

    if geo_raw.empty:
        return pd.DataFrame(), "No geography rows found from patient CensusBlockGroupFipsCode.", tiger_path

    geo_raw["geoid_clean"] = geo_raw["geoid_raw"].map(clean_geoid_value)
    geo_raw = geo_raw.dropna(subset=["geoid_clean"])

    try:
        pc_geo = run_query(f"""
            WITH patient_geo AS (
                SELECT DISTINCT
                    enc_PatientDurableKey AS patient_id,
                    CAST(pat_CensusBlockGroupFipsCode AS VARCHAR) AS geoid_raw
                FROM '{MASTER_PATH}'
                WHERE pat_CensusBlockGroupFipsCode IS NOT NULL
                  AND enc_PatientDurableKey IS NOT NULL
            )

            SELECT
                patient_geo.geoid_raw,
                ROUND(AVG(pc.journey_complexity_score), 2) AS avg_complexity_score,
                ROUND(AVG(CASE WHEN pc.complexity_bucket IN ('High', 'Extreme') THEN 1 ELSE 0 END) * 100, 2) AS pct_high_or_extreme_complexity
            FROM patient_geo
            LEFT JOIN '{PATIENT_COMPLEXITY_PATH}' pc
                ON patient_geo.patient_id = pc.patient_id
            GROUP BY patient_geo.geoid_raw
        """)

        pc_geo["geoid_clean"] = pc_geo["geoid_raw"].map(clean_geoid_value)

        geo_raw = geo_raw.merge(
            pc_geo[["geoid_clean", "avg_complexity_score", "pct_high_or_extreme_complexity"]],
            on="geoid_clean",
            how="left"
        )

    except Exception:
        geo_raw["avg_complexity_score"] = np.nan
        geo_raw["pct_high_or_extreme_complexity"] = np.nan

    geo = geo_raw.merge(
        tiger[["geoid_clean", "GEOID", "CENTLAT", "CENTLONG", "Population"]],
        on="geoid_clean",
        how="inner"
    )

    geo["encounters_per_patient"] = geo["encounters"] / geo["patients"].replace(0, np.nan)
    geo["ed_visits_per_100_patients"] = geo["ed_visits"] * 100 / geo["patients"].replace(0, np.nan)
    geo["admissions_per_100_patients"] = geo["hospital_admissions"] * 100 / geo["patients"].replace(0, np.nan)
    geo["patients_per_1000_population"] = geo["patients"] * 1000 / geo["Population"].replace(0, np.nan)

    geo = fix_us_longitude_if_needed(geo, lon_col="CENTLONG", lat_col="CENTLAT")

    return geo, None, tiger_path


# ============================================================
# Sidebar navigation styling
# ============================================================

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 45%, #1e293b 100%);
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.06);
        color: #f8fafc;
        padding: 0.85rem 1rem;
        font-weight: 750;
        text-align: left;
        transition: all 0.22s ease-in-out;
        box-shadow: 0 8px 22px rgba(0,0,0,0.12);
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateX(6px) scale(1.015);
        background: rgba(59,130,246,0.28);
        border-color: rgba(147,197,253,0.7);
        box-shadow: 0 12px 28px rgba(59,130,246,0.22);
    }

    section[data-testid="stSidebar"] .stButton > button:active {
        transform: translateX(4px) scale(0.995);
        background: rgba(16,185,129,0.28);
    }

    .nav-title {
        font-size: 1.05rem;
        font-weight: 850;
        color: #ffffff;
        margin-top: 0.4rem;
        margin-bottom: 0.3rem;
    }

    .nav-subtitle {
        font-size: 0.82rem;
        color: #cbd5e1;
        margin-bottom: 1.0rem;
        line-height: 1.25rem;
    }

    .active-page-card {
        background: linear-gradient(135deg, rgba(37,99,235,0.42), rgba(16,185,129,0.26));
        border: 1px solid rgba(191,219,254,0.45);
        padding: 0.8rem 0.9rem;
        border-radius: 16px;
        margin: 0.8rem 0rem 1rem 0rem;
        box-shadow: 0 10px 28px rgba(37,99,235,0.20);
    }

    .active-page-label {
        font-size: 0.72rem;
        color: #bfdbfe;
        text-transform: uppercase;
        font-weight: 750;
        letter-spacing: 0.08em;
    }

    .active-page-name {
        font-size: 1.0rem;
        color: #ffffff;
        font-weight: 850;
        margin-top: 0.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)






# ============================================================
# Choropleth / GeoJSON helper functions
# ============================================================

def find_geojson_file():
    """
    Find Census Block Group GeoJSON if available.
    A true choropleth requires polygon boundaries.
    """
    search_dirs = [
        Path("."),
        Path(".."),
        DATA_DIR,
        OUTPUT_DIR,
        DASHBOARD_DIR,
        Path("../data"),
        Path("../outputs"),
        Path("../../data"),
        Path("../../outputs"),
    ]

    patterns = [
        "*.geojson",
        "*.json",
        "*block*group*.geojson",
        "*block*group*.json",
        "*cbg*.geojson",
        "*cbg*.json",
        "*census*.geojson",
        "*census*.json",
        "*tiger*.geojson",
        "*tiger*.json",
    ]

    candidates = []

    for folder in search_dirs:
        if folder.exists():
            for pat in patterns:
                candidates.extend(list(folder.glob(pat)))
            try:
                candidates.extend(list(folder.rglob("*block*group*.geojson")))
                candidates.extend(list(folder.rglob("*cbg*.geojson")))
                candidates.extend(list(folder.rglob("*census*.geojson")))
                candidates.extend(list(folder.rglob("*tiger*.geojson")))
            except Exception:
                pass

    candidates = list(dict.fromkeys([p for p in candidates if p.exists()]))

    if not candidates:
        return None

    def score_file(p):
        name = p.name.lower()
        score = 0
        if "block" in name and "group" in name:
            score += 80
        if "cbg" in name:
            score += 70
        if "census" in name:
            score += 40
        if "tiger" in name:
            score += 40
        if p.suffix.lower() == ".geojson":
            score += 20
        return score

    candidates = sorted(candidates, key=score_file, reverse=True)
    return candidates[0]


@st.cache_data(show_spinner=False)
def load_geojson_boundaries():
    """
    Load GeoJSON boundaries if available.
    The feature id must be mappable to GEOID.
    """
    geojson_path = find_geojson_file()

    if geojson_path is None:
        return None, None, "No GeoJSON boundary file found."

    try:
        with open(geojson_path, "r") as f:
            gj = json.load(f)
    except Exception as e:
        return None, geojson_path, f"Could not read GeoJSON file: {e}"

    # Try to normalize GEOID property names.
    try:
        for feat in gj.get("features", []):
            props = feat.get("properties", {})
            found = None

            for key in props.keys():
                key_norm = normalize_colname_for_match(key)
                if key_norm in [
                    "geoid",
                    "geoid20",
                    "geoid2020",
                    "censusblockgroupfipscode",
                    "blockgroupgeoid",
                ] or "geoid" in key_norm or "fips" in key_norm:
                    found = key
                    break

            if found is not None:
                props["GEOID_CLEAN"] = clean_geoid_value(props.get(found))
            elif "id" in feat:
                props["GEOID_CLEAN"] = clean_geoid_value(feat.get("id"))
            else:
                props["GEOID_CLEAN"] = None

    except Exception as e:
        return None, geojson_path, f"Could not normalize GeoJSON GEOID properties: {e}"

    return gj, geojson_path, None


def fix_us_longitude_if_needed(df, lon_col="CENTLONG", lat_col="CENTLAT"):
    """
    Kansas longitudes should be negative.
    If lat looks like Kansas/US but longitude is positive, flip it.
    """
    if df is None or df.empty or lon_col not in df.columns or lat_col not in df.columns:
        return df

    df = df.copy()

    median_lat = pd.to_numeric(df[lat_col], errors="coerce").median()
    median_lon = pd.to_numeric(df[lon_col], errors="coerce").median()

    # Kansas-ish latitude, but longitude positive means sign likely got lost.
    if pd.notna(median_lat) and pd.notna(median_lon):
        if 35 <= median_lat <= 42 and median_lon > 0:
            df[lon_col] = -1 * pd.to_numeric(df[lon_col], errors="coerce").abs()

    return df


def get_kansas_center(df=None):
    """
    Return a reasonable Kansas/Topeka center.
    If valid geography data exists, use mean of visible points.
    """
    default_center = {"lat": 39.0473, "lon": -95.6752}  # Topeka-ish

    if df is None or df.empty:
        return default_center

    if "CENTLAT" not in df.columns or "CENTLONG" not in df.columns:
        return default_center

    lat = pd.to_numeric(df["CENTLAT"], errors="coerce")
    lon = pd.to_numeric(df["CENTLONG"], errors="coerce")

    lat_mean = lat.mean()
    lon_mean = lon.mean()

    if pd.isna(lat_mean) or pd.isna(lon_mean):
        return default_center

    # If center is clearly not Kansas / surrounding region, use Topeka fallback.
    if not (35 <= lat_mean <= 42 and -103 <= lon_mean <= -94):
        return default_center

    return {"lat": float(lat_mean), "lon": float(lon_mean)}


# ============================================================
# Geography helper functions
# These MUST be defined before the SDOH & Geography page runs.
# ============================================================

def clean_geoid_value(x):
    """
    Convert GEOID / CensusBlockGroupFipsCode to comparable string.
    Handles values loaded as int, float, or string.
    """
    if pd.isna(x):
        return pd.NA

    s = str(x).strip()

    if s.endswith(".0"):
        s = s[:-2]

    s = s.replace(" ", "").replace("-", "")

    if s.lower() in UNKNOWN_VALUES:
        return pd.NA

    return s


def normalize_colname_for_match(c):
    """
    Make column matching robust:
    'CENT LONG', 'cent_lon', 'CENTLON', 'CENTLONG' all become comparable.
    """
    return (
        str(c)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
    )


def find_tiger_file():
    """
    Find tigercensuscodes file in ../data, ../outputs, current folder,
    parent folders, or dashboard folder.
    Supports csv/parquet with names containing tiger/census/geoid.
    """
    search_dirs = [
        Path("."),
        Path(".."),
        DATA_DIR,
        OUTPUT_DIR,
        DASHBOARD_DIR,
        Path("../outputs"),
        Path("../data"),
        Path("../../data"),
        Path("../../outputs"),
    ]

    candidates = []

    for folder in search_dirs:
        if folder.exists():
            patterns = [
                "*tigercensus*.parquet",
                "*tiger*census*.parquet",
                "*tiger*.parquet",
                "*census*.parquet",
                "*geoid*.parquet",
                "*tigercensus*.csv",
                "*tiger*census*.csv",
                "*tiger*.csv",
                "*census*.csv",
                "*geoid*.csv",
            ]

            for pat in patterns:
                candidates.extend(list(folder.glob(pat)))
                candidates.extend(list(folder.glob(pat.upper())))

    # Also one-level recursive search in data/output if available
    for folder in [DATA_DIR, OUTPUT_DIR, Path("../data"), Path("../outputs")]:
        if folder.exists():
            try:
                candidates.extend(list(folder.rglob("*tiger*.parquet")))
                candidates.extend(list(folder.rglob("*census*.parquet")))
                candidates.extend(list(folder.rglob("*geoid*.parquet")))
                candidates.extend(list(folder.rglob("*tiger*.csv")))
                candidates.extend(list(folder.rglob("*census*.csv")))
                candidates.extend(list(folder.rglob("*geoid*.csv")))
            except Exception:
                pass

    # Deduplicate
    candidates = list(dict.fromkeys([p for p in candidates if p.exists()]))

    if not candidates:
        return None

    # Prefer exact-looking tigercensuscodes files
    def score_file(p):
        name = p.name.lower()
        score = 0
        if "tigercensuscodes" in name:
            score += 100
        if "tiger" in name:
            score += 50
        if "census" in name:
            score += 30
        if "geoid" in name:
            score += 20
        if p.suffix.lower() == ".parquet":
            score += 5
        return score

    candidates = sorted(candidates, key=score_file, reverse=True)
    return candidates[0]


@st.cache_data(show_spinner=False)
def load_tiger_codes():
    tiger_path = find_tiger_file()

    if tiger_path is None:
        return pd.DataFrame(), None

    try:
        if tiger_path.suffix.lower() == ".parquet":
            tiger = pd.read_parquet(tiger_path)
        else:
            # dtype=str preserves GEOID formatting
            tiger = pd.read_csv(tiger_path, dtype=str)
    except Exception:
        return pd.DataFrame(), tiger_path

    # Robust column matching
    col_lookup = {normalize_colname_for_match(c): c for c in tiger.columns}

    geoid_keys = [
        "geoid",
        "geoid20",
        "geoid2020",
        "censusblockgroupfipscode",
        "blockgroup",
        "blockgroupgeoid",
    ]

    lat_keys = [
        "centlat",
        "centerlat",
        "centerlatitude",
        "lat",
        "latitude",
        "centroidlat",
        "centroidlatitude",
    ]

    lon_keys = [
        "centlong",
        "centlon",       # IMPORTANT: your file may use CENTLON
        "centerlong",
        "centerlon",
        "centerlongitude",
        "lon",
        "long",
        "longitude",
        "centroidlong",
        "centroidlon",
        "centroidlongitude",
    ]

    pop_keys = [
        "population",
        "pop",
        "pop2020",
        "population2020",
    ]

    def find_col(keys):
        for k in keys:
            if k in col_lookup:
                return col_lookup[k]
        return None

    geoid_col = find_col(geoid_keys)
    lat_col = find_col(lat_keys)
    lon_col = find_col(lon_keys)
    pop_col = find_col(pop_keys)

    # Fallback: substring matching
    if geoid_col is None:
        for norm, orig in col_lookup.items():
            if "geoid" in norm or "fips" in norm:
                geoid_col = orig
                break

    if lat_col is None:
        for norm, orig in col_lookup.items():
            if "lat" in norm:
                lat_col = orig
                break

    if lon_col is None:
        for norm, orig in col_lookup.items():
            if "lon" in norm or "long" in norm:
                lon_col = orig
                break

    if geoid_col is None or lat_col is None or lon_col is None:
        # Store columns for debugging in Streamlit session if possible
        try:
            st.session_state["tiger_debug_path"] = str(tiger_path)
            st.session_state["tiger_debug_columns"] = list(tiger.columns)
        except Exception:
            pass
        return pd.DataFrame(), tiger_path

    tiger = tiger.rename(columns={
        geoid_col: "GEOID",
        lat_col: "CENTLAT",
        lon_col: "CENTLONG",
    })

    if pop_col is not None:
        tiger = tiger.rename(columns={pop_col: "Population"})
    else:
        tiger["Population"] = np.nan

    tiger["geoid_clean"] = tiger["GEOID"].map(clean_geoid_value)
    tiger["CENTLAT"] = pd.to_numeric(tiger["CENTLAT"], errors="coerce")
    tiger["CENTLONG"] = pd.to_numeric(tiger["CENTLONG"], errors="coerce")
    tiger["Population"] = pd.to_numeric(tiger["Population"], errors="coerce")

    tiger = tiger.dropna(subset=["geoid_clean", "CENTLAT", "CENTLONG"])

    # Drop invalid coordinate rows
    tiger = tiger[
        tiger["CENTLAT"].between(-90, 90)
        & tiger["CENTLONG"].between(-180, 180)
    ].copy()

    tiger = fix_us_longitude_if_needed(tiger, lon_col="CENTLONG", lat_col="CENTLAT")

    return tiger, tiger_path


@st.cache_data(show_spinner=True)
def load_geography_map_data():
    """
    Build patient geography map data by joining patient CensusBlockGroupFipsCode
    to TIGER census centroid coordinates.
    """
    cols = get_columns(str(MASTER_PATH))

    if "pat_CensusBlockGroupFipsCode" not in cols:
        return pd.DataFrame(), "Missing pat_CensusBlockGroupFipsCode in master_encounters.parquet.", None

    tiger, tiger_path = load_tiger_codes()

    if tiger.empty:
        debug_cols = st.session_state.get("tiger_debug_columns", None)
        debug_path = st.session_state.get("tiger_debug_path", None)

        if debug_cols is not None:
            msg = (
                "Found TIGER/census file but could not identify GEOID, latitude, and longitude columns. "
                f"File: {debug_path}. Columns found: {debug_cols}. "
                "Expected something like GEOID, CENTLAT, CENTLON/CENTLONG."
            )
        else:
            msg = (
                "Could not find/read tigercensuscodes file with GEOID, CENTLAT, and CENTLON/CENTLONG. "
                "Place the file in ../data or ../outputs."
            )

        return pd.DataFrame(), msg, tiger_path

    is_ed = bool_flag_expr(cols, "enc_IsEdVisit")
    is_hosp = bool_flag_expr(cols, "enc_IsHospitalAdmission")
    is_inpatient = bool_flag_expr(cols, "enc_IsInpatientAdmission")
    deceased = deceased_expr(cols)

    geo_raw = run_query(f"""
        WITH base AS (
            SELECT
                CAST(pat_CensusBlockGroupFipsCode AS VARCHAR) AS geoid_raw,
                enc_PatientDurableKey AS patient_id,
                enc_EncounterKey AS encounter_key,
                {is_ed} AS is_ed_visit,
                {is_hosp} AS is_hospital_admission,
                {is_inpatient} AS is_inpatient_admission,
                {deceased} AS is_deceased
            FROM '{MASTER_PATH}'
            WHERE pat_CensusBlockGroupFipsCode IS NOT NULL
              AND enc_PatientDurableKey IS NOT NULL
        )

        SELECT
            geoid_raw,
            COUNT(DISTINCT patient_id) AS patients,
            COUNT(DISTINCT encounter_key) AS encounters,
            SUM(is_ed_visit) AS ed_visits,
            SUM(is_hospital_admission) AS hospital_admissions,
            SUM(is_inpatient_admission) AS inpatient_admissions,
            MAX(is_deceased) AS any_deceased_flag
        FROM base
        GROUP BY geoid_raw
    """)

    if geo_raw.empty:
        return pd.DataFrame(), "No geography rows found from patient CensusBlockGroupFipsCode.", tiger_path

    geo_raw["geoid_clean"] = geo_raw["geoid_raw"].map(clean_geoid_value)
    geo_raw = geo_raw.dropna(subset=["geoid_clean"])

    try:
        pc_geo = run_query(f"""
            WITH patient_geo AS (
                SELECT DISTINCT
                    enc_PatientDurableKey AS patient_id,
                    CAST(pat_CensusBlockGroupFipsCode AS VARCHAR) AS geoid_raw
                FROM '{MASTER_PATH}'
                WHERE pat_CensusBlockGroupFipsCode IS NOT NULL
                  AND enc_PatientDurableKey IS NOT NULL
            )

            SELECT
                patient_geo.geoid_raw,
                ROUND(AVG(pc.journey_complexity_score), 2) AS avg_complexity_score,
                ROUND(AVG(CASE WHEN pc.complexity_bucket IN ('High', 'Extreme') THEN 1 ELSE 0 END) * 100, 2) AS pct_high_or_extreme_complexity
            FROM patient_geo
            LEFT JOIN '{PATIENT_COMPLEXITY_PATH}' pc
                ON patient_geo.patient_id = pc.patient_id
            GROUP BY patient_geo.geoid_raw
        """)

        pc_geo["geoid_clean"] = pc_geo["geoid_raw"].map(clean_geoid_value)

        geo_raw = geo_raw.merge(
            pc_geo[["geoid_clean", "avg_complexity_score", "pct_high_or_extreme_complexity"]],
            on="geoid_clean",
            how="left"
        )

    except Exception:
        geo_raw["avg_complexity_score"] = np.nan
        geo_raw["pct_high_or_extreme_complexity"] = np.nan

    geo = geo_raw.merge(
        tiger[["geoid_clean", "GEOID", "CENTLAT", "CENTLONG", "Population"]],
        on="geoid_clean",
        how="inner"
    )

    geo["encounters_per_patient"] = geo["encounters"] / geo["patients"].replace(0, np.nan)
    geo["ed_visits_per_100_patients"] = geo["ed_visits"] * 100 / geo["patients"].replace(0, np.nan)
    geo["admissions_per_100_patients"] = geo["hospital_admissions"] * 100 / geo["patients"].replace(0, np.nan)
    geo["patients_per_1000_population"] = geo["patients"] * 1000 / geo["Population"].replace(0, np.nan)

    return geo, None, tiger_path


# ============================================================
# Sidebar
# ============================================================

st.sidebar.markdown("## 🏥 SVH Dashboard Controls")

if not MASTER_PATH.exists():
    st.error(
        f"""
        Could not find `{MASTER_PATH}`.

        Make sure you run this app from the same project folder where:
        `../outputs/master_encounters.parquet` exists.
        """
    )
    st.stop()

rebuild_cache = st.sidebar.checkbox(
    "Rebuild dashboard cache",
    value=False,
    help="Use this if you changed the master file or want to regenerate dashboard parquet tables."
)

with st.spinner("Preparing dashboard data. First run may take a few minutes..."):
    create_dashboard_assets(rebuild=rebuild_cache)

st.sidebar.success("Dashboard data ready")

# Nice button-based navigation
PAGES = [
    ("Executive Overview", "📊"),
    ("Journey Complexity", "🧭"),
    ("Department Flow", "🔀"),
    ("Diagnosis Explorer", "🧬"),
    ("Edema Outreach", "🫀"),
    ("SDOH & Geography", "🗺️"),
]

if "active_page" not in st.session_state:
    st.session_state.active_page = "Executive Overview"

st.sidebar.markdown('<div class="nav-title">Navigation</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="nav-subtitle">Choose a dashboard module. Each page focuses on a different SVH operations lens.</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    f"""
    <div class="active-page-card">
        <div class="active-page-label">Current page</div>
        <div class="active-page-name">{st.session_state.active_page}</div>
    </div>
    """,
    unsafe_allow_html=True
)

for page_name, icon in PAGES:
    active = page_name == st.session_state.active_page
    label = f"{icon}  {page_name}"
    if st.sidebar.button(label, key=f"nav_{page_name}", use_container_width=True):
        st.session_state.active_page = page_name
        st.rerun()

page = st.session_state.active_page

# ============================================================
# Load common small dimension data
# ============================================================

@st.cache_data(show_spinner=False)
def load_patient_complexity():
    return clean_dashboard_df(run_query(f"SELECT * FROM '{PATIENT_COMPLEXITY_PATH}'"), 'patient_complexity')

@st.cache_data(show_spinner=False)
def load_patient_dx():
    return clean_dashboard_df(run_query(f"SELECT * FROM '{PATIENT_DX_JOURNEYS_PATH}'"), 'patient_dx')

@st.cache_data(show_spinner=False)
def load_dept_transitions():
    return clean_dashboard_df(run_query(f"SELECT * FROM '{DEPT_TRANSITIONS_PATH}'"), 'dept_transitions')

@st.cache_data(show_spinner=False)
def load_dept_metrics():
    return clean_dashboard_df(run_query(f"SELECT * FROM '{DEPT_METRICS_PATH}'"), 'dept_metrics')

@st.cache_data(show_spinner=False)
def load_dx_metrics():
    return clean_dashboard_df(run_query(f"SELECT * FROM '{DX_METRICS_PATH}'"), 'dx_metrics')

@st.cache_data(show_spinner=False)
def load_monthly():
    return clean_dashboard_df(run_query(f"SELECT * FROM '{MONTHLY_OVERVIEW_PATH}'"), 'monthly')

@st.cache_data(show_spinner=False)
def load_sdoh():
    return clean_dashboard_df(run_query(f"SELECT * FROM '{SDOH_METRICS_PATH}'"), 'sdoh')

pc = load_patient_complexity()
dx_metrics = load_dx_metrics()
dept_metrics = load_dept_metrics()
monthly = load_monthly()

# ============================================================
# Header
# ============================================================

st.markdown('<div class="dashboard-title">SVH Journey Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dashboard-subtitle">Patient flow, journey complexity, department handoffs, and early outreach opportunities.</div>',
    unsafe_allow_html=True
)

# ============================================================
# Page 1: Executive Overview
# ============================================================

if page == "Executive Overview":
    st.markdown('<div class="section-header">Executive Overview</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
        This page summarizes overall encounter volume, patient volume, care setting mix,
        and high-level utilization trends. It is meant for hospital operations review, not clinical diagnosis.
        </div>
        """,
        unsafe_allow_html=True
    )

    kpi = run_query(f"""
        SELECT
            COUNT(*) AS total_encounters,
            COUNT(DISTINCT patient_id) AS total_patients,
            SUM(ed_visits) AS ed_visits,
            SUM(hospital_admissions) AS hospital_admissions,
            SUM(inpatient_admissions) AS inpatient_admissions,
            SUM(observation_visits) AS observation_visits,
            SUM(outpatient_face_to_face_visits) AS outpatient_face_to_face_visits,
            ROUND(AVG(journey_complexity_score), 2) AS avg_complexity,
            ROUND(AVG(CASE WHEN complexity_bucket IN ('High', 'Extreme') THEN 1 ELSE 0 END) * 100, 2) AS pct_high_complexity
        FROM '{PATIENT_COMPLEXITY_PATH}'
    """).iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Patients", fmt_int(kpi["total_patients"]), "Unique patients")
    with c2:
        metric_card("Encounters", fmt_int(kpi["total_encounters"]), "Patient journeys summarized")
    with c3:
        metric_card("ED visits", fmt_int(kpi["ed_visits"]), "Emergency utilization")
    with c4:
        metric_card("Hospital admissions", fmt_int(kpi["hospital_admissions"]), "Admission burden")
    with c5:
        metric_card("High complexity", fmt_pct(kpi["pct_high_complexity"]), "High or extreme patients")

    st.markdown("### Utilization over time")

    monthly_total = (
        monthly.groupby("month", as_index=False)
        .agg(
            encounters=("encounters", "sum"),
            patients=("patients", "sum"),
            ed_visits=("ed_visits", "sum"),
            hospital_admissions=("hospital_admissions", "sum"),
            inpatient_admissions=("inpatient_admissions", "sum")
        )
        .sort_values("month")
    )

    fig = px.line(
        monthly_total,
        x="month",
        y=["encounters", "ed_visits", "hospital_admissions", "inpatient_admissions"],
        markers=True,
        title="Monthly utilization trend"
    )
    fig.update_layout(height=440, legend_title_text="Metric")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        dept_top = (
            monthly.groupby("department_type", as_index=False)["encounters"]
            .sum()
            .sort_values("encounters", ascending=False)
            .head(15)
        )
        fig = px.bar(
            dept_top,
            x="encounters",
            y="department_type",
            orientation="h",
            title="Top department types by encounter volume"
        )
        fig.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_dx = (
            dx_metrics.groupby("diagnosis_group", as_index=False)
            .agg(patients=("patients", "sum"), encounters=("encounters", "sum"))
            .sort_values("patients", ascending=False)
            .head(15)
        )
        fig = px.bar(
            top_dx,
            x="patients",
            y="diagnosis_group",
            orientation="h",
            title="Top diagnosis groups by patient count"
        )
        fig.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Department pressure table")
    st.dataframe(
        dept_metrics.sort_values("encounters", ascending=False).head(50),
        use_container_width=True,
        height=420
    )

# ============================================================
# Page 2: Journey Complexity
# ============================================================

elif page == "Journey Complexity":
    st.markdown('<div class="section-header">Journey Complexity / Care Friction</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
        Journey Complexity Score is a 0–100 operations score summarizing historical care burden:
        encounter volume, department fragmentation, diagnosis complexity, acute utilization,
        length-of-stay burden, and provider fragmentation. It is not a clinical severity score.
        </div>
        """,
        unsafe_allow_html=True
    )

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        selected_age = st.multiselect("Age group", sorted(pc["age_group"].dropna().unique().tolist()))
    with col_f2:
        selected_bucket = st.multiselect("Complexity bucket", ["Low", "Moderate", "High", "Extreme"])
    with col_f3:
        selected_vital = st.multiselect("Vital status", sorted(pc["vital_status"].dropna().astype(str).unique().tolist()))

    filtered_pc = pc.copy()
    if selected_age:
        filtered_pc = filtered_pc[filtered_pc["age_group"].isin(selected_age)]
    if selected_bucket:
        filtered_pc = filtered_pc[filtered_pc["complexity_bucket"].isin(selected_bucket)]
    if selected_vital:
        filtered_pc = filtered_pc[filtered_pc["vital_status"].astype(str).isin(selected_vital)]

    kpi = {
        "patients": len(filtered_pc),
        "avg_score": filtered_pc["journey_complexity_score"].mean(),
        "median_days": filtered_pc["journey_days"].median(),
        "high_extreme": filtered_pc["complexity_bucket"].isin(["High", "Extreme"]).mean() * 100 if len(filtered_pc) else 0,
        "avg_depts": filtered_pc["num_departments"].mean(),
        "avg_dx": filtered_pc["num_diagnosis_values"].mean()
    }

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Patients", fmt_int(kpi["patients"]), "Filtered cohort")
    with c2:
        metric_card("Avg complexity", fmt_float(kpi["avg_score"]), "0–100 score")
    with c3:
        metric_card("High/Extreme", fmt_pct(kpi["high_extreme"]), "Complexity bucket")
    with c4:
        metric_card("Median journey days", fmt_int(kpi["median_days"]), "First to last encounter")
    with c5:
        metric_card("Avg departments", fmt_float(kpi["avg_depts"]), "Distinct departments")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            filtered_pc,
            x="journey_complexity_score",
            nbins=40,
            color="complexity_bucket",
            title="Journey Complexity Score distribution",
            marginal="box"
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        bucket_counts = (
            filtered_pc["complexity_bucket"]
            .value_counts()
            .rename_axis("complexity_bucket")
            .reset_index(name="patients")
        )
        fig = px.pie(
            bucket_counts,
            names="complexity_bucket",
            values="patients",
            title="Complexity bucket mix",
            hole=0.45
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.scatter(
            filtered_pc.sample(min(len(filtered_pc), 5000), random_state=42) if len(filtered_pc) > 5000 else filtered_pc,
            x="num_encounters",
            y="journey_complexity_score",
            color="complexity_bucket",
            hover_data=["patient_id", "num_diagnosis_values", "num_departments", "hospital_admissions"],
            title="Complexity vs encounter volume"
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.scatter(
            filtered_pc.sample(min(len(filtered_pc), 5000), random_state=42) if len(filtered_pc) > 5000 else filtered_pc,
            x="max_los_hours",
            y="journey_complexity_score",
            color="complexity_bucket",
            hover_data=["patient_id", "num_encounters", "hospital_admissions", "inpatient_admissions"],
            title="Complexity vs maximum length of stay"
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Extreme journey queue")

    extreme_cols = [
        "patient_id",
        "journey_complexity_score",
        "complexity_bucket",
        "num_encounters",
        "num_diagnosis_values",
        "num_diagnosis_groups",
        "num_departments",
        "num_department_specialties",
        "num_attending_providers",
        "ed_visits",
        "hospital_admissions",
        "inpatient_admissions",
        "max_los_hours",
        "journey_days",
        "age_group",
        "vital_status"
    ]

    st.dataframe(
        filtered_pc[extreme_cols]
        .sort_values("journey_complexity_score", ascending=False)
        .head(200),
        use_container_width=True,
        height=520
    )

# ============================================================
# Page 3: Department Flow
# ============================================================

elif page == "Department Flow":
    st.markdown('<div class="section-header">Department Flow & Handoff Analytics</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="warning-box">
        These metrics do not prove that a department is underperforming. They surface flow patterns,
        possible handoff burden, routing bottlenecks, and pathways that may deserve operational review.
        </div>
        """,
        unsafe_allow_html=True
    )

    transitions = load_dept_transitions()

    min_transitions = st.slider("Minimum transitions to include", 10, 1000, 100, step=10)

    t = transitions[transitions["transition_count"] >= min_transitions].copy()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Transition pairs", fmt_int(len(t)), f"With ≥ {min_transitions} transitions")
    with c2:
        metric_card("Total transitions", fmt_int(t["transition_count"].sum()), "Filtered pairs")
    with c3:
        metric_card("Median same diagnosis", fmt_pct(t["pct_same_diagnosis"].median()), "Across transition pairs")
    with c4:
        metric_card("Median same specialty", fmt_pct(t["pct_same_specialty"].median()), "Across transition pairs")

    st.markdown("### Department transition heatmap")

    top_origin = (
        t.groupby("origin_department_type")["transition_count"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .index
    )
    top_next = (
        t.groupby("next_department_type")["transition_count"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .index
    )

    heat = t[
        t["origin_department_type"].isin(top_origin)
        & t["next_department_type"].isin(top_next)
    ].pivot_table(
        index="origin_department_type",
        columns="next_department_type",
        values="transition_count",
        aggfunc="sum",
        fill_value=0
    )

    fig = px.imshow(
        heat,
        text_auto=True,
        aspect="auto",
        title="Department type transition heatmap"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Patient flow Sankey")

    sankey_df = (
        t.groupby(["origin_department_type", "next_department_type"], as_index=False)["transition_count"]
        .sum()
        .sort_values("transition_count", ascending=False)
        .head(30)
    )

    labels = pd.unique(
        pd.concat([
            sankey_df["origin_department_type"],
            sankey_df["next_department_type"]
        ], ignore_index=True)
    ).tolist()

    label_map = {label: i for i, label in enumerate(labels)}

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=18,
                    line=dict(color="rgba(0,0,0,0.2)", width=0.5),
                    label=labels
                ),
                link=dict(
                    source=sankey_df["origin_department_type"].map(label_map),
                    target=sankey_df["next_department_type"].map(label_map),
                    value=sankey_df["transition_count"]
                )
            )
        ]
    )
    fig.update_layout(title_text="Top department type flows", height=600)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        leakage = (
            t.groupby("origin_department_type", as_index=False)
            .agg(
                transitions=("transition_count", "sum"),
                avg_changed_department_type=("pct_changed_department_type", "mean"),
                avg_same_diagnosis=("pct_same_diagnosis", "mean"),
                avg_same_specialty=("pct_same_specialty", "mean")
            )
            .sort_values("avg_changed_department_type", ascending=False)
            .head(20)
        )

        fig = px.bar(
            leakage,
            x="avg_changed_department_type",
            y="origin_department_type",
            orientation="h",
            title="Highest department-type switching rates"
        )
        fig.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        same_specialty = (
            t.groupby("origin_department_specialty", as_index=False)
            .agg(
                transitions=("transition_count", "sum"),
                avg_same_specialty=("pct_same_specialty", "mean")
            )
            .query("transitions >= @min_transitions")
            .sort_values("avg_same_specialty", ascending=False)
            .head(20)
        )

        fig = px.bar(
            same_specialty,
            x="avg_same_specialty",
            y="origin_department_specialty",
            orientation="h",
            title="Same-specialty movement across departments"
        )
        fig.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Department transition detail table")
    st.dataframe(
        t.sort_values("transition_count", ascending=False).head(500),
        use_container_width=True,
        height=520
    )

# ============================================================
# Page 4: Diagnosis Explorer
# ============================================================

elif page == "Diagnosis Explorer":
    st.markdown('<div class="section-header">Diagnosis / Cohort Explorer</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
        Use this page to explore patient journeys by diagnosis. DiagnosisValue / DiagnosisName are useful
        for tracking patient journeys over time.
        </div>
        """,
        unsafe_allow_html=True
    )

    dx = dx_metrics.copy()
    patient_dx = load_patient_dx()

    group_options = sorted(dx["diagnosis_group"].dropna().astype(str).unique().tolist())
    selected_groups = st.multiselect("Diagnosis group", group_options, default=[])

    search_text = st.text_input("Search diagnosis name/value", "")

    filtered_dx = dx.copy()
    if selected_groups:
        filtered_dx = filtered_dx[filtered_dx["diagnosis_group"].astype(str).isin(selected_groups)]
    if search_text.strip():
        s = search_text.lower().strip()
        filtered_dx = filtered_dx[
            filtered_dx["diagnosis_name"].astype(str).str.lower().str.contains(s, na=False)
            | filtered_dx["diagnosis_value"].astype(str).str.lower().str.contains(s, na=False)
            | filtered_dx["diagnosis_group"].astype(str).str.lower().str.contains(s, na=False)
        ]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Diagnosis rows", fmt_int(len(filtered_dx)), "Filtered diagnosis entries")
    with c2:
        metric_card("Patients", fmt_int(filtered_dx["patients"].sum()), "Across selected diagnoses")
    with c3:
        metric_card("Encounters", fmt_int(filtered_dx["encounters"].sum()), "Across selected diagnoses")
    with c4:
        metric_card("Avg deceased rate", fmt_pct(filtered_dx["pct_deceased"].mean()), "Unweighted avg")

    col1, col2 = st.columns(2)

    with col1:
        top_complex = (
            filtered_dx[filtered_dx["patients"] >= 50]
            .sort_values("avg_dx_complexity_percentile", ascending=False)
            .head(20)
        )
        fig = px.bar(
            top_complex,
            x="avg_dx_complexity_percentile",
            y="diagnosis_name",
            orientation="h",
            hover_data=["diagnosis_group", "patients", "pct_deceased", "pct_with_hospital_admission"],
            title="Top diagnoses by average normalized complexity"
        )
        fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_volume = (
            filtered_dx.sort_values("patients", ascending=False)
            .head(20)
        )
        fig = px.bar(
            top_volume,
            x="patients",
            y="diagnosis_name",
            orientation="h",
            hover_data=["diagnosis_group", "encounters", "pct_deceased"],
            title="Top diagnoses by patient count"
        )
        fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Diagnosis metrics table")
    st.dataframe(
        filtered_dx.sort_values(["patients", "avg_dx_complexity_percentile"], ascending=False).head(500),
        use_container_width=True,
        height=520
    )

    st.markdown("### Extreme patient-diagnosis journeys")

    dx_filtered_journeys = patient_dx.copy()
    if selected_groups:
        dx_filtered_journeys = dx_filtered_journeys[
            dx_filtered_journeys["diagnosis_group"].astype(str).isin(selected_groups)
        ]
    if search_text.strip():
        s = search_text.lower().strip()
        dx_filtered_journeys = dx_filtered_journeys[
            dx_filtered_journeys["diagnosis_name"].astype(str).str.lower().str.contains(s, na=False)
            | dx_filtered_journeys["diagnosis_value"].astype(str).str.lower().str.contains(s, na=False)
            | dx_filtered_journeys["diagnosis_group"].astype(str).str.lower().str.contains(s, na=False)
        ]

    st.dataframe(
        dx_filtered_journeys.sort_values("dx_complexity_percentile_within_group", ascending=False).head(300),
        use_container_width=True,
        height=520
    )

# ============================================================
# Page 5: Edema Outreach
# ============================================================

elif page == "Edema Outreach":
    st.markdown('<div class="section-header">Edema Early Outreach</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="success-box">
        This page is a focused use case: rank edema patients for proactive outreach.
        The goal is not to predict death directly, but to prioritize patients whose early journey
        resembles future high-risk trajectories.
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="info-box">
        <b>Final edema model:</b> strict high-risk trajectory target, 30-day survivorship guard,
        random forest selected by cross-validation, and top-k outreach evaluation.
        The best interpretation is outreach prioritization: the model identifies the top-ranked
        edema patients whose early encounter patterns are enriched for future high-risk trajectories.
        </div>
        """,
        unsafe_allow_html=True
    )


    possible_topk_files = [
        TABLE_DIR / "consolidated_06_final_test_topk.csv",
        TABLE_DIR / "ranking_no_age_05_topk_precision_recall_lift.csv",
        TABLE_DIR / "ranking_model_05_topk_precision_recall_lift.csv",
        TABLE_DIR / "model_compare_05_all_topk_precision_recall_lift.csv",
        TABLE_DIR / "edema_landmark_04_top_risk_capture_metrics.csv",
        TABLE_DIR / "final_edema_model_04_top_risk_capture.csv"
    ]

    possible_ranked_files = [
        TABLE_DIR / "consolidated_10_top10pct_outreach_list.csv",
        TABLE_DIR / "ranking_no_age_06_all_test_patients_ranked.csv",
        TABLE_DIR / "ranking_model_06_all_test_patients_ranked.csv",
        TABLE_DIR / "model_compare_06_best_model_ranked_test_patients.csv",
        TABLE_DIR / "edema_landmark_05_top_predicted_risk_test_patients.csv",
        TABLE_DIR / "final_edema_model_05_top_predicted_risk_patients.csv"
    ]

    possible_importance_files = [
        TABLE_DIR / "consolidated_08_feature_importance.csv",
        TABLE_DIR / "ranking_no_age_10_feature_importance.csv",
        TABLE_DIR / "ranking_model_10_feature_importance.csv",
        TABLE_DIR / "model_compare_07_best_model_feature_importance.csv",
        TABLE_DIR / "edema_landmark_06_feature_importance.csv",
        TABLE_DIR / "final_edema_model_06_feature_importance.csv"
    ]

    topk_file = next((p for p in possible_topk_files if p.exists()), None)
    ranked_file = next((p for p in possible_ranked_files if p.exists()), None)
    importance_file = next((p for p in possible_importance_files if p.exists()), None)

    if topk_file is None:
        st.warning(
            """
            No edema model top-k file found yet.

            Expected one of:
            - ranking_no_age_05_topk_precision_recall_lift.csv
            - ranking_model_05_topk_precision_recall_lift.csv
            - model_compare_05_all_topk_precision_recall_lift.csv
            - edema_landmark_04_top_risk_capture_metrics.csv
            """
        )
    else:
        topk = pd.read_csv(topk_file)

        # Normalize final consolidated output column names
        if "precision_high_risk_rate_pct" not in topk.columns and "precision_pct" in topk.columns:
            topk["precision_high_risk_rate_pct"] = topk["precision_pct"]

        if "recall_pct_of_all_high_risk_captured" not in topk.columns and "recall_pct" in topk.columns:
            topk["recall_pct_of_all_high_risk_captured"] = topk["recall_pct"]

        if "baseline_high_risk_rate_pct" not in topk.columns:
            if "lift_vs_baseline" in topk.columns and "precision_high_risk_rate_pct" in topk.columns:
                topk["baseline_high_risk_rate_pct"] = topk["precision_high_risk_rate_pct"] / topk["lift_vs_baseline"]
            else:
                topk["baseline_high_risk_rate_pct"] = np.nan

        if "actual_high_risk_in_flagged" not in topk.columns and "actual_positives_in_flagged" in topk.columns:
            topk["actual_high_risk_in_flagged"] = topk["actual_positives_in_flagged"]

        st.caption(f"Loaded: {topk_file}")

        # Normalize column names from older outputs
        if "precision_high_risk_rate_pct" not in topk.columns and "precision_positive_rate_in_flagged" in topk.columns:
            topk["precision_high_risk_rate_pct"] = topk["precision_positive_rate_in_flagged"]
        if "recall_pct_of_all_high_risk_captured" not in topk.columns and "recall_pct_of_all_positive_captured" in topk.columns:
            topk["recall_pct_of_all_high_risk_captured"] = topk["recall_pct_of_all_positive_captured"]
        if "baseline_high_risk_rate_pct" not in topk.columns:
            if "baseline_high_risk_rate" in topk.columns:
                topk["baseline_high_risk_rate_pct"] = topk["baseline_high_risk_rate"]
            else:
                topk["baseline_high_risk_rate_pct"] = np.nan

        first_row = topk.iloc[0]
        baseline = topk["baseline_high_risk_rate_pct"].dropna().iloc[0] if topk["baseline_high_risk_rate_pct"].notna().any() else np.nan

        risk_group_clean = topk["risk_group"].astype(str).str.lower().str.replace(" ", "_", regex=False)

        top5 = topk[risk_group_clean.isin(["top_5pct", "top_5%", "top5", "top_5"])].head(1)
        top10 = topk[risk_group_clean.isin(["top_10pct", "top_10%", "top10", "top_10"])].head(1)

        top5_precision = top5["precision_high_risk_rate_pct"].iloc[0] if len(top5) else np.nan
        top10_precision = top10["precision_high_risk_rate_pct"].iloc[0] if len(top10) else np.nan
        top10_lift = top10["lift_vs_baseline"].iloc[0] if len(top10) and "lift_vs_baseline" in top10.columns else (
            top10_precision / baseline if pd.notna(top10_precision) and pd.notna(baseline) and baseline > 0 else np.nan
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Baseline risk", fmt_pct(baseline), "Random outreach benchmark")
        with c2:
            metric_card("Top 5% precision", fmt_pct(top5_precision), "Urgent outreach group")
        with c3:
            metric_card("Top 10% precision", fmt_pct(top10_precision), "High-priority group")
        with c4:
            metric_card("Top 10% lift", f"{fmt_float(top10_lift, 2)}×", "vs baseline")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(
                topk,
                x="risk_group",
                y="precision_high_risk_rate_pct",
                markers=True,
                title="Top-k precision vs baseline"
            )
            if pd.notna(baseline):
                fig.add_hline(
                    y=baseline,
                    line_dash="dash",
                    annotation_text=f"Baseline {baseline:.2f}%"
                )
            fig.update_layout(height=450, xaxis_title="Risk group", yaxis_title="High-risk rate (%)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "lift_vs_baseline" in topk.columns:
                fig = px.bar(
                    topk,
                    x="risk_group",
                    y="lift_vs_baseline",
                    title="Lift over baseline"
                )
                fig.update_layout(height=450, xaxis_title="Risk group", yaxis_title="Lift")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Top-k metrics table")
        st.dataframe(topk, use_container_width=True, height=300)

    if ranked_file is not None:
        ranked = pd.read_csv(ranked_file)
        st.markdown("### Outreach queue / ranked test patients")
        st.caption(f"Loaded: {ranked_file}")

        if "predicted_risk" in ranked.columns:
            fig = px.histogram(
                ranked,
                x="predicted_risk",
                nbins=35,
                color="actual_high_risk_trajectory" if "actual_high_risk_trajectory" in ranked.columns else None,
                title="Predicted risk distribution"
            )
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(ranked.head(300), use_container_width=True, height=520)

    if importance_file is not None:
        imp = pd.read_csv(importance_file)
        st.markdown("### Model feature importance")
        st.caption(f"Loaded: {importance_file}")

        if "importance" in imp.columns and "feature" in imp.columns:
            top_imp = imp.sort_values("importance", ascending=False).head(25)
            fig = px.bar(
                top_imp.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                title="Top model features"
            )
            fig.update_layout(height=620)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Page 6: SDOH & Geography
# ============================================================

elif page == "SDOH & Geography":
    st.markdown('<div class="section-header">Social Determinants & Geography</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="warning-box">
        SDOH screening was not necessarily administered uniformly across all encounters.
        Screening gaps should be interpreted as workflow and rollout signals, not automatic department quality judgments.
        </div>
        """,
        unsafe_allow_html=True
    )

    sdoh = load_sdoh()

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Rows", fmt_int(len(sdoh)), "Department × diagnosis groups")
    with c2:
        metric_card("Total encounters", fmt_int(sdoh["encounters"].sum()), "Across SDOH table")
    with c3:
        metric_card("Avg screening rate", fmt_pct(sdoh["pct_encounters_screened"].mean()), "Unweighted avg")

    col1, col2 = st.columns(2)

    with col1:
        by_dept = (
            sdoh.groupby("department_type", as_index=False)
            .agg(
                encounters=("encounters", "sum"),
                screened_encounters=("screened_encounters", "sum")
            )
        )
        by_dept["screening_rate_pct"] = (
            by_dept["screened_encounters"] * 100 / by_dept["encounters"].replace(0, np.nan)
        ).round(2)

        by_dept = by_dept.sort_values("screening_rate_pct", ascending=False).head(20)

        fig = px.bar(
            by_dept,
            x="screening_rate_pct",
            y="department_type",
            orientation="h",
            title="SDOH screening rate by department type"
        )
        fig.update_layout(height=560, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        by_dx = (
            sdoh.groupby("diagnosis_group", as_index=False)
            .agg(
                encounters=("encounters", "sum"),
                screened_encounters=("screened_encounters", "sum")
            )
        )
        by_dx["screening_rate_pct"] = (
            by_dx["screened_encounters"] * 100 / by_dx["encounters"].replace(0, np.nan)
        ).round(2)

        by_dx = by_dx[by_dx["encounters"] >= 1000].sort_values("screening_rate_pct", ascending=False).head(20)

        fig = px.bar(
            by_dx,
            x="screening_rate_pct",
            y="diagnosis_group",
            orientation="h",
            title="SDOH screening rate by diagnosis group"
        )
        fig.update_layout(height=560, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### SDOH screening table")
    st.dataframe(
        sdoh.sort_values("encounters", ascending=False).head(500),
        use_container_width=True,
        height=520
    )

    st.markdown("### Geography layer")

    st.markdown(
        """
        <div class="info-box">
        This map joins patient home Census Block Group to TIGER census centroid coordinates.
        It is intended for geographic access and complexity review. Interpret small-area patterns carefully
        because privacy suppression and small counts may affect rural or low-volume areas.
        </div>
        """,
        unsafe_allow_html=True
    )

    geo, geo_error, tiger_path = load_geography_map_data()

    if tiger_path is not None:
        st.caption(f"TIGER census file used: {tiger_path}")

    if geo_error is not None:
        st.warning(geo_error)
        st.markdown(
            """
            To enable the map, make sure your TIGER census file exists in `../data/` and has columns:
            `GEOID`, `CENTLAT`, `CENTLONG`, and optionally `Population`.
            """
        )
    elif geo.empty:
        st.warning("Geography data was created but is empty after joining patient CensusBlockGroupFipsCode to TIGER GEOID.")
    else:
        # Optional privacy/count filter
        min_patients_geo = st.slider(
            "Minimum patients per geography point",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            help="Higher values reduce noise and avoid over-emphasizing tiny locations."
        )

        map_metric = st.selectbox(
            "Map color metric",
            [
                "patients",
                "encounters",
                "avg_complexity_score",
                "pct_high_or_extreme_complexity",
                "ed_visits_per_100_patients",
                "admissions_per_100_patients",
                "patients_per_1000_population"
            ],
            index=0
        )

        geo_plot = geo[geo["patients"] >= min_patients_geo].copy()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Mapped block groups", fmt_int(len(geo_plot)), f"≥ {min_patients_geo} patients")
        with c2:
            metric_card("Mapped patients", fmt_int(geo_plot["patients"].sum()), "Across visible geographies")
        with c3:
            metric_card("Avg complexity", fmt_float(geo_plot["avg_complexity_score"].mean()), "Visible geographies")
        with c4:
            metric_card("ED / 100 patients", fmt_float(geo_plot["ed_visits_per_100_patients"].mean()), "Visible geographies")

        if geo_plot.empty:
            st.warning("No geography points meet the selected minimum patient threshold.")
        else:
            geo_plot = fix_us_longitude_if_needed(geo_plot, lon_col="CENTLONG", lat_col="CENTLAT")
            center = get_kansas_center(geo_plot)

            hover_cols = {
                "GEOID": True,
                "patients": ":,",
                "encounters": ":,",
                "avg_complexity_score": ":.2f",
                "pct_high_or_extreme_complexity": ":.2f",
                "ed_visits_per_100_patients": ":.2f",
                "admissions_per_100_patients": ":.2f",
                "Population": ":,",
                "patients_per_1000_population": ":.2f",
                "CENTLAT": False,
                "CENTLONG": False
            }

            map_type = st.radio(
                "Map style",
                ["Choropleth if boundary file exists", "Point map", "Density map"],
                horizontal=True
            )

            gj, gj_path, gj_error = load_geojson_boundaries()

            if map_type == "Choropleth if boundary file exists" and gj is not None:
                st.caption(f"GeoJSON boundary file used: {gj_path}")

                fig = px.choropleth_mapbox(
                    geo_plot,
                    geojson=gj,
                    locations="geoid_clean",
                    featureidkey="properties.GEOID_CLEAN",
                    color=map_metric,
                    hover_name="GEOID",
                    hover_data={
                        "patients": ":,",
                        "encounters": ":,",
                        "avg_complexity_score": ":.2f",
                        "pct_high_or_extreme_complexity": ":.2f",
                        "ed_visits_per_100_patients": ":.2f",
                        "admissions_per_100_patients": ":.2f",
                        "patients_per_1000_population": ":.2f",
                        "geoid_clean": False,
                    },
                    mapbox_style="open-street-map",
                    center=center,
                    zoom=6.6,
                    opacity=0.68,
                    title=f"Census block group choropleth: {map_metric}"
                )
                fig.update_layout(height=700, margin={"r": 0, "t": 45, "l": 0, "b": 0})
                st.plotly_chart(fig, use_container_width=True)

            elif map_type == "Density map":
                if map_type == "Choropleth if boundary file exists" and gj is None:
                    st.warning(
                        f"True choropleth requires a Census Block Group GeoJSON boundary file. {gj_error}"
                    )

                fig = px.density_mapbox(
                    geo_plot,
                    lat="CENTLAT",
                    lon="CENTLONG",
                    z=map_metric,
                    radius=24,
                    center=center,
                    zoom=6.6,
                    mapbox_style="open-street-map",
                    hover_name="GEOID",
                    title=f"Geographic density surface: {map_metric}"
                )
                fig.update_layout(height=700, margin={"r": 0, "t": 45, "l": 0, "b": 0})
                st.plotly_chart(fig, use_container_width=True)

            else:
                if map_type == "Choropleth if boundary file exists" and gj is None:
                    st.warning(
                        f"True choropleth requires polygon boundaries. {gj_error} "
                        "Showing point map fallback using Census Block Group centroids."
                    )

                fig = px.scatter_mapbox(
                    geo_plot,
                    lat="CENTLAT",
                    lon="CENTLONG",
                    size="patients",
                    color=map_metric,
                    hover_name="GEOID",
                    hover_data=hover_cols,
                    size_max=34,
                    zoom=6.6,
                    center=center,
                    mapbox_style="open-street-map",
                    title=f"Patient geography point map colored by {map_metric}"
                )
                fig.update_layout(height=700, margin={"r": 0, "t": 45, "l": 0, "b": 0})
                st.plotly_chart(fig, use_container_width=True)

            col_map1, col_map2 = st.columns(2)

            with col_map1:
                top_geo_complexity = (
                    geo_plot.sort_values("avg_complexity_score", ascending=False)
                    .head(20)
                )
                fig = px.bar(
                    top_geo_complexity.sort_values("avg_complexity_score"),
                    x="avg_complexity_score",
                    y="GEOID",
                    orientation="h",
                    title="Top geographies by average complexity score",
                    hover_data=["patients", "encounters", "pct_high_or_extreme_complexity"]
                )
                fig.update_layout(height=520, yaxis_title="Census block group")
                st.plotly_chart(fig, use_container_width=True)

            with col_map2:
                top_geo_ed = (
                    geo_plot.sort_values("ed_visits_per_100_patients", ascending=False)
                    .head(20)
                )
                fig = px.bar(
                    top_geo_ed.sort_values("ed_visits_per_100_patients"),
                    x="ed_visits_per_100_patients",
                    y="GEOID",
                    orientation="h",
                    title="Top geographies by ED visits per 100 patients",
                    hover_data=["patients", "encounters", "hospital_admissions"]
                )
                fig.update_layout(height=520, yaxis_title="Census block group")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Geography detail table")
            st.dataframe(
                geo_plot.sort_values(map_metric, ascending=False).head(500),
                use_container_width=True,
                height=520
            )



# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.caption(
    "SVH Journey Intelligence Dashboard. This dashboard surfaces operational patterns for review; it does not replace clinical judgment."
)
