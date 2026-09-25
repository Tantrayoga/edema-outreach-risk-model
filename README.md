# Edema Early-Warning System

Risk-ranking model that flags edema patients likely to deteriorate in the 60 days after diagnosis, so a hospital's limited outreach staff can call the highest-risk patients first instead of at random.

Built for **DataFest 2026** (healthcare track, Team 6 — "Git Push & Pray").

## The problem

Patients who come in with edema (swelling) are routinely given a diuretic and sent home. Most are fine. A minority are the first visible sign of heart, kidney, or liver failure, and that group has a mortality rate several times higher than the general edema population. With only a handful of nursing coordinators and thousands of edema patients, random follow-up calls catch a small fraction of the people who actually need one.

## The approach

- Build a 60-day "early window" feature set per patient from their encounters, diagnoses, department visits, and utilization patterns before and after their first edema diagnosis — no vitals or labs required, since this needs to work as an early screen.
- Deliberately exclude age/birth-year as a model feature (age is reported for auditing only) so the model isn't just re-deriving "old patients are risky."
- Compare several classifiers (Random Forest, Extra Trees, Logistic Regression, Gradient Boosting, XGBoost, Histogram Gradient Boosting) with repeated stratified cross-validation.
- Optimize for **top-decile precision and lift** rather than overall accuracy, because the real operational question is: *"If staff can only call the top 5–10% of patients, how much better is that list than calling people at random?"*
- Calibrate the winning model (isotonic) so predicted risk scores are usable as actual probabilities, not just a ranking.
- Surface results through a Streamlit dashboard for hospital ops/coordinators.

## Results

Best model: **Random Forest**, selected on mean top-10% precision across CV folds.

| Model | ROC AUC | Avg Precision | Top-10% Precision | Top-10% Lift |
|---|---|---|---|---|
| **Random Forest** | 0.674 | 0.169 | 0.169 | **3.00x** |
| Extra Trees | 0.677 | 0.157 | 0.168 | 2.97x |
| Logistic Regression | 0.633 | 0.146 | 0.132 | 2.34x |
| Gradient Boosting | 0.626 | 0.125 | 0.130 | 2.31x |
| XGBoost | 0.613 | 0.126 | 0.112 | 1.99x |
| Hist. Gradient Boosting | 0.594 | 0.115 | 0.109 | 1.94x |

Cohort: 4,289 edema patients, 6.99% baseline high-risk rate. On this cohort, targeting the top 10% by predicted risk instead of calling patients at random gets ~3x more true high-risk patients per call — turning a $50 phone call into the tool that catches what would otherwise become a $15,000 ICU admission.

Full metric tables and charts (all cohort-level aggregates, no patient rows) are in [`outputs/tables/`](outputs/tables/) and [`outputs/figures/`](outputs/figures/), including feature importance, calibration curves, threshold sensitivity, bootstrap confidence intervals, and top-k precision/recall/lift at every cutoff.

## Repo structure

```
final/
  SVH_final_submission.ipynb     # data prep -> feature engineering -> model training -> outreach list
  svh_journey_dashboard.py       # Streamlit dashboard reading the notebook's saved outputs
  dashboard_requirements.txt
outputs/
  figures/                       # model performance & cohort charts (aggregate, no patient data)
  tables/                        # model metrics, feature importance, calibration, cohort summaries
```

## Running it

The notebook expects `data/` (raw hospital extracts) and writes to `outputs/` one level up. That raw data is **not included in this repo** (see below), so the notebook can't be re-run end-to-end without it — the code and the aggregate results it produced are here for review.

If you do have access to the underlying data:

```bash
cd final
python -m venv venv && source venv/bin/activate
python -m pip install -r dashboard_requirements.txt
# run SVH_final_submission.ipynb top to bottom, then:
python -m streamlit run svh_journey_dashboard.py
```

## Data & privacy

This project uses a hospital patient dataset provided under DataFest's competition data-use terms, which prohibit redistributing the raw or patient-level data. Accordingly, this repo contains **only**:

- the modeling/dashboard code,
- and cohort-level aggregate outputs (counts, rates, feature importances, calibration curves, model metrics) with no per-patient rows or identifiers.

No raw patient records, per-patient predictions, or outreach lists are published here.

## Limitations

- This is a risk-ranking tool, not a diagnostic or clinical decision system.
- Trained and evaluated on one health system; generalization elsewhere is unverified.
- The "deceased" label used for training doesn't carry an exact death date, so the target is a heuristic, not a ground-truth clinical outcome.
- Features lean on prior healthcare utilization, which correlates with age, comorbidity, and access — the model needs to be re-validated before any real deployment, not used as-is.
