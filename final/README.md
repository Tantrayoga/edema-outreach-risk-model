# SVH Patient Journey Dashboard Submission

This submission has two main parts:

1. A Jupyter notebook that prepares the data, trains the prediction model, and saves all dashboard resources.
2. A Streamlit dashboard that reads the saved output files and displays the final results.

Run the notebook first. The dashboard will not work correctly until the notebook has created the `outputs/` files.

---

## 1. Required folder structure

Use this structure:

```text
svh_project/
├── data/
│   ├── encounters.csv
│   ├── patients.csv
│   ├── diagnosis.csv
│   ├── departments.csv
│   ├── providers.csv
│   ├── social_determinants.csv
│   └── tigercensuscodes.csv
│
├── outputs/
│   ├── tables/
│   ├── figures/
│   └── models/
│
└── submission/
    ├── SVH_final_submission.ipynb
    ├── svh_journey_dashboard.py
    ├── dashboard_requirements.txt
    └── README.md
```

Important: the notebook uses `../data` and `../outputs`, so the notebook should be inside a folder such as `submission/`, with `data/` and `outputs/` one level above it.

If your notebook has a different name, that is okay. Just run the cleaned final `.ipynb` file first.

---

## 2. Set up the Python environment on Mac

Open Terminal and go to the submission folder:

```bash
cd path/to/svh_project/submission
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r dashboard_requirements.txt
```

If `streamlit` is not found later, use:

```bash
python -m streamlit run svh_journey_dashboard.py
```

instead of:

```bash
streamlit run svh_journey_dashboard.py
```

---

## 3. Run the notebook first

Start Jupyter Notebook from the `submission/` folder:

```bash
python -m notebook
```

Open:

```text
SVH_final_submission.ipynb
```

Then run all cells from top to bottom.

The notebook will:

1. Convert raw CSV files in `../data/` into Parquet files.
2. Build the master encounter table.
3. Build patient journey friction tables.
4. Create dashboard tables and figures.
5. Build the edema early-window feature table.
6. Train the final outreach risk model.
7. Save final prediction and outreach files.
8. Optionally create the geography file for the map page.

---

## 4. Files created by the notebook

After the notebook runs, these folders should contain files:

```text
svh_project/outputs/tables/
svh_project/outputs/figures/
svh_project/outputs/models/
```

Important dashboard resources include:

```text
outputs/tables/01_overall_summary.csv
outputs/tables/02_mychart_vs_high_friction.csv
outputs/tables/03_sd_screening_vs_high_friction.csv
outputs/tables/04_screening_gap_high_friction_by_diagnosis_group.csv
outputs/tables/05_extreme_friction_diagnosis_groups.csv
outputs/tables/06_top_extreme_journeys.csv
outputs/tables/07_department_type_sd_screening_heatmap_data.csv
outputs/tables/consolidated_05_cv_summary.csv
outputs/tables/consolidated_06_final_test_topk.csv
outputs/tables/consolidated_07_calibration.csv
outputs/tables/consolidated_08_feature_importance.csv
outputs/tables/consolidated_09_ranked_test_patients.csv
outputs/tables/consolidated_10_top10pct_outreach_list.csv
outputs/tables/consolidated_11_model_card.txt
outputs/tables/consolidated_14_threshold_sensitivity.csv

outputs/figures/consolidated_final_topk.png
outputs/figures/consolidated_feature_importance.png
outputs/figures/consolidated_calibration.png
outputs/figures/consolidated_threshold_sensitivity.png

outputs/models/consolidated_best_*.joblib
```

Optional geography output:

```text
outputs/fast_shaded_cbg.geojson
```

The dashboard can still run without the optional geography file, but the Geography page will show a missing-file message.

---

## 5. Run the Streamlit dashboard

After the notebook finishes successfully, run:

```bash
python -m streamlit run svh_journey_dashboard.py
```

Streamlit should open the dashboard in your browser.

If it does not open automatically, copy the local URL shown in the terminal, usually something like:

```text
http://localhost:8501
```

---

## 6. Correct run order

Always use this order:

```text
1. Put raw CSV files in svh_project/data/
2. Activate venv
3. Install requirements
4. Run the notebook from top to bottom
5. Run the Streamlit dashboard
```

Do not run the dashboard first. It only displays files that the notebook already created.

---

## 7. Troubleshooting

### Problem: `zsh: command not found: streamlit`

Use:

```bash
python -m streamlit run svh_journey_dashboard.py
```

### Problem: dashboard says output files are missing

Run the notebook first. Then check that these folders exist:

```text
../outputs/tables/
../outputs/figures/
../outputs/models/
```

### Problem: notebook cannot find the data

Make sure the raw CSV files are in:

```text
svh_project/data/
```

and the notebook is in:

```text
svh_project/submission/
```

The notebook expects the data folder to be one level above the notebook folder.

### Problem: dashboard is using old or weird files

Check the Streamlit sidebar. It shows:

```text
Running file
Output directory
Latest key output
```

Make sure it is running:

```text
svh_journey_dashboard.py
```

from the correct folder, and make sure the output directory points to your current project’s `outputs/` folder.

If needed, delete the old `outputs/` folder, rerun the notebook, then rerun the dashboard.

---

## 8. What to submit

Submit these files:

```text
SVH_final_submission.ipynb
svh_journey_dashboard.py
dashboard_requirements.txt
README.md
```

If the graders need to reproduce the dashboard, they should also have access to the required `data/` folder or the generated `outputs/` folder.
