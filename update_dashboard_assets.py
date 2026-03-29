from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = ROOT / "dashboard"
ASSETS_DIR = DASHBOARD_DIR / "assets"
DATA_DIR = DASHBOARD_DIR / "data"

COMMON_ASSETS = [
    "01_Cohort_Overview.png",
    "02_Target_Prevalence.png",
    "03_Feature_Correlations.png",
]

TARGET_ASSETS = [
    "02_Confusion_Matrix_{suffix}.png",
    "03_ROC_Curve_{suffix}.png",
    "06_Precision_Recall_{suffix}.png",
    "04_Feature_Importance_{suffix}.png",
    "07_Performance_Summary_{suffix}.png",
    "08_Predicted_Risk_Distribution_{suffix}.png",
    "09_AI_vs_Anesthesiologist_{suffix}.png",
    "05_Patient_Scoreboard_{suffix}.csv",
]


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


summary_path = ROOT / "dashboard_summary.json"
if not summary_path.exists():
    raise FileNotFoundError("dashboard_summary.json not found. Run predict_emergence.py first.")

summary = json.loads(summary_path.read_text())

ASSETS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

for asset in COMMON_ASSETS:
    asset_path = ROOT / asset
    if asset_path.exists():
        copy_file(asset_path, ASSETS_DIR / asset)

for target in summary.get("targets", []):
    if not target.get("available"):
        continue
    suffix = target.get("output_suffix")
    if not suffix:
        continue
    for template in TARGET_ASSETS:
        filename = template.format(suffix=suffix)
        asset_path = ROOT / filename
        if not asset_path.exists():
            continue
        if filename.endswith(".csv"):
            copy_file(asset_path, DATA_DIR / filename)
        else:
            copy_file(asset_path, ASSETS_DIR / filename)

# Report files
for report_file in ["00_Study_Report.txt", "00_Study_Report.pdf"]:
    report_path = ROOT / report_file
    if report_path.exists():
        copy_file(report_path, DATA_DIR / report_file)

# Full dataset export (optional)
for dataset_file in ["peds_data_92.csv", "peds_data.xlsx"]:
    dataset_path = ROOT / dataset_file
    if dataset_path.exists():
        copy_file(dataset_path, DATA_DIR / dataset_file)

# Summary JSON for dashboard
copy_file(summary_path, DATA_DIR / "summary.json")

print("Dashboard assets updated.")
