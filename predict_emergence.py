import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_auc_score,
    PrecisionRecallDisplay,
    average_precision_score,
)
import sys

# ==========================================
# 1. DATA INTAKE & CLEANING
# ==========================================
print("--- INITIALIZING RESEARCH PIPELINE ---")
file_path = "peds_data.xlsx"

try:
    df = pd.read_excel(file_path, engine="openpyxl")
    print("[OK] Data Loaded.")
except Exception as e:
    print(f"[FAIL] Critical Error: {e}")
    sys.exit()

# Normalize columns
_df_columns = [str(col).strip() for col in df.columns]
df.columns = _df_columns

participant_col = "Participant Number"

# Drop blank/empty rows that Excel may include
df.dropna(how="all", inplace=True)
_core_cols = [c for c in ["Age", "Gender", "PAED score at 5 mins"] if c in df.columns]
if _core_cols:
    df.dropna(subset=_core_cols, how="all", inplace=True)
df.reset_index(drop=True, inplace=True)

if participant_col in df.columns:
    df[participant_col] = df[participant_col].astype(str)


def apply_plot_style():
    plt.style.use("seaborn-v0_8")
    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.facecolor": "#f7f7f7",
            "figure.facecolor": "white",
            "grid.alpha": 0.3,
            "font.size": 10,
        }
    )


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def parse_peds_age(age_val):
    s = str(age_val).lower().strip()
    num_str = "".join([c for c in s if c.isdigit() or c == "."])
    if not num_str:
        return np.nan
    val = float(num_str)
    if "month" in s:
        return val / 12.0
    return val


def coerce_numeric_columns(dataframe, columns):
    for col in columns:
        if col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")


# Core numeric cleanup
numeric_cols = [
    "Age",
    "Weight (kg)",
    "Preop mYPAS score",
    "Duration of surgery (mins)",
    "Time to emergence (mins)",
    "PAED score at awaking",
    "PAED score at 5 mins",
]

if "Age" in df.columns:
    df["Age"] = df["Age"].apply(parse_peds_age)

coerce_numeric_columns(df, numeric_cols)

# DIAGNOSTIC: warn if PAED score at 5 mins has too few unique values after coercion
_paed_diag_col = "PAED score at 5 mins"
if _paed_diag_col in df.columns:
    _n_unique = df[_paed_diag_col].nunique(dropna=True)
    _n_nan = df[_paed_diag_col].isna().sum()
    print(f"[DIAG] '{_paed_diag_col}': {_n_unique} unique values, {_n_nan} NaN before fillna")
    if _n_unique <= 1:
        print(f"[WARN] '{_paed_diag_col}' has only {_n_unique} unique value(s) after coercion — "
              f"check that the column is numeric in the source file!")

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

for col in ["Gender", "Surgery Type", "Airway Device Removed"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()


def find_item_columns(columns, keyword, time_filter=None):
    matches = [col for col in columns if keyword in col.lower()]
    if time_filter:
        matches = [col for col in matches if time_filter(col.lower())]
    return matches


def compute_ed1_score(dataframe, item_columns):
    values = dataframe[item_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    # PAED items 1-3 are reverse-scored (0-4), so convert to delirium intensity
    return (4 - values[item_columns[0]]) + (4 - values[item_columns[1]]) + (
        4 - values[item_columns[2]]
    )


def normalize_binary_prediction(series):
    if series is None:
        return None
    cleaned = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
    }
    return cleaned.map(mapping)


def compute_confusion_metrics(y_true, y_pred):
    if y_true is None or y_pred is None:
        return None
    mask = (~y_true.isna()) & (~y_pred.isna())
    if mask.sum() == 0:
        return None
    y_true_valid = y_true[mask].astype(int)
    y_pred_valid = y_pred[mask].astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true_valid, y_pred_valid, labels=[0, 1]
    ).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    acc = accuracy_score(y_true_valid, y_pred_valid)
    return {
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n": int(mask.sum()),
    }


print("[...] Preparing target definitions...")
paed_5min_col = "PAED score at 5 mins"

if paed_5min_col in df.columns:
    df["ED_Target_PAED12_5min"] = (df[paed_5min_col] > 12).astype(int)
else:
    print("[WARN] PAED score at 5 mins column missing; ED target not created.")
    df["ED_Target_PAED12_5min"] = np.nan

anesth_col = "Anesthesiologist prediction of ED"
if anesth_col in df.columns:
    df["Anesthesiologist_Prediction"] = normalize_binary_prediction(df[anesth_col])
else:
    df["Anesthesiologist_Prediction"] = np.nan

# Optional ED I (items 1-3) detection remains for future use
awake_filter = lambda col: "awak" in col or "awake" in col
five_min_filter = lambda col: "5" in col and "min" in col

ed1_awake_cols = [
    find_item_columns(df.columns, "eye contact", awake_filter),
    find_item_columns(df.columns, "purpose", awake_filter),
    find_item_columns(df.columns, "aware", awake_filter),
]
ed1_5min_cols = [
    find_item_columns(df.columns, "eye contact", five_min_filter),
    find_item_columns(df.columns, "purpose", five_min_filter),
    find_item_columns(df.columns, "aware", five_min_filter),
]

ed1_available = all(len(group) > 0 for group in ed1_awake_cols) or all(
    len(group) > 0 for group in ed1_5min_cols
)

ed1_column_notes = []

if ed1_available:
    ed1_scores = []
    if all(len(group) > 0 for group in ed1_awake_cols):
        awake_columns = [group[0] for group in ed1_awake_cols]
        ed1_scores.append(compute_ed1_score(df, awake_columns))
        ed1_column_notes.append(f"Awaking columns: {awake_columns}")
    if all(len(group) > 0 for group in ed1_5min_cols):
        five_min_columns = [group[0] for group in ed1_5min_cols]
        ed1_scores.append(compute_ed1_score(df, five_min_columns))
        ed1_column_notes.append(f"5-min columns: {five_min_columns}")

    if ed1_scores:
        ed1_score_final = pd.concat(ed1_scores, axis=1).max(axis=1)
        df["ED_Target_ED1_6"] = (ed1_score_final >= 6).astype(int)
    else:
        df["ED_Target_ED1_6"] = np.nan
        ed1_available = False
else:
    df["ED_Target_ED1_6"] = np.nan

# Capture base_columns here, after all target/derived columns exist,
# so df.insert(len(base_columns), ...) places "AI Correct?" at the correct position.
base_columns = df.columns.tolist()

# ---- EXTENDED DIAGNOSTICS ----
print("[DIAG] ---- TARGET & SPLIT DIAGNOSTICS ----")
_y_diag = df["ED_Target_PAED12_5min"]
_pos = int(_y_diag.sum())
_neg = int((1 - _y_diag).sum())
_total = len(_y_diag)
print(f"[DIAG] ED_Target_PAED12_5min  -> total={_total}  positive(ED)={_pos}  negative={_neg}  prevalence={_pos/_total:.1%}")
print(f"[DIAG] PAED score at 5 mins   -> min={df['PAED score at 5 mins'].min():.1f}  max={df['PAED score at 5 mins'].max():.1f}  median={df['PAED score at 5 mins'].median():.1f}")
# Simulate the same train/test split to show how many positives land in test
from sklearn.model_selection import train_test_split as _tts
_X_tmp, _, _y_train_tmp, _y_test_tmp = _tts(_y_diag, _y_diag, test_size=0.3, random_state=42, stratify=_y_diag)
print(f"[DIAG] Simulated test split    -> test_n={len(_y_test_tmp)}  test_positive={int(_y_test_tmp.sum())}  test_negative={int((1-_y_test_tmp).sum())}")
print(f"[DIAG] Anesthesiologist_Prediction NaN count: {df['Anesthesiologist_Prediction'].isna().sum()}")
print("[DIAG] ------------------------------------------")

# ==========================================
# 2. MODEL TRAINING
# ==========================================
print("[...] Training Models...")
feature_cols = [
    "Age",
    "Gender",
    "Surgery Type",
    "Weight (kg)",
    "Preop mYPAS score",
    "Duration of surgery (mins)",
    "Time to emergence (mins)",
    "Airway Device Removed",
]

missing_features = [col for col in feature_cols if col not in df.columns]
if missing_features:
    print(f"[WARN] Missing feature columns: {missing_features}")

X = df[[col for col in feature_cols if col in df.columns]].copy()
X_encoded = pd.get_dummies(
    X,
    columns=[
        col
        for col in ["Gender", "Surgery Type", "Airway Device Removed"]
        if col in X.columns
    ],
)

participant_col = "Participant Number"
participant_ids = df[participant_col] if participant_col in df.columns else df.index


def generate_oof_correctness(target_series):
    if target_series is None:
        print("[WARN] Missing target values for AI correctness.")
        return pd.Series(pd.NA, index=df.index, dtype="object")
    valid_mask = target_series.notna()
    if valid_mask.sum() == 0:
        print("[WARN] Target values are empty; AI correctness not computed.")
        return pd.Series(pd.NA, index=df.index, dtype="object")
    target_valid = target_series[valid_mask]
    if target_valid.nunique() < 2:
        print("[WARN] Target has a single class; AI correctness not computed.")
        return pd.Series(pd.NA, index=df.index, dtype="object")
    min_class = target_valid.value_counts().min()
    if min_class < 2:
        print("[WARN] Not enough samples per class for out-of-fold correctness.")
        return pd.Series(pd.NA, index=df.index, dtype="object")

    n_splits = min(5, int(min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rf_oof = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )
    oof_probs = cross_val_predict(
        rf_oof,
        X_encoded.loc[valid_mask],
        target_valid,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    oof_pred = (oof_probs >= 0.5).astype(int)
    correctness = np.where(oof_pred == target_valid.astype(int).to_numpy(), "Yes", "No")
    correctness_series = pd.Series(pd.NA, index=df.index, dtype="object")
    correctness_series.loc[valid_mask] = pd.Series(
        correctness, index=target_valid.index, dtype="object"
    )
    return correctness_series


def train_evaluate_target(target_name, y, output_suffix, clinician_pred=None):
    if y is None or y.isna().all():
        return {
            "available": False,
            "message": f"Target {target_name} not available in this dataset.",
        }
    if y.nunique() < 2:
        return {
            "available": False,
            "message": f"Target {target_name} has only one class; cannot train.",
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_probs = rf.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    acc = accuracy_score(y_test, y_pred)

    print(f"[DIAG] [{output_suffix}] y_test distribution: positive={int(y_test.sum())} negative={int((1-y_test).sum())}")
    print(f"[DIAG] [{output_suffix}] confusion matrix: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"[DIAG] [{output_suffix}] sensitivity={sensitivity:.2%}  specificity={specificity:.2%}  accuracy={acc:.2%}")

    roc_auc = None
    if len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_probs)

    avg_precision = average_precision_score(y_test, y_probs)

    apply_plot_style()

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_estimator(
        rf, X_test, y_test, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion Matrix ({target_name})")
    save_figure(f"02_Confusion_Matrix_{output_suffix}.png")

    # ROC Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random Chance")
    ax.set_title(f"ROC Curve ({target_name})")
    ax.legend(loc="lower right")
    save_figure(f"03_ROC_Curve_{output_suffix}.png")

    # Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_estimator(rf, X_test, y_test, ax=ax)
    ax.set_title(f"Precision-Recall Curve ({target_name})")
    ax.text(
        0.02,
        0.02,
        f"Average Precision: {avg_precision:.2f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    save_figure(f"06_Precision_Recall_{output_suffix}.png")

    # Feature Importance (Top 12)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-12:]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], color="#4c78a8")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([X_encoded.columns[i] for i in indices])
    ax.set_title(f"Top Feature Importance ({target_name})")
    save_figure(f"04_Feature_Importance_{output_suffix}.png")

    # Performance Metrics Bar Chart
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics = {
        "Accuracy": acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
    }
    bars = ax.bar(metrics.keys(), metrics.values(), color="#72b7b2")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(f"Model Performance Summary ({target_name})")
    for bar, value in zip(bars, metrics.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_figure(f"07_Performance_Summary_{output_suffix}.png")

    # Predicted Risk Distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_probs, bins=10, color="#f58518", edgecolor="white")
    ax.set_title(f"Predicted Risk Distribution ({target_name})")
    ax.set_xlabel("Predicted Probability of ED")
    ax.set_ylabel("Patient Count")
    save_figure(f"08_Predicted_Risk_Distribution_{output_suffix}.png")

    clinician_test_metrics = None
    clinician_pred_test = None
    if clinician_pred is not None:
        clinician_pred_test = clinician_pred.loc[X_test.index]
        clinician_test_metrics = compute_confusion_metrics(y_test, clinician_pred_test)

    if clinician_test_metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        metrics = ["accuracy", "sensitivity", "specificity"]
        ai_values = [acc, sensitivity, specificity]
        clinician_values = [
            clinician_test_metrics["accuracy"],
            clinician_test_metrics["sensitivity"],
            clinician_test_metrics["specificity"],
        ]
        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width / 2, ai_values, width, label="AI", color="#4c78a8")
        ax.bar(
            x + width / 2,
            clinician_values,
            width,
            label="Anesthesiologist",
            color="#f58518",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([m.title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title(f"AI vs Anesthesiologist ({target_name})")
        ax.legend()
        save_figure(f"09_AI_vs_Anesthesiologist_{output_suffix}.png")

    # Patient Scoreboard
    scoreboard = X.loc[X_test.index].copy()
    scoreboard["Actual_Outcome"] = y_test
    scoreboard["AI_Prediction"] = y_pred
    scoreboard["AI_Confidence"] = (y_probs * 100).round(2)
    if clinician_pred_test is not None:
        scoreboard["Anesthesiologist_Prediction"] = clinician_pred_test
        scoreboard["Anesthesiologist_Correct"] = (
            clinician_pred_test == y_test
        ).astype(int)
    scoreboard.to_csv(f"05_Patient_Scoreboard_{output_suffix}.csv", index=False)

    return {
        "available": True,
        "target_name": target_name,
        "output_suffix": output_suffix,
        "prevalence": y.mean(),
        "train_n": len(X_train),
        "test_n": len(X_test),
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "clinician_test_metrics": clinician_test_metrics,
    }


# ==========================================
# 3. GENERATING ASSETS
# ==========================================
print("[...] Generating Visual Assets...")
apply_plot_style()

# Cohort overview
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
if "Age" in df.columns:
    axes[0, 0].hist(df["Age"], bins=12, color="#4c78a8", edgecolor="white")
    axes[0, 0].set_title("Age Distribution")
    axes[0, 0].set_xlabel("Age (years)")
    axes[0, 0].set_ylabel("Count")

if "Weight (kg)" in df.columns:
    axes[0, 1].hist(df["Weight (kg)"], bins=12, color="#72b7b2", edgecolor="white")
    axes[0, 1].set_title("Weight Distribution")
    axes[0, 1].set_xlabel("Weight (kg)")

if "Surgery Type" in df.columns:
    surgery_counts = df["Surgery Type"].value_counts().head(6)
    axes[1, 0].barh(surgery_counts.index[::-1], surgery_counts.values[::-1], color="#f58518")
    axes[1, 0].set_title("Top Surgery Types")

if "Preop mYPAS score" in df.columns:
    axes[1, 1].hist(
        df["Preop mYPAS score"], bins=10, color="#54a24b", edgecolor="white"
    )
    axes[1, 1].set_title("Preop mYPAS Distribution")

save_figure("01_Cohort_Overview.png")

# Target prevalence
targets_present = {
    "PAED > 12 at 5 mins": df["ED_Target_PAED12_5min"]
}
if ed1_available:
    targets_present["ED I ≥ 6"] = df["ED_Target_ED1_6"]

fig, ax = plt.subplots(figsize=(6, 4))
prevalences = {name: series.mean() for name, series in targets_present.items()}
ax.bar(prevalences.keys(), prevalences.values(), color="#e45756")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title("Target Prevalence")
for idx, (name, value) in enumerate(prevalences.items()):
    ax.text(idx, value + 0.02, f"{value:.0%}", ha="center", fontsize=9)
save_figure("02_Target_Prevalence.png")

# Correlation heatmap
numeric_available = [col for col in numeric_cols if col in df.columns]
if len(numeric_available) >= 2:
    corr = df[numeric_available].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title("Numeric Feature Correlations")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    save_figure("03_Feature_Correlations.png")

results = []
results.append(
    train_evaluate_target(
        "PAED > 12 at 5 mins",
        df["ED_Target_PAED12_5min"],
        "PAED12_5MIN",
        clinician_pred=df["Anesthesiologist_Prediction"],
    )
)
results.append(train_evaluate_target("ED I ≥ 6", df["ED_Target_ED1_6"], "ED1_6"))

# ==========================================
# 4. GENERATING REPORT TEXT
# ==========================================
print("[...] Writing Technical Report...")

report_lines = [
    "STUDY: Machine Learning Prediction of Pediatric Emergence Delirium",
    "TECHNICAL REPORT & MODEL AUDIT",
    "===============================================================",
    "",
    "1. COHORT ANALYSIS",
    f"   Total Patients (N): {len(df)}",
    "",
    "2. TARGET DEFINITIONS",
    "   - PAED > 12 at 5 mins (primary study endpoint)",
    "   - ED I ≥ 6 (PAED items 1-3 only; reverse-scored, optional)",
]

if ed1_available:
    report_lines.append("   ED I columns used:")
    report_lines.extend([f"     * {note}" for note in ed1_column_notes])
else:
    report_lines.append("   ED I target unavailable in this dataset (item-level columns not found).")

report_lines.extend(
    [
        "",
        "3. MODEL PERFORMANCE METRICS (TEST SET)",
    ]
)

for result in results:
    if not result["available"]:
        report_lines.append(f"   - {result['message']}")
        continue

    report_lines.extend(
        [
            f"   Target: {result['target_name']}",
            f"     Training Set:       {result['train_n']}",
            f"     Test Set:           {result['test_n']}",
            f"     Prevalence:         {result['prevalence']:.2%}",
            f"     Accuracy:           {result['accuracy']:.2%}",
            f"     Sensitivity:        {result['sensitivity']:.2%}",
            f"     Specificity:        {result['specificity']:.2%}",
            f"     PPV (Precision):    {result['ppv']:.2%}",
            f"     NPV:                {result['npv']:.2%}",
            f"     ROC AUC:            {result['roc_auc']:.3f}" if result["roc_auc"] is not None else "     ROC AUC:            N/A",
            f"     Avg Precision:      {result['avg_precision']:.3f}",
            f"     Confusion Matrix:   TP={result['tp']} FP={result['fp']} TN={result['tn']} FN={result['fn']}",
            "",
        ]
    )

report_lines.extend(
    [
        "4. CLINICIAN VS AI COMPARISON (PRIMARY TARGET)",
    ]
)

primary_result = next((r for r in results if r.get("available")), None)
clinician_full_metrics = compute_confusion_metrics(
    df["ED_Target_PAED12_5min"], df["Anesthesiologist_Prediction"]
)
if clinician_full_metrics is None:
    report_lines.append("   - Clinician prediction column missing or incomplete.")
else:
    report_lines.extend(
        [
            f"   Clinician Accuracy (Full Cohort): {clinician_full_metrics['accuracy']:.2%}",
            f"   Clinician Sensitivity: {clinician_full_metrics['sensitivity']:.2%}",
            f"   Clinician Specificity: {clinician_full_metrics['specificity']:.2%}",
            f"   Clinician PPV: {clinician_full_metrics['ppv']:.2%}",
            f"   Clinician NPV: {clinician_full_metrics['npv']:.2%}",
        ]
    )

if primary_result and primary_result.get("clinician_test_metrics"):
    clinician_test = primary_result["clinician_test_metrics"]
    report_lines.extend(
        [
            "",
            "   Test Set (AI vs Clinician)",
            f"     AI Accuracy:        {primary_result['accuracy']:.2%}",
            f"     Clinician Accuracy: {clinician_test['accuracy']:.2%}",
            f"     Accuracy Delta:     {(primary_result['accuracy'] - clinician_test['accuracy']):.2%}",
        ]
    )

report_lines.extend(
    [
        "",
        "5. GENERATED ASSETS",
        "   - 01_Cohort_Overview.png",
        "   - 02_Target_Prevalence.png",
        "   - 03_Feature_Correlations.png",
        "   - 02_Confusion_Matrix_[target].png",
        "   - 03_ROC_Curve_[target].png",
        "   - 06_Precision_Recall_[target].png",
        "   - 04_Feature_Importance_[target].png",
        "   - 07_Performance_Summary_[target].png",
        "   - 08_Predicted_Risk_Distribution_[target].png",
        "   - 09_AI_vs_Anesthesiologist_[target].png",
        "   - 05_Patient_Scoreboard_[target].csv",
        "",
        "===============================================================",
        "GENERATED BY SCIKIT-LEARN RANDOM FOREST (n=300, class_weight=balanced)",
    ]
)

with open("00_Study_Report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# ==========================================
# 5. DASHBOARD SUMMARY OUTPUT
# ==========================================

def normalize_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def normalize_dict(data):
    normalized = {}
    for key, value in data.items():
        if isinstance(value, dict):
            normalized[key] = normalize_dict(value)
        elif isinstance(value, list):
            normalized[key] = [
                normalize_dict(item) if isinstance(item, dict) else normalize_value(item)
                for item in value
            ]
        else:
            normalized[key] = normalize_value(value)
    return normalized


assets_common = [
    "01_Cohort_Overview.png",
    "02_Target_Prevalence.png",
    "03_Feature_Correlations.png",
]

comparison_summary = normalize_dict(
    compute_confusion_metrics(
        df["ED_Target_PAED12_5min"], df["Anesthesiologist_Prediction"]
    )
)

target_summaries = []
for result in results:
    normalized = normalize_dict(result)
    if not normalized.get("available"):
        target_summaries.append(normalized)
        continue

    suffix = normalized["output_suffix"]
    normalized["assets"] = {
        "confusion_matrix": f"02_Confusion_Matrix_{suffix}.png",
        "roc_curve": f"03_ROC_Curve_{suffix}.png",
        "precision_recall": f"06_Precision_Recall_{suffix}.png",
        "feature_importance": f"04_Feature_Importance_{suffix}.png",
        "performance_summary": f"07_Performance_Summary_{suffix}.png",
        "risk_distribution": f"08_Predicted_Risk_Distribution_{suffix}.png",
        "ai_vs_clinician": f"09_AI_vs_Anesthesiologist_{suffix}.png",
        "scoreboard": f"05_Patient_Scoreboard_{suffix}.csv",
    }
    target_summaries.append(normalized)

summary_payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "cohort": {
        "total_patients": int(len(df)),
        "ed1_available": bool(ed1_available),
        "ed1_columns": ed1_column_notes,
    },
    "primary_target": "PAED > 12 at 5 mins",
    "targets": target_summaries,
    "assets_common": assets_common,
    "comparison": {
        "clinician_full_metrics": comparison_summary,
    },
    "report": {
        "text_file": "00_Study_Report.txt",
        "pdf_file": "00_Study_Report.pdf",
    },
}

with open("dashboard_summary.json", "w") as f:
    json.dump(summary_payload, f, indent=2)

# Write AI correctness back to Excel (Column Z)
ai_correct_col = "AI Correct?"
if ai_correct_col in df.columns:
    df.drop(columns=[ai_correct_col], inplace=True)

df.insert(len(base_columns), ai_correct_col, generate_oof_correctness(df["ED_Target_PAED12_5min"]))

output_columns = base_columns + [col for col in df.columns if col not in base_columns]
output_df = df[output_columns]

output_path = "peds_data_with_ai_correct.xlsx"
output_df.to_excel(output_path, index=False)
print(f"[OK] Wrote updated file to {output_path}")

# Bundle outputs into zip for easy sharing
zip_name = "study_outputs.zip"
zip_paths = [
    "00_Study_Report.pdf",
    "00_Study_Report.txt",
    "01_Cohort_Overview.png",
    "02_Target_Prevalence.png",
    "03_Feature_Correlations.png",
    "02_Confusion_Matrix_PAED12_5MIN.png",
    "03_ROC_Curve_PAED12_5MIN.png",
    "04_Feature_Importance_PAED12_5MIN.png",
    "06_Precision_Recall_PAED12_5MIN.png",
    "07_Performance_Summary_PAED12_5MIN.png",
    "08_Predicted_Risk_Distribution_PAED12_5MIN.png",
    "09_AI_vs_Anesthesiologist_PAED12_5MIN.png",
    "05_Patient_Scoreboard_PAED12_5MIN.csv",
    output_path,
]

with ZipFile(zip_name, "w") as zip_file:
    for file_name in zip_paths:
        file_path = Path(file_name)
        if file_path.exists():
            zip_file.write(file_path, arcname=file_path.name)

print(f"[OK] Wrote output bundle to {zip_name}")

print("--- PROCESS COMPLETE ---")
print("Check your folder for updated report and image assets.")
