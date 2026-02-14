import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict


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


def generate_oof_correctness(features, target, participant_ids):
    valid_mask = target.notna()
    if valid_mask.sum() == 0 or target[valid_mask].nunique() < 2:
        print("[WARN] Target values missing or single-class; AI correctness not computed.")
        return pd.Series(pd.NA, index=participant_ids, dtype="object")

    min_class = target[valid_mask].value_counts().min()
    if min_class < 2:
        print("[WARN] Not enough samples per class for out-of-fold correctness.")
        return pd.Series(pd.NA, index=participant_ids, dtype="object")

    n_splits = min(5, int(min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rf_oof = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )
    oof_probs = cross_val_predict(
        rf_oof,
        features.loc[valid_mask],
        target[valid_mask],
        cv=cv,
        method="predict_proba",
    )[:, 1]
    oof_pred = (oof_probs >= 0.5).astype(int)
    correctness = np.where(
        oof_pred == target[valid_mask].astype(int).to_numpy(), "Yes", "No"
    )
    correctness_series = pd.Series(pd.NA, index=participant_ids.index, dtype="object")
    correctness_series.loc[valid_mask] = pd.Series(
        correctness, index=target[valid_mask].index, dtype="object"
    )
    return correctness_series


def main(input_path: str, output_path: str) -> None:
    df = pd.read_excel(input_path, engine="openpyxl")
    df.columns = [str(col).strip() for col in df.columns]

    participant_col = "Participant Number"
    if participant_col in df.columns:
        df[participant_col] = df[participant_col].astype(str)
    participant_ids = df[participant_col] if participant_col in df.columns else df.index

    if "Age" in df.columns:
        df["Age"] = df["Age"].apply(parse_peds_age)

    numeric_cols = [
        "Age",
        "Weight (kg)",
        "Preop mYPAS score",
        "Duration of surgery (mins)",
        "Time to emergence (mins)",
        "PAED score at 5 mins",
    ]
    coerce_numeric_columns(df, numeric_cols)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in ["Gender", "Surgery Type", "Airway Device Removed"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    if "PAED score at 5 mins" not in df.columns:
        raise ValueError("Missing 'PAED score at 5 mins' column in dataset.")

    df["ED_Target_PAED12_5min"] = (df["PAED score at 5 mins"] > 12).astype(int)

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
    X = df[[col for col in feature_cols if col in df.columns]].copy()
    X_encoded = pd.get_dummies(
        X,
        columns=[
            col
            for col in ["Gender", "Surgery Type", "Airway Device Removed"]
            if col in X.columns
        ],
    )

    ai_correct_col = "AI Correct?"
    if ai_correct_col in df.columns:
        df.drop(columns=[ai_correct_col], inplace=True)

    df.insert(
        len(df.columns),
        ai_correct_col,
        generate_oof_correctness(X_encoded, df["ED_Target_PAED12_5min"], participant_ids),
    )

    df.to_excel(output_path, index=False)
    print(f"[OK] Wrote updated file to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write AI correctness column to Excel file.")
    parser.add_argument("--input", default="peds_data.xlsx", help="Input Excel file path")
    parser.add_argument(
        "--output",
        default="peds_data_with_ai_correct.xlsx",
        help="Output Excel file path",
    )
    args = parser.parse_args()
    main(args.input, args.output)
