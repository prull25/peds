import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             ConfusionMatrixDisplay, RocCurveDisplay, recall_score)
import sys

# ==========================================
# 1. DATA INTAKE & CLEANING
# ==========================================
print("--- INITIALIZING RESEARCH PIPELINE ---")
file_path = 'peds_data.xlsx'

try:
    df = pd.read_excel(file_path, engine='openpyxl')
    print("[OK] Data Loaded.")
except Exception as e:
    print(f"[FAIL] Critical Error: {e}")
    sys.exit()

# Cleaning
df.columns = [str(col).strip() for col in df.columns]

def parse_peds_age(age_val):
    s = str(age_val).lower().strip()
    num_str = "".join([c for c in s if c.isdigit() or c == '.'])
    if not num_str: return np.nan
    val = float(num_str)
    if 'month' in s: return val / 12.0
    return val

df['Age'] = df['Age'].apply(parse_peds_age)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['PAED score at awaking'] = pd.to_numeric(df['PAED score at awaking'], errors='coerce').fillna(0)
df['PAED score at 5 mins'] = pd.to_numeric(df['PAED score at 5 mins'], errors='coerce').fillna(0)
df['ED_Target'] = ((df['PAED score at awaking'] > 12) | (df['PAED score at 5 mins'] > 12)).astype(int)

for col in ['Gender', 'Surgery Type', 'Airway Device Removed']:
    df[col] = df[col].astype(str).str.strip().str.lower()

# ==========================================
# 2. MODEL TRAINING
# ==========================================
print("[...] Training Model...")
feature_cols = ['Age', 'Gender', 'Surgery Type', 'Weight (kg)', 
                'Preop mYPAS score', 'Duration of surgery (mins)', 
                'Time to emergence (mins)', 'Airway Device Removed']

X = df[feature_cols].copy()
y = df['ED_Target']
X_encoded = pd.get_dummies(X, columns=['Gender', 'Surgery Type', 'Airway Device Removed'])

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ==========================================
# 3. GENERATING ASSETS
# ==========================================
print("[...] Generating Visual Assets...")

# Asset 1: Demographics
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
df['Surgery Type'].value_counts().plot(kind='bar')
plt.title('Surgery Type')
plt.subplot(1, 2, 2)
plt.hist(df['Age'], bins=10)
plt.title('Age Distribution')
plt.tight_layout()
plt.savefig('01_Demographics.png')

# Asset 2: Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, cmap='Blues', ax=ax)
plt.title("Confusion Matrix")
plt.savefig('02_Confusion_Matrix.png')

# Asset 3: ROC Curve
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax)
plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
plt.title("ROC Curve")
plt.savefig('03_ROC_Curve.png')

# Asset 4: Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X_encoded.columns[i] for i in indices])
plt.title('Feature Importance (Model Drivers)')
plt.tight_layout()
plt.savefig('04_Feature_Importance.png')

# Asset 5: The Data Dump
y_pred = rf.predict(X_test)
y_probs = rf.predict_proba(X_test)[:, 1]
scoreboard = X.loc[X_test.index].copy()
scoreboard['Actual_Outcome'] = y_test
scoreboard['AI_Prediction'] = y_pred
scoreboard['AI_Confidence'] = (y_probs * 100).round(2)
scoreboard.to_csv('05_Patient_Scoreboard.csv')

# ==========================================
# 4. GENERATING REPORT TEXT
# ==========================================
print("[...] Writing Technical Report...")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0
acc = accuracy_score(y_test, y_pred)

report_content = f"""
STUDY: Machine Learning Prediction of Pediatric Emergence Delirium
TECHNICAL REPORT & MODEL AUDIT
===============================================================

1. COHORT ANALYSIS
   Total Patients (N): {len(df)}
   Training Set:       {len(X_train)}
   Test Set:           {len(X_test)}
   Prevalence of ED:   {df['ED_Target'].mean():.2%}

2. MODEL PERFORMANCE METRICS (TEST SET)
   Accuracy:           {acc:.2%}
   Sensitivity (Recall): {sensitivity:.2%}  (Ability to detect true positives)
   Specificity:        {specificity:.2%}  (Ability to exclude true negatives)
   PPV (Precision):    {ppv:.2%}  (Likelihood a positive prediction is correct)
   NPV:                {npv:.2%}  (Likelihood a negative prediction is correct)

3. CONFUSION MATRIX RAW COUNTS
   True Positives:     {tp}
   False Positives:    {fp}
   True Negatives:     {tn}
   False Negatives:    {fn}

===============================================================
GENERATED BY SCIKIT-LEARN RANDOM FOREST (n=100)
"""

with open('00_Study_Report.txt', 'w') as f:
    f.write(report_content)

print("--- PROCESS COMPLETE ---")
print("Check your folder for 00_Study_Report.txt and 5 image assets.")