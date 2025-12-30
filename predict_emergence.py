import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. LOAD DATA
file_path = 'peds_data.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')
df.columns = [str(col).strip() for col in df.columns]

# 2. CLINICAL CLEANING
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

# 3. MODEL PREP
feature_cols = ['Age', 'Gender', 'Surgery Type', 'Weight (kg)', 
                'Preop mYPAS score', 'Duration of surgery (mins)', 
                'Time to emergence (mins)', 'Airway Device Removed']

X = df[feature_cols].copy()
y = df['ED_Target']
X_encoded = pd.get_dummies(X, columns=['Gender', 'Surgery Type', 'Airway Device Removed'])

# 4. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 5. RESULTS ON TEST SET (The 9 kids it didn't see)
y_pred = rf.predict(X_test)
df['Doc_Prediction'] = df['Anesthesiologist prediction of ED'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
doc_test_preds = df.loc[y_test.index, 'Doc_Prediction']

print("\n" + "="*35)
print("     PEDIATRIC EMERGENCE STUDY")
print("="*35)
print(f"AI Test Accuracy (Unseen Data): {accuracy_score(y_test, y_pred):.2%}")
print(f"Doc Test Accuracy:             {accuracy_score(y_test, doc_test_preds):.2%}")

# 6. FEATURE IMPORTANCE (The "Why")
importances = rf.feature_importances_
feature_names = X_encoded.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 8))
plt.title('Clinical Drivers of Emergence Reactions')
plt.barh(range(len(indices)), importances[indices], color='crimson', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Importance Score (Weighted Impact)')
plt.tight_layout()
plt.savefig('feature_importance.png')

print("\nSuccess! Results calculated and chart saved as 'feature_importance.png'!")
print("="*35)