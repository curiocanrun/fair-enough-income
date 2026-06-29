"""
Loads Adult Income Dataset, applies encoding, and prepares X (features), y (target), P (protected)

Target: y = 1 (income > $50K), y = 0 (income <= $50K)
Protected Attribute: gender = 0 (Female), 1 (Male)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ROOT = Path(__file__).parent
# DATASET_PATH = ROOT / "datasets" / "adult.csv"

# if not DATASET_PATH.exists():
#     raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv('adult.csv', skipinitialspace=True)

df = df.replace('?', 'Unknown')

df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

# ==============================
#      Protected Attribute (P) - GENDER
# ==============================

df['gender_binary'] = df['gender'].map({
    'Male': 1,    # Privileged
    'Female': 0   # Unprivileged
})

# ==============================
#      One-Hot Encoding
# ==============================

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ==============================
#      EXPORT: X, y, P
# ==============================

X = df.drop(['income', 'fnlwgt'], axis=1).values
P = df['gender_binary'].values
y = df['income'].values

print(f"Loaded: {df.shape[0]} rows, {X.shape[1]} features")
print(f"Privileged (Male): {np.sum(P==1)}, Unprivileged (Female): {np.sum(P==0)}")
print(f"Protected ratio: {np.sum(P==1)/np.sum(P==0):.2f}")

__all__ = [X, y, P]
