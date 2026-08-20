"""
CodeAlpha Machine Learning Internship - Task 1
Credit Scoring Model

Objective:
Predict whether a loan applicant is a GOOD credit risk or a BAD credit
risk (i.e., likely to default) using past financial data, with a simple
and understandable Logistic Regression / Random Forest pipeline.

Dataset:
Statlog (German Credit Data) - UCI Machine Learning Repository.
1000 rows, 20 input features (7 numerical, 13 categorical) and one
binary target column describing credit risk.

This script:
    1. Loads the dataset
    2. Inspects it
    3. Cleans / preprocesses it (encoding, scaling)
    4. Trains two models (Logistic Regression and Random Forest)
    5. Evaluates both models (Accuracy, Precision, Recall, F1, ROC-AUC)
    6. Saves plots (confusion matrix, ROC curve, feature importance)
      and a text summary of the results into the results/ folder
"""

# ----------------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ----------------------------------------------------------------------
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# Make plots look clean and consistent
sns.set(style="whitegrid")

# ----------------------------------------------------------------------
# 2. PATHS (kept relative to the project root so the script works
#    the same way for anyone who clones the repository)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "german.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 3. LOAD THE DATASET
# ----------------------------------------------------------------------
# The raw file has NO header row, so we supply readable column names
# ourselves. These names follow the official Statlog German Credit
# Data documentation (see README.md for the full description).
COLUMN_NAMES = [
    "checking_account_status",   # status of existing checking account
    "duration_months",           # duration of the credit in months
    "credit_history",            # credit history
    "purpose",                   # purpose of the loan
    "credit_amount",             # loan amount
    "savings_account",           # savings account / bonds
    "employment_since",          # present employment since
    "installment_rate",          # installment rate (% of disposable income)
    "personal_status_sex",       # personal status and sex
    "other_debtors",             # other debtors / guarantors
    "present_residence_since",   # years at present residence
    "property",                  # property owned
    "age",                       # age in years
    "other_installment_plans",   # other installment plans
    "housing",                   # housing situation
    "existing_credits",          # number of existing credits at this bank
    "job",                       # job / employment type
    "num_dependents",            # number of people financially dependent
    "telephone",                 # has telephone or not
    "foreign_worker",            # foreign worker status
    "credit_risk",               # TARGET: 1 = good credit, 2 = bad credit
]

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)

# ----------------------------------------------------------------------
# 4. BASIC DATA INSPECTION
# ----------------------------------------------------------------------
print("\nDataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nColumn data types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum())

# ----------------------------------------------------------------------
# 5. TARGET VARIABLE CLEAN-UP
# ----------------------------------------------------------------------
# Original coding: 1 = Good credit risk, 2 = Bad credit risk.
# We convert this into an easier-to-read binary target:
#   0 = Good credit (creditworthy)
#   1 = Bad credit  (likely to default)  <- this is the "positive" class
df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})

print("\nTarget class distribution (0 = Good, 1 = Bad):")
print(df["credit_risk"].value_counts())

# ----------------------------------------------------------------------
# 6. HANDLE MISSING VALUES
# ----------------------------------------------------------------------
# The Statlog German Credit dataset does not contain missing values,
# but we still add a safety check so the script stays robust if the
# dataset is swapped out for a similar one that does have gaps.
if df.isnull().sum().sum() > 0:
    # Numerical columns -> fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Categorical columns -> fill with the most frequent value (mode)
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    print("\nMissing values were found and filled.")
else:
    print("\nNo missing values found in the dataset.")

# ----------------------------------------------------------------------
# 7. ENCODE CATEGORICAL FEATURES
# ----------------------------------------------------------------------
# The dataset uses short codes (e.g. "A11", "A34") for categorical
# attributes. We use Label Encoding, which is simple, fast, and works
# well for tree-based models like Random Forest. Logistic Regression
# still works fine on label-encoded + scaled data for a project of
# this scope, keeping the pipeline simple as required.
categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
print("\nCategorical columns to encode:", categorical_cols)

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ----------------------------------------------------------------------
# 8. FEATURE / TARGET SEPARATION
# ----------------------------------------------------------------------
X = df.drop("credit_risk", axis=1)
y = df["credit_risk"]

# ----------------------------------------------------------------------
# 9. TRAIN / TEST SPLIT
# ----------------------------------------------------------------------
# stratify=y keeps the same good/bad ratio in both train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# ----------------------------------------------------------------------
# 10. FEATURE SCALING
# ----------------------------------------------------------------------
# Logistic Regression benefits from scaled features. Random Forest
# does not need scaling, but using the same scaled data for both
# models keeps the pipeline simple and does not hurt Random Forest.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 11. MODEL TRAINING
# ----------------------------------------------------------------------
print("\nTraining Logistic Regression model...")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

print("Training Random Forest model...")
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
rf_clf.fit(X_train_scaled, y_train)

models = {
    "Logistic Regression": log_reg,
    "Random Forest": rf_clf,
}

# ----------------------------------------------------------------------
# 12. PREDICTIONS + EVALUATION
# ----------------------------------------------------------------------
results_summary = []

for name, model in models.items():
    print(f"\n{'=' * 60}\nEvaluating: {name}\n{'=' * 60}")

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    results_summary.append(
        {
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1,
            "ROC-AUC": auc,
        }
    )

    # ------------------- Confusion Matrix -------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Good (0)", "Bad (1)"],
        yticklabels=["Good (0)", "Bad (1)"],
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    filename = f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

    # ------------------- ROC Curve -------------------
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    filename = f"roc_curve_{name.replace(' ', '_').lower()}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

# ----------------------------------------------------------------------
# 13. FEATURE IMPORTANCE (Random Forest only - it directly supports it)
# ----------------------------------------------------------------------
importances = rf_clf.feature_importances_
feat_importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": importances}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(data=feat_importance_df, x="Importance", y="Feature", color="steelblue")
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_importance_random_forest.png"))
plt.close()

# ----------------------------------------------------------------------
# 14. SAVE A TEXT SUMMARY OF ALL RESULTS
# ----------------------------------------------------------------------
results_df = pd.DataFrame(results_summary)
results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

with open(os.path.join(RESULTS_DIR, "results_summary.txt"), "w") as f:
    f.write("CodeAlpha Task 1 - Credit Scoring Model\n")
    f.write("Model comparison (test set)\n")
    f.write("=" * 60 + "\n\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")
    f.write("Top 5 most important features (Random Forest):\n")
    f.write(feat_importance_df.head(5).to_string(index=False))

print("\nAll done! Plots and result files were saved to the 'results/' folder.")
print(results_df.to_string(index=False))
