# CodeAlpha_CreditScoringModel

## Project Overview
This project was built for the **CodeAlpha Machine Learning Internship — Task 1**.
It predicts whether a loan applicant is a **good credit risk** or a **bad
credit risk** (likely to default) using their past financial and personal
data. The project uses classic, easy-to-understand machine learning
classifiers rather than deep learning, in line with the task requirements.

## Objective
Given information about a loan applicant (income-related behaviour, credit
history, loan purpose, employment status, etc.), predict their
**creditworthiness** as a binary label:

- `0` → Good credit risk (creditworthy)
- `1` → Bad credit risk (likely to default)

## Dataset
**Name:** Statlog (German Credit Data)
**Source:** UCI Machine Learning Repository
(original page: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
**File used:** `data/german.csv` (a clean, comma-separated mirror of the
original dataset, 1000 rows × 21 columns, no header row)

- **Target variable:** `credit_risk` — originally coded `1` = good credit,
  `2` = bad credit. In this project it is remapped to `0` = good, `1` = bad.
- **Important features:** `checking_account_status`, `duration_months`,
  `credit_history`, `purpose`, `credit_amount`, `savings_account`,
  `employment_since`, `age`, `housing`, `property`, `job`, and 10 other
  demographic/financial attributes.
- **Preprocessing needed:** The dataset has **no missing values**, but all
  categorical columns are stored as short codes (e.g. `A11`, `A34`) and
  must be **label-encoded** before being used by the models. Numerical
  features are **standardized** (scaled) before training.

The dataset is already included in this repository at `data/german.csv`,
so no manual download or Kaggle account is required. It was originally
sourced from the official UCI mirror on GitHub
(`jbrownlee/Datasets/german.csv`).

## Technologies Used
- Python 3
- Pandas, NumPy — data loading & preprocessing
- Matplotlib, Seaborn — visualization
- Scikit-learn — modeling & evaluation

## Methodology
1. Load the dataset and inspect its shape, types, and missing values.
2. Recode the target variable into a simple binary label (0 = good, 1 = bad).
3. Encode all categorical columns using Label Encoding.
4. Split the data into training (80%) and test (20%) sets, stratified by
   the target so both sets keep the same good/bad ratio.
5. Scale numerical features with `StandardScaler`.
6. Train two classification models.
7. Evaluate both models on the held-out test set.
8. Save all plots and a results summary to the `results/` folder.

## Machine Learning Algorithms
- **Logistic Regression** — a simple, interpretable linear baseline.
- **Random Forest Classifier** — a stronger ensemble model that also
  provides feature importance scores.

## Evaluation Metrics
Both models are evaluated on the test set using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve

## Results
Results below are from an actual run of `src/credit_scoring.py`
(`random_state=42`, 80/20 train-test split):

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.765    | 0.627     | 0.533  | 0.577    | 0.790   |
| Random Forest       | 0.770    | 0.706     | 0.400  | 0.511    | 0.798   |

**Top 5 most important features (Random Forest):**
1. `credit_amount`
2. `checking_account_status`
3. `age`
4. `duration_months`
5. `purpose`

Re-running the script will reproduce these exact numbers because the
random seed is fixed. All generated plots (confusion matrices, ROC
curves, feature importance chart) are saved automatically to `results/`.

## Project Structure
```
CodeAlpha_CreditScoringModel/
│
├── data/
│   └── german.csv                # Statlog German Credit dataset
├── src/
│   └── credit_scoring.py         # Main script: load → preprocess → train → evaluate
├── results/                      # Auto-generated plots & metrics (created after running)
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── roc_curve_logistic_regression.png
│   ├── roc_curve_random_forest.png
│   ├── feature_importance_random_forest.png
│   ├── model_comparison.csv
│   └── results_summary.txt
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation Instructions

### 1. Create a virtual environment
```bash
python -m venv venv
```

### 2. Activate it (Windows)
```bash
venv\Scripts\activate
```
On macOS/Linux, use `source venv/bin/activate` instead.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## How to Run
From the project's root folder:
```bash
python src/credit_scoring.py
```

## Expected Output
After running the script, you should see in your terminal:
- Dataset shape and a preview of the first rows
- Confirmation that there are no missing values
- Training progress messages for both models
- Accuracy, Precision, Recall, F1-score, and ROC-AUC for each model
- A full classification report for each model

And in the `results/` folder, you should find:
- `confusion_matrix_logistic_regression.png` and `confusion_matrix_random_forest.png`
- `roc_curve_logistic_regression.png` and `roc_curve_random_forest.png`
- `feature_importance_random_forest.png`
- `model_comparison.csv` — table comparing both models
- `results_summary.txt` — plain text summary of all results

## Future Improvements
- Try additional models (e.g., Gradient Boosting, XGBoost) for comparison.
- Use one-hot encoding instead of label encoding for nominal categorical
  features, and compare performance.
- Perform hyperparameter tuning (GridSearchCV) for both models.
- Apply class-imbalance handling techniques (e.g., SMOTE) since the
  dataset has more "good" than "bad" credit examples (700 vs 300).
- Deploy the trained model behind a simple web API or Streamlit app.
