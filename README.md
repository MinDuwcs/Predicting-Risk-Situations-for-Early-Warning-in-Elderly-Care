# 🏥 Predicting Risk Situations for Early Warning in Elderly Care

A Machine Learning classification project analyzing IoT wearable sensor data to detect fall events in elderly individuals and provide early warnings.

## 📋 Project Overview

| | |
|---|---|
| **Dataset** | `safety_monitoring.csv` — 10,000 records from IoT wearable sensors |
| **Problem** | Binary classification: Fall vs No Fall |
| **Target** | `Fall Detected (Yes/No)` — Imbalanced (499 Fall / 9,501 No Fall) |
| **Tools** | Python · Scikit-learn · Imbalanced-learn · Seaborn · Pandas |

## 🗂️ Project Pipeline

1. 📊 **Exploratory Data Analysis (EDA)** — Timestamp patterns, univariate & correlation analysis
2. 🔧 **Feature Engineering & Preprocessing** — Inactivity_Risk, Hour, Day_of_Week, Label Encoding
3. ⚠️ **Data Leakage Detection & Analysis** *(critical finding)*
4. 🔴 **Scenario A** — Full Features (Benchmark, demonstrates leakage effect)
5. 🟢 **Scenario B** — Pre-Event Features Only (Realistic, deployable model)
6. 📈 **5-Model Comparison** with Stratified 5-Fold Cross-Validation
7. 💡 **Conclusions & Recommendations**

## 🔑 Key Findings

- **Data Leakage detected:** `Post-Fall Inactivity Duration`, `Alert Triggered`, and `Caregiver Notified` are post-event features — using them yields 100% accuracy but has no real-world value.
- **Scenario B — Random Forest** achieves **F1 = 0.9275, AUC = 0.9530** using only 4 pre-event features, demonstrating a realistic and deployable model.
- Class imbalance (95% No Fall) was addressed using **SMOTE**, balancing the dataset to 9,501 samples per class.

## 📊 Model Comparison (Scenario B — 5-Fold CV)

| Rank | Model | F1 | AUC |
|------|-------|----|-----|
| 🥇 | Random Forest | 0.9275 | 0.9530 |
| 🥈 | Decision Tree | 0.8973 | 0.9202 |
| 🥉 | Gradient Boosting | 0.8953 | 0.9255 |
| 4 | Naive Bayes | 0.8937 | 0.9053 |
| 5 | Logistic Regression | 0.6998 | 0.6439 |

## 📁 Repository Structure

```
├── FINALCUOICUNG_reconstructed.ipynb   # Main notebook (fully executed)
├── safety_monitoring.csv               # Dataset
├── requirements.txt                    # Python dependencies
└── README.md
```

## ⚙️ Setup & Run

```bash
# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook FINALCUOICUNG_reconstructed.ipynb
```

## 📦 Requirements

See `requirements.txt` for full dependency list. Key libraries:
- `scikit-learn` — ML models & evaluation
- `imbalanced-learn` — SMOTE
- `pandas`, `numpy` — Data processing
- `matplotlib`, `seaborn` — Visualization
