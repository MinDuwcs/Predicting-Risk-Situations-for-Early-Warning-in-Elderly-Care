# Phase 1 — Data Leakage Analysis Report

## Summary

4 post-event features were identified and removed from the dataset.
Correcting SMOTE placement (within-fold instead of before-split) revealed
that original notebook metrics were significantly inflated.

---

## 1. Post-Event Features Identified

### 1.1 Post-Fall Inactivity Duration (Seconds)

**Leakage type:** Perfect target separator

| Fall Detected | Duration = 0 | Duration > 0 |
|---|---|---|
| No (9,501) | 9,501 | 0 |
| Yes (499) | 0 | 499 |

**Correlation with target:** 1.000 (perfect)
**Evidence:** This column is only populated AFTER a fall occurs. A real-time monitoring system cannot know the inactivity duration until AFTER the fall has already been detected.

### 1.2 Alert Triggered (Yes/No)

**Leakage type:** Post-event response feature

| Fall Detected | Alert = No | Alert = Yes |
|---|---|---|
| No (9,501) | 9,501 | 0 |
| Yes (499) | 150 | 349 |

**Evidence:** Alerts are triggered as a RESPONSE to a detected fall. No non-fall record has Alert = Yes.

### 1.3 Caregiver Notified (Yes/No)

**Leakage type:** Post-event response feature

| Fall Detected | Notified = No | Notified = Yes |
|---|---|---|
| No (9,501) | 9,501 | 0 |
| Yes (499) | 150 | 349 |

**Evidence:** Identical pattern to Alert Triggered. Caregiver notification is downstream of fall detection.

### 1.4 Impact Force Level

**Leakage type:** Post-event measurement

| Fall Detected | Impact = "-" | Impact = Low/Med/High |
|---|---|---|
| No (9,501) | 9,501 | 0 |
| Yes (499) | 0 | 499 |

**Evidence:** Impact force is only measured during a fall event. ALL non-fall records have "-" (missing), ALL fall records have a value. This is a perfect binary separator when encoded.

> **Note:** The original notebook removed this column but classified it as "non-informative" rather than as a leaked feature. It is in fact a post-event feature with perfect separation.

---

## 2. SMOTE Placement Fix

### Original Notebook (Incorrect)
```
Full dataset (10,000) → SMOTE (499→9,501 = 19,002) → Train/Test split
```
**Problem:** Synthetic minority samples generated from the same source records appear in both train and test sets. The model effectively "sees" test data during training.

### Phase 1 (Correct)
```
Full dataset (10,000) → Train/Test split (8,000/2,000) → SMOTE on train ONLY within each CV fold
```
**Fix:** SMOTE applied inside each CV fold's training portion. Validation/test data is always original, unaugmented data.

---

## 3. Metric Comparison

### Cross-Validation Results (SMOTE-within-fold, 5-fold stratified)

| Rank | Model | F1 (mean±std) | AUC (mean±std) |
|---|---|---|---|
| 🥇 | Naive Bayes | 0.3333 ± 0.0061 | 0.8933 ± 0.0047 |
| 🥈 | Gradient Boosting | 0.3316 ± 0.0068 | 0.8888 ± 0.0073 |
| 🥉 | Random Forest | 0.2962 ± 0.0255 | 0.8754 ± 0.0110 |
| 4 | Decision Tree | 0.2935 ± 0.0289 | 0.7476 ± 0.0196 |
| 5 | Logistic Regression | 0.1540 ± 0.0124 | 0.6335 ± 0.0031 |

### Held-Out Test Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Naive Bayes | 0.8000 | 0.2000 | 1.0000 | 0.3333 | 0.9007 |
| Gradient Boosting | 0.8045 | 0.1989 | 0.9600 | 0.3293 | 0.8959 |
| Decision Tree | 0.8645 | 0.2082 | 0.6100 | 0.3104 | 0.7569 |
| Random Forest | 0.8600 | 0.2039 | 0.6200 | 0.3069 | 0.8710 |
| Logistic Regression | 0.5795 | 0.0879 | 0.7900 | 0.1582 | 0.6351 |

### Original Notebook vs Phase 1

| Model | Original F1 | Phase 1 F1 | Change |
|---|---|---|---|
| Random Forest | 0.9275 | 0.3069 | **-0.6206** |
| Decision Tree | 0.8973 | 0.3104 | **-0.5869** |
| Gradient Boosting | 0.8953 | 0.3293 | **-0.5660** |
| Naive Bayes | 0.8937 | 0.3333 | **-0.5604** |
| Logistic Regression | 0.6998 | 0.1582 | **-0.5416** |

---

## 4. Interpretation

The massive F1 drop (~0.55-0.62 points) confirms two compounding issues:

1. **Post-event features provided near-perfect separation.** Without them, the 4 remaining pre-event features (Movement Activity, Location, Hour, Day_of_Week) have very weak individual predictive power for fall detection.

2. **SMOTE before split inflated metrics further.** Even Scenario B's results (0.9275 F1) were inflated because synthetic samples from the same minority records leaked between train and test.

3. **Low precision across all models (~0.20)** reflects the fundamental difficulty: with only 5% prevalence and 4 near-uniform categorical features, models generate many false positives to catch true falls.

4. **AUC is much more stable** than F1 (only -0.01 to -0.16 change), suggesting models can rank-order risk but struggle with the classification threshold under extreme imbalance.

---

## 5. Recommendations for Phase 2

- **Feature engineering is critical** — current features are too weak for reliable classification
- Consider threshold tuning — AUC is reasonable (~0.87-0.90) but precision suffers at default 0.5 threshold
- Explore interaction features, temporal patterns, or additional sensor data
- Consider cost-sensitive learning as an alternative or complement to SMOTE
