# Credit Card Default Prediction

University project — Politecnico di Torino
Author: Luca Camillini

## Overview

This project tackles credit card default prediction as a binary classification task (`default.payment.next.month` ∈ {0, 1}), evaluated with **Macro F1** to equally weight the default and non-default classes. The development set contains 24,000 samples (80/20 train/eval split), with a class imbalance of 77.9% non-defaulters vs. 22.1% defaulters (3.52:1).

The final submitted model is a **tuned Random Forest** trained on domain-specific engineered features (S3 configuration), achieving a cross-validated **Macro F1 of 0.707**, a 12.6% relative improvement over the logistic regression baseline (0.628).

## Data

23 original features:
- Demographics: `SEX`, `AGE`, `EDUCATION`, `MARRIAGE`, `LIMIT_BAL`
- Repayment status (6 months): `PAY_0` to `PAY_6`
- Bill amounts (6 months): `BILL_AMT1` to `BILL_AMT6`
- Payment amounts (6 months): `PAY_AMT1` to `PAY_AMT6`

No external data was used, and no missing values were found.

## Preprocessing

- **Data cleaning**: undocumented `EDUCATION` codes (0, 5, 6) remapped to category 4 ("other"); `MARRIAGE` code 0 remapped to 3 ("other").
- **Ambiguous PAY_X codes** (`-2`, `0`): kept as-is, since they carry predictive signal and are handled correctly by tree-based models.
- **Class imbalance**: handled via `class_weight='balanced'` (factor 3.52). SMOTE was also tested but performed worse than feature engineering alone.

## Feature Engineering (S3)

15 domain-specific features were derived and appended to the original 23, for a total of 38 features:

| Feature | Description |
|---|---|
| `UTIL_RATE` | `BILL_AMT1 / (|LIMIT_BAL| + 1)` |
| `PAY_RATIOi` | `PAY_AMTi / (|BILL_AMTi| + 1)`, for i = 1..6 |
| `MAX_DELAY` | max(PAY_0, ..., PAY_6) |
| `AVG_DELAY` | mean(PAY_0, ..., PAY_6) |
| `N_DELAYED` | number of months with delay > 0 |
| `DELAY_TREND` | PAY_0 − PAY_6 |
| `BILL_TREND` | BILL_AMT1 − BILL_AMT6 |
| `TOTAL_BILL` / `TOTAL_PAY` | sum of BILL_AMTi / PAY_AMTi |
| `AVG_PAY_RATIO` | TOTAL_PAY / (|TOTAL_BILL| + 1) |

An additional **S3+OHE** variant one-hot encodes `SEX`, `EDUCATION`, and `MARRIAGE` (41 features total), but this had negligible effect on tree-based models (ΔMacro F1 < 0.001).

## Model Selection

Six classifiers were benchmarked on S3 with 5-fold stratified CV:

| Model | Macro F1 | F1 Default | ROC-AUC |
|---|---|---|---|
| Gradient Boosting | 0.684 | 0.477 | 0.779 |
| **Random Forest** | 0.679 | 0.467 | 0.770 |
| Logistic Regression | 0.677 | 0.525 | 0.758 |
| KNN (k=7) | 0.658 | 0.437 | 0.724 |
| SVM (linear) | 0.641 | 0.398 | 0.758 |
| Naïve Bayes | 0.397 | 0.398 | 0.737 |

Random Forest was selected for tuning: it is 16x faster than Gradient Boosting (10s vs. 166s per CV) for only a 0.005 drop in Macro F1.

## Hyperparameter Tuning

- **Bayesian optimization** with Optuna's TPE sampler, 5-fold stratified CV:
  - Random Forest: 100 trials on S3+OHE → Macro F1 0.679 → **0.707**
  - XGBoost: 50 trials on S3 → Macro F1 0.686 → 0.705

| Parameter | Default | Tuned |
|---|---|---|
| n_estimators | 300 | 650 / 350 |
| max_depth | None | 35 / 38 |
| min_samples_split | 2 | 9 / 9 |
| min_samples_leaf | 1 | 7 / 5 |
| max_features | auto | sqrt |
| class_weight | None | balanced |

- **Threshold calibration**: a CV sweep over t ∈ [0.25, 0.65] (step 0.005) found an optimal threshold t* = 0.533 for the tuned RF (S3+OHE), giving a further +0.003 → Macro F1 = 0.711.

## Results

| Phase | Configuration | Macro F1 |
|---|---|---|
| Baseline | LR, no balancing | 0.628 |
| Balancing | LR + balanced weights | 0.631 |
| SMOTE | LR + SMOTE + balanced | 0.638 |
| Feature eng. (S3) | LR + S3 + balanced | 0.677 |
| Model selection | Random Forest, S3 | 0.679 |
| Model selection | Gradient Boosting, S3 | 0.684 |
| Tuning | XGB (S3, Optuna, t=0.5) | 0.705 |
| Tuning | RF (S3, Optuna, t=0.5) | 0.707 |
| OHE + t-calib. | RF (S3+OHE, t=0.533) | 0.711 |
| **Submitted** | **RF (S3, t=0.5)** | **0.707** |

Although the S3+OHE configuration with calibrated threshold (t=0.533) scored highest on CV (0.711), it generalized worse on the held-out evaluation set. The simpler **S3 + t=0.5** Random Forest was therefore chosen as the final submission.

## Key Findings

- **Feature engineering is the dominant lever**: +0.046 Macro F1, more than 6x the gain from model selection alone, and comparable to the gain from hyperparameter tuning (+0.029).
- **OHE is negligible for tree-based models**: Random Forests find optimal splits on ordinal categoricals regardless of encoding.
- **CV score doesn't always predict submission performance**: the highest-CV configuration overfit slightly to the development set, likely due to threshold optimization on the same data used for model selection, and extra splits introduced by OHE.

## Limitations & Future Work

- Recall on the minority (Default) class is ~0.57, with F1 ≈ 0.54 — the task remains intrinsically difficult.
- Possible improvements: ensemble stacking (RF + XGB + LR meta-learner), LightGBM, RobustScaler for skewed bill/payment columns, and additional interaction features (e.g., polynomial PAY_0 terms, cross-month ratios).

## References

1. I.-C. Yeh and C.-h. Lien, "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients," *Expert Systems with Applications*, vol. 36, no. 2, pp. 2473–2480, 2009.
2. N. V. Chawla et al., "SMOTE: Synthetic minority over-sampling technique," *JAIR*, vol. 16, pp. 321–357, 2002.
3. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
4. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *KDD*, pp. 785–794, 2016.
5. T. Akiba et al., "Optuna: A next-generation hyperparameter optimization framework," *KDD*, pp. 2623–2631, 2019.
