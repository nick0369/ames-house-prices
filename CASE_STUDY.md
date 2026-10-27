# Ames House Prices: from validation to a scored submission

## Question and context
Can structured home attributes produce a more useful price estimate than a constant baseline? This educational prototype simulates an analyst supporting a property research team. Stakeholders would include analysts, model reviewers and a valuation operations lead. The measurable success criterion is lower out-of-sample log-RMSE with reproducible data handling, not a claim of commercial ROI.

## Results — 28 October 2026
| Candidate | Five-fold OOF log-RMSE | OOF dollar MAE |
|---|---:|---:|
| Mean log-price | 0.39958 | $55,655 |
| Ridge, alpha 20 | 0.15133 | $17,634 |
| Huber gradient boosting | **0.13153** | **$15,200** |

The selected gradient boosting model received **0.12728 public log-RMSE** on Kaggle. Submission status: **Complete**. It predicted all 1,459 required test homes. This is one submission, not a winning result, medal or finished competition. The rolling practice competition remains open indefinitely. The secondary temporal stress test (947 earlier homes → 513 later homes) scored **0.13591** log-RMSE.

## Statistical interpretation
Log-space errors align with the official metric and reduce the influence of the largest dollar values. Use fold-fitted preprocessing to avoid validation leakage. The five shuffled fold scores for gradient boosting were 0.13250, 0.11287, 0.16462, 0.12768, 0.11313: material variation remains. Comparing candidates on identical folds reduces split noise but does not establish statistical significance. The temporal check tests a different split assumption; it is not a fresh external validation. The public leaderboard score must not be treated as an unbiased final production estimate.

## Pipeline
Official competition CSVs → unique/disjoint ID and positive-target checks → row-local feature derivation → fold-fitted imputation, scaling and encoding → model comparison → temporal stress test → full-training refit → positive/finite 1,459-row CSV with exact expected IDs. Raw data and row-level prediction files are excluded from the public repository under competition data restrictions.

## Recommendations
Use this result as evidence of reproducible modeling and careful validation. Before any operational valuation, obtain current licensed data, validate by location and time, review expensive-home and rare-category errors, estimate prediction intervals and monitor drift. Do not infer causal renovation benefits from predictive variables. This historical US dataset is unsuitable for present-day Indian price estimates.

## Interview explanation
Explain why the ID is dropped, why the target is logged, why preprocessing belongs inside each fold, why public leaderboard and local validation differ, and why a temporal holdout is useful. Prepared with AI coding assistance; independently review the code before presenting it as a skill demonstration.

## Evidence
- [Official competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- [Computed metrics](reports/metrics.json)
- [Aggregate out-of-fold error by year](reports/oof_error_by_year.csv)
- Submission proof is provided on the portfolio case study; account owners can also view the Kaggle Submissions tab.
