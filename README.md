# Ames House Prices — reproducible regression benchmark

An independent educational entry for Kaggle's **House Prices – Advanced Regression Techniques**. Predict the sale price of 1,459 held-out Ames, Iowa homes using 1,460 labelled examples and 79 original features. This is a historical learning benchmark, not a production valuation service or an employment project.

## Kaggle notebook

[Open the competition-linked Kaggle notebook](https://www.kaggle.com/code/nikhilchandekar/ames-house-prices-validation-and-huber-boosting). The `.ipynb` in this repository contains the same workflow, adapted to Kaggle input/output paths. Kaggle public notebook licensing is displayed on Kaggle; repository source remains MIT licensed.

## Reproduce

1. Join/read the [competition rules](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/rules) under your own Kaggle account.
2. Download the official files from the [data page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). Unzip into `data/` locally. Raw competition data is deliberately excluded from this repository.
3. With Python 3.12:

```sh
python -m venv .venv
# Activate the environment for your OS, then:
python -m pip install -r requirements.txt
python train.py --data data --out reports
```

The script writes `reports/metrics.json`, aggregate out-of-fold errors by year, and a validated `reports/submission.csv`. Upload the submission from your own competition account. No API credentials are stored or required by this code.

## Method

- Deterministic 5-fold shuffled cross-validation, seed 42, identical folds for all candidates.
- Mean-log-price baseline, ridge regression (alpha 20), and Huber-loss gradient boosting (650 trees, learning rate .035, depth 3, minimum leaf 8).
- Target uses natural `log(SalePrice)`, matching Kaggle's log-RMSE metric; inverse predictions use `exp`.
- Drop row identifier, treat building subclass/month as categorical, derive total area, bathrooms, age and time since remodeling. Features are row-local and use no target statistics.
- Median numeric imputation with missing indicators, constant categorical imputation, scaling and one-hot encoding are fitted separately inside each training fold. Unknown categories are ignored at inference.
- Select one model by pooled out-of-fold log-RMSE. A separate 2006–2008 → 2009–2010 evaluation probes time generalization. This is a secondary stress test, not nested validation or a guarantee of future performance.
- Fit the selected model on all training examples only; preserve all 1,459 test IDs and validate finite positive predictions before export.

## Interpretation and limits

See [computed results](reports/metrics.json) and [case study](CASE_STUDY.md). Lower log-RMSE is better; it is not classification accuracy. Dollar MAE is descriptive and affected by expensive homes. No test labels, hand-labeling, external price lookup or leaderboard-driven tuning are used. Five-fold model selection creates some optimism; uncertainty requires repeated/nested validation and further data.

This benchmark uses historical homes in one US city. It cannot support present-day Indian home valuations, causal renovation ROI, lending decisions, or realised financial savings. A real deployment would need recent data, geographic/time holdouts, fairness and error-segment review, uncertainty estimation, drift monitoring and licensed data access.

## Code assistance and attribution

Prepared with AI coding assistance at Nikhil Chandekar's request. Review and understand the feature/validation choices before presenting this work in interviews. The course completion badge is a separate credential, not a certification awarded for this competition.

Data: Dean De Cock, Ames Housing; competition citation: Ana Lina Morelia and DataCanary, *House Prices – Advanced Regression Techniques*, Kaggle, 2016. [Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). Code is MIT licensed; competition data remains subject to Kaggle's rules and is not redistributed.
