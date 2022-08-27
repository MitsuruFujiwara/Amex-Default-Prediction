# Amex-Default-Prediction
This repository is my solution for kaggle [American Express - Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction).

### Result
- 85th place on public LB (score: 0.80040). 
- 125th place on private LB (score: 0.80761).

### Data
- Use [clean data](https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format) by raddar.

### Main Features
- Train models with data before aggregation (no_agg models) and then use their prediction as a feature (pred_no_agg features).
- Basic stat fearues (mean, std, min, max, sum, last, first).
- Difference between mean & last features.

### Cross Validation
- 5 fold stradified group k-fold (stradified by target & grouped by customer_ID) for no-agg models.
- 5 fold stratified k-fold for agg models.

### Ensemble
- Seed averaging with seed 42, 52, 62.
- Weighted average LightGBM, CatBoost, XGBoost (weights are determined by maxmizing cv score).
- Blending [best public score notebook](https://www.kaggle.com/code/hikarutabata/exponential-ensemble) with weights 0.6, 0.4.

### Final Submission
I choose one with best public score and one with best cv score for final submission.
The result is as below.

||cv|public|private|
|:---:|:---:|:---:|:---:|
|cv best|0.798074|0.79828|0.80602|
|public best|???|0.80040|0.80761|