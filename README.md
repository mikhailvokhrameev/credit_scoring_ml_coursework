# Credit Scoring Model Training Pipeline

This repository contains a project for developing an end-to-end ML pipeline for credit scoring based on the **Home Credit Default Risk** dataset: aggregation of seven relational tables, engineering of 300 features, training of four models with Bayesian optimization, tracking in MLflow, and interpretation via SHAP. As part of the project, **5 statistical hypotheses were also formulated and tested** regarding the nature of the data and the effectiveness of training methods.

<div align="center">
<img src="https://github.com/user-attachments/assets/504ad478-e1fe-4ab6-a9ac-82d41de85a78" width="250"><br>
</div>

---

## Brief Description

The project solves a binary classification problem — predicting the probability of borrower default, which is the likelihood that a client will delay payment by more than 90 days within a 12-month horizon. At the output, the model returns an estimate `f(x) ∈ [0, 1]`, which serves as the basis for the credit approval decision.

Key features:

- **Multi-module architecture** separating data loading, preprocessing, feature engineering, training, and interpretation.
- **Bank-grade reproducibility**: stratified Out-of-Fold cross-validation, tracking of every run in MLflow, model versioning in Model Registry.
- **Regulatory interpretability**: global and local SHAP explanations, avoiding "black box" logic.
- **Unified CLI**: launching the full training cycle of any model with a single command.
- **Testing 5 hypotheses**: statistical study of data nature, feature value, and training method efficiency.

---

## Motivation

I wanted to build a complete ML pipeline: from raw data to versioned models in the registry, with reproducible training and production-ready artifacts.

During this work, I gained experience in:

- designing **modular feature engineering** on top of a relational database of 7+ tables with transaction history aggregation into a single feature vector per borrower;
- controlling **data leakage** when preparing training and test sets;
- building a **unified OOP pipeline** based on an abstract base class, where any new algorithm is added with a single subclass;
- **Bayesian hyperparameter optimization** via Optuna;
- organizing **MLOps infrastructure**: experiment tracking, artifact logging, and model registry in MLflow;
- **model interpretation** via SHAP and formulating business explanations for credit refusal;
- working with **severe class imbalance**.

---

## Tech Stack

- **Python 3.10**
- **pandas, ydata-profiling, kaggle** — data and EDA
- **scikit-learn, LightGBM, XGBoost, CatBoost** — ML
- **optuna, optuna-dashboard** — hyperparameter optimization
- **mlflow** — experiment tracking
- **shap** — interpretability (XAI)
- **matplotlib, seaborn** — visualization
- **jupyterlab, nbconvert, quarto** — notebooks and reports
- **joblib** — model serialization

---

## Project Features

#### Data Loading and Caching
Upon first access, CSV files are converted into binary Apache Parquet format, significantly speeding up subsequent iterations when working with tables containing millions of rows.

#### Leakage-Free Preprocessing
Encoders and grouping of rare categories into "Other" are fitted strictly on the training sample and applied to test without data leakage.

#### Modular Feature Engineering
Each data source is processed separately: financial ratios, behavioral indicators, and statistical aggregates are generated. The final pipeline merges them via left-join by borrower ID and selects the top 300 features using Ridge regression.

#### Unified Model Training
An abstract base class establishes a single contract upon which four models are implemented: Logistic Regression, LightGBM, XGBoost, and CatBoost. Adding a new algorithm simply requires writing one subclass.

#### Bayesian Hyperparameter Optimization
Optuna maximizes the mean ROC-AUC across folds; trial history is stored in `optuna.db` (SQLite), enabling search resumption.

#### Experiment Tracking in MLflow
Nested run structure: parent Run → child HPO and Final_Production. Parameters, metrics, SHAP reports, and model dumps are logged. The best candidates are moved to the Model Registry.

<div align="center">
<img src="https://github.com/user-attachments/assets/ac4c1d9d-8be4-4f1f-b32f-f6e79e03962c"><br>
<em>MLflow UI Dashboard</em>
<img src="https://github.com/user-attachments/assets/1a23353e-d1cd-4bd0-87d5-b5d163a72fb7"><br>
<em>Run List</em> 
<img src="https://github.com/user-attachments/assets/1e46a9c3-15c5-4463-9481-e72939f3ac04"><br>
<em>Artifact Logging</em> 
<img src="https://github.com/user-attachments/assets/b0ce9a32-1767-431c-be0e-22ca49c46228"><br>
<em>Model Registry</em> 
</div>


#### Decision Interpretation (XAI)
After training, SHAP values are automatically computed (TreeExplainer for boostings, LinearExplainer for LogReg). Summary Bar, Beeswarm, and local Waterfall charts are generated for specific clients.

<div align="center">
<img src="https://github.com/user-attachments/assets/829e180e-bab0-41cf-a42e-02f3684623cb" width="600"><br>
<em>Prediction Explanation for a High-Risk Client</em>
</div>

---

## Jupyter Notebooks: Research Stages

Each stage corresponds to a dedicated notebook in [notebooks/](notebooks/). They serve as an interactive environment for experimentation and leverage functions from `src/`.

| Notebook | Content |
|---|---|
| [01_EDA.ipynb](notebooks/01_EDA.ipynb) | Data loading, HTML table profiling, analysis of imbalance, missing values, and anomalies |
| [02_feature_engineering.ipynb](notebooks/02_feature_engineering.ipynb) | Pipeline execution, analysis of selected features and their correlations |
| [03_modelling.ipynb](notebooks/03_modelling.ipynb) | Model comparison, HPO via Optuna, ROC/PR curves, threshold optimization |
| [04_interpretation.ipynb](notebooks/04_interpretation.ipynb) | Global and local SHAP, comparison of logic across models on a single client |

---

## Getting Started

### 1. Dependency Installation

```bash
git clone https://github.com/mikhailvokhrameev/credit_scoring_ml_coursework.git
cd credit_scoring_ml_coursework

conda create -n credit-scoring python=3.10 -y && conda activate credit-scoring

pip install -r requirements.txt
```

### 2. Data Downloading

Data is retrieved from the Kaggle competition [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk). Place CSV files in `data/raw/` (requires a configured `kaggle` API token):

```bash
kaggle competitions download -c home-credit-default-risk -p data/
unzip "data/*.zip" -d data/raw/
```

### 3. Feature Engineering

Full pipeline: preprocessing → feature generation across all tables → feature selection via Ridge → saving to Parquet.

```bash
python -m src.features.pipeline
```

**Output** (in `data/processed/`): `train_features.parquet`, `test_features.parquet`, `selected_features.json`, `feature_importance.csv`. These files serve as input for the training stage.

> For baseline comparison (main table only): `python -m src.features.pipeline_base`.

### 4. Model Training

Full training cycle with a single command via CLI:

```bash
python -m src.models.train --model lgbm --trials 30 --device gpu
```

Flags:

| Flag | Values | Default | Description |
|---|---|---|---|
| `--model` | `logreg` / `lgbm` / `xgb` / `catboost` | — (required) | Algorithm to train |
| `--trials` | int | `30` | Number of Optuna trials (`0` — skip HPO) |
| `--folds` | int | `5` | Number of final CV folds |
| `--device` | `cpu` / `gpu` | `cpu` | Computing device |

Can be run sequentially to train the entire suite of models:

```bash
python -m src.models.train --model logreg   --trials 30
python -m src.models.train --model lgbm     --trials 30 --device gpu
python -m src.models.train --model xgb      --trials 30 --device gpu
python -m src.models.train --model catboost --trials 30 --device gpu
```

**Output** (in `artifacts/<model>_model/`): `model.joblib` (full wrapper), `serving_model.joblib`, `metadata.json`, `xai/` (SHAP plots and `shap_feature_importance.csv`), `inference/thresholds.json`. Everything is simultaneously logged to MLflow.

> Quick draft model without HPO on a single table: `python -m src.models.train_base --model lgbm`.

### 5. Viewing Experiments in MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000 — model comparison by ROC-AUC, nested HPO/Final_Production runs, artifacts, and Model Registry.

### 6. Interpretation and Export

For deep SHAP analysis, use [04_interpretation.ipynb](notebooks/04_interpretation.ipynb).

---

## Hypotheses

During the study, five hypotheses were formulated and statistically tested, covering data nature, feature importance, and training method efficiency.

---

#### Hypothesis 1 — Age as a Risk Predictor

**Statement:** Default risk monotonically decreases with borrower age — younger clients systematically default more frequently, without dips within the age range.

**Testing Method:** The training sample is split into 10 equal deciles based on the `DAYS_BIRTH` feature; the default rate is calculated in each decile; monotonicity is evaluated using Spearman's rank correlation coefficient.

**Acceptance Criterion:** ρ < −0.7 and p-value < 0.05.

**Result: Confirmed.** ρ = −1.00, p ≈ 0. Default rate is 12.3% for the youngest vs. 3.73% for the oldest, with zero violations of monotonicity.

<div align="center">
<img src="https://github.com/user-attachments/assets/e8f7ee0c-d518-43de-a95e-62a07d75ab83" width="600"><br>
</div>

---

#### Hypothesis 2 — Missing Values as a Risk Signal

**Statement:** Defaulted clients have a significantly higher proportion of missing values in their application form than reliable clients — meaning missing values are not random and serve as a credit risk signal in themselves.

**Testing Method:** For each row in `application_train`, the proportion of missing values is calculated; default and reliable client groups are compared using the non-parametric Mann–Whitney U test.

**Acceptance Criterion:** p-value < 0.05 and the median proportion of missing values is higher for defaulted clients than for reliable ones.

**Result: Confirmed.** Median is 39.7% for defaulted vs. 28.1% for reliable clients; Mann–Whitney test p ≈ 0. Missing values are an informative predictor and are utilized as binary `is_missing` flags.

---

#### Hypothesis 3 — External Credit Scores as Top Predictors

**Statement:** Features `EXT_SOURCE_1/2/3` are the most informative predictors, as they aggregate ready-made credit history from external sources.

**Testing Method:** Evaluation of global feature importance via SHAP (`shap.TreeExplainer`); checking whether `EXT_SOURCE_1/2/3` or their derivatives are in the top 5.

**Acceptance Criterion:** `EXT_SOURCE_1/2/3` or their derivatives appear in the top 5 SHAP feature importances of the final model.

**Result: Confirmed.** The derived feature `EXT_SOURCES_MEAN` (mean of the three sources) ranked **1st** in the SHAP importance ranking.

<div align="center">
<img src="https://github.com/user-attachments/assets/28f71ec9-a480-4446-8b30-efdcdd1a1841" width="500"><br>
<em>SHAP Summary Bar</em>
</div>

---

#### Hypothesis 4 — Value of Auxiliary Tables

**Statement:** Using data from auxiliary tables (`bureau`, `previous_application`, etc.) significantly improves prediction quality compared to using only the main application form — by capturing the temporal dynamics of client behavior.

**Testing Method:** Comparison of ROC-AUC for two LightGBM models: trained on the full set of 7 tables (`lgbm_non_hpo`) and trained only on `application_train` (`lgbm_base`).

**Acceptance Criterion:** ROC-AUC gain ≥ 0.015.

**Result: Confirmed.** ROC-AUC increased from 0.766 (`lgbm_base`) to 0.786 (`lgbm_non_hpo`), a gain of **+0.020** — exceeding the threshold. Aggregated payment history contains critically important signal invisible in a static application form.

---

#### Hypothesis 5 — Efficiency of Bayesian Hyperparameter Optimization

**Statement:** Automated hyperparameter tuning via Optuna outperforms default parameters due to data specificity.

**Testing Method:** Comparison of ROC-AUC between `non_hpo` (default parameters) and `full` (30 Optuna trials) models on the complete dataset for each algorithm.

**Acceptance Criterion:** ROC-AUC gain ≥ 0.005 for at least one algorithm.

**Result: Partially Confirmed.** The gain depends on the algorithm:

| Algorithm | ROC-AUC (Non-HPO) | ROC-AUC (Full) | Gain |
|---|---|---|---|
| XGBoost | 0.767 | 0.788 | **+0.021** |
| CatBoost | 0.782 | 0.787 | **+0.005** |
| LightGBM | 0.786 | 0.786 | 0.000 |

---

## Results

Testing on real data confirmed the advantage of ensemble methods over linear ones. The best result in terms of pure performance is **`xgb_full`** (ROC-AUC = 0.788), while the best balance of performance and speed is **CatBoost**.

| Algorithm | Version | ROC-AUC | PR-AUC | Gini | Business Loss |
|---|---|---|---|---|---|
| **XGBoost** | Full | **0.788** | **0.281** | **0.576** | **149,582** |
| CatBoost | Full | 0.787 | 0.278 | 0.573 | 150,005 |
| LightGBM | Full | 0.786 | 0.278 | 0.571 | 150,627 |
| LightGBM | Non-HPO | 0.786 | 0.277 | 0.572 | 150,029 |
| LogReg | Full | 0.773 | 0.253 | 0.546 | 155,780 |
| LogReg | Base *(baseline)* | 0.751 | 0.235 | 0.502 | 165,541 |

<div align="center">
<img src="https://github.com/user-attachments/assets/ca246b5c-4787-44d0-801d-d5f5402801af" width="800"><br>
<em>Comparison of cv_mean_roc_auc across all models</em>
</div>

**Key Takeaways:**

- **Data enrichment matters more than algorithm complexity.** Transitioning from one table to seven yielded a ROC-AUC gain of +0.020, whereas transitioning from LogReg to XGBoost on full data yielded only +0.015.
- **PR-AUC is more sensitive than ROC-AUC** under class imbalance: range of 0.228–0.281 versus thousandths of a point in ROC-AUC, making it more informative for rare event tasks.
- All 5 hypotheses were confirmed.

---

## Future Roadmap

- **Stacking Ensemble**: meta-model (LogReg) on top of predictions from CatBoost + LightGBM + XGBoost to maximize ROC-AUC.
- **External File Configuration** (YAML/JSON) for flexible pipeline parameterization without code modification.
- **Data Drift Monitoring** to track performance degradation during distribution shifts (economic crises).
- **Probability Calibration** and score scaling to bank-standard range [0, 1000].

---

## Connection to the LoanSight Project

Models trained in this repository formed the core of **[LoanSight](https://github.com/mikhailvokhrameev/loan_sight.git)** — a web service for experimenting with credit risk assessment ML models.

LoanSight loads artifacts exported here, retrieves client data from a database, runs the selected model, and returns default probability along with a human-readable risk label (Low / Medium / High) and an interactive **SHAP waterfall chart**. The service supports side-by-side comparison of two models, manual feature overriding ("what-if" analysis), multi-currency, and experiment history.

Thus, **this repository is the research and training engine**, while LoanSight serves as the product frontend wrapper around the resulting models.
