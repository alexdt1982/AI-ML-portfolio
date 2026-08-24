## Detecting Credit Card Fraud with Machine Learning

**Alex Tran**

### Executive summary

**Project overview and goals:** The goal of this project is to identify patterns in fraudulent credit card activity and to build a machine learning model that can accurately flag fraudulent transactions before they cause financial loss. We train and tune four classification models — Logistic Regression, Decision Tree, K‑Nearest Neighbors, and Random Forest — and compare them using metrics that are appropriate for a **highly imbalanced** problem, where fraud is a tiny fraction of all activity. We then select the best model, tune its decision threshold to balance catching fraud against raising false alarms, and interpret it to identify *which* transaction characteristics most strongly signal fraud. The work is delivered as two notebooks: Part I (analysis and modeling) and Part II (evaluation and interpretation).

**Findings:** The best model for catching fraud is **Logistic Regression** with balanced class weights. It achieves the highest recall and the highest Precision‑Recall AUC (PR‑AUC) of the four models, meaning it correctly identifies the largest share of fraudulent transactions — the outcome that matters most, because a missed fraud is the most expensive error. Its tradeoff is lower precision at the default threshold (more false alarms), but this is tunable: raising the decision threshold trades a small amount of recall for a large gain in precision. A small subset of the anonymized features carries almost all of the predictive signal, and transaction amount is a weaker but real indicator (fraud skews toward smaller "test" charges).

Representative model comparison (test set):

| Model | Recall | Precision | F1 | ROC‑AUC | PR‑AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **1.00** | 0.24 | 0.38 | 0.999 | **0.93** |
| K‑Nearest Neighbors | 0.79 | 0.93 | 0.85 | 0.989 | 0.87 |
| Random Forest | 0.44 | 0.92 | 0.60 | 0.997 | 0.74 |
| Decision Tree | 0.65 | 0.22 | 0.32 | 0.824 | 0.36 |

**Results and conclusion:** With balanced class weights at the default 0.5 threshold, the Logistic Regression model catches essentially all fraud but raises a number of false alarms (high recall, lower precision). By tuning the decision threshold, the model reaches a far more balanced operating point — representatively about **0.98 precision at 0.79 recall (F1 ≈ 0.87)**. Interpreting the model's coefficients shows that a handful of the anonymized PCA features — consistently **V14, V12, V10, V17, V4, and V11** on the real data — account for most of the separation between fraudulent and legitimate transactions. These same features surface both in the correlation analysis and in the model's learned weights, and they dominate the explanation of individual fraud predictions.

**Future research and development:** The clearest opportunity is to push recall and precision higher simultaneously. This can be pursued by adding resampling methods such as SMOTE (via the `imbalanced-learn` library) and comparing them against class weighting; by introducing **gradient‑boosted trees** (XGBoost, LightGBM), which frequently top the leaderboard on this dataset; and by ensembling the linear model with tree‑based models.

**Next steps and recommendations:** We recommend (1) **engineering time‑based behavioral features** — such as transactions per card per hour or rolling averages of amount — to capture bursts of activity that a per‑transaction model cannot see; (2) **optimizing the decision threshold against real dollar costs**, choosing the cutoff that minimizes total expected cost (cost of missed fraud × missed fraud + cost per false alarm × false alarms) rather than maximizing F1; and (3) **monitoring for drift**, since fraud tactics evolve and a production model should be retrained and re‑evaluated regularly.

### Rationale

Credit card fraud is a large and growing cost to the financial system. When fraudulent transactions are not caught quickly, organizations can suffer direct financial losses, damaged customer trust, and higher operational costs from disputes/investigations. Individual cardholders bear the stress and inconvenience of stolen funds and cancelled cards. Detecting fraud faster and more accurately reduces losses, protects customers, and lets fraud teams focus their limited attention on the transactions most likely to be fraudulent. A machine learning approach is well suited to this because fraud leaves subtle, multi‑feature patterns that are difficult to capture with fixed hand‑written rules.

### Research Question

How can machine learning models be used to identify patterns in fraudulent financial activity such as credit card fraud — and which model and which transaction characteristics are most effective for detecting it?

### Data Sources

**Dataset:** The dataset is sourced from Kaggle and can be accessed at https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.

The data are real credit card transactions made by European cardholders over two days in September 2013. The original dataset has **284,807 transactions** and 31 columns: `Time` (seconds elapsed since the first transaction), `V1`–`V28` (anonymized features produced by a **PCA transformation** to protect confidentiality, so their original meaning is hidden), `Amount` (the transaction value), and `Class` (the target label: `1` = fraud, `0` = legitimate).

**Exploratory data analysis:** The defining characteristic of this data is **extreme class imbalance** — only **492 of the 284,807 transactions are fraudulent (~0.172%)**. This fact drives every modeling decision: a model that predicts "never fraud" would be ~99.8% accurate while catching zero fraud, so **accuracy is not a useful metric**. There are no missing values. Fraudulent transactions tend toward smaller amounts than legitimate ones (consistent with card‑testing behavior), and a correlation analysis against the `Class` label shows that a small group of the PCA features (notably V14, V17, V12, V10, V11, V4) are far more associated with fraud than the rest. These EDA visualizations — the class‑count bar chart (plotted on a log scale so the tiny fraud bar is visible), amount and time distributions by class, and the feature‑correlation chart — are generated in Part I of the notebooks.

**Cleaning and preparation:** The dataset is already clean (no nulls, all numeric). Two preparation steps are applied. First, `Time` and `Amount` are **standardized** to the same scale as the already‑scaled PCA features, because distance‑ and gradient‑based models (KNN, Logistic Regression) are sensitive to feature scale; standardization is done inside each model pipeline. Second, the data is split into training and test sets using a **stratified** 75/25 split so that both sets preserve the rare fraud proportion — without stratification, a random split could leave too few fraud cases in the test set to evaluate reliably.

**Handling class imbalance:** Two complementary techniques address the imbalance. Logistic Regression, Decision Tree, and Random Forest use `class_weight="balanced"`, which penalizes mistakes on the rare fraud class more heavily. K‑Nearest Neighbors, which has no class‑weight option, is trained on a more balanced subset created by keeping every fraud case and randomly undersampling legitimate cases.

### Methodology

**Holdout cross‑validation** is used: each model is tuned on the training set with `RandomizedSearchCV` (5‑fold, stratified) and then evaluated once on the untouched test set. The tuning objective is **average precision (PR‑AUC)**, which summarizes the precision/recall tradeoff and is the appropriate objective when the positive class is rare. Models are compared on **recall, precision, F1, ROC‑AUC, and PR‑AUC** — deliberately *not* accuracy.

Four models were trained and tuned:

- **Logistic Regression** — a pipeline standardizes the features and fits a Logistic Regression with balanced class weights; `RandomizedSearchCV` tunes the regularization strength `C`.
- **Decision Tree** — a single tree with balanced class weights; the search tunes the split criterion (gini/entropy), maximum depth, and minimum samples to split.
- **K‑Nearest Neighbors** — features are standardized and the model is trained on a balanced (undersampled) training set; the search tunes the number of neighbors.
- **Random Forest** — an ensemble of trees with balanced class weights; the search tunes the number of trees, maximum depth, and minimum samples to split. This is the slowest model to train.

The best model is then re‑fit in Part II, where its **decision threshold is tuned** (a business lever unique to probabilistic classifiers) and its coefficients are interpreted globally and locally.

### Model evaluation and results

Model performance is compared with confusion matrices and ROC / Precision‑Recall curves (all generated in the notebooks). Under severe imbalance the **Precision‑Recall curve** is the more honest view because it focuses on the rare fraud class. The two error types carry very different business costs: a **false negative** (missed fraud) lets a fraudulent charge through, while a **false positive** (false alarm) inconveniences a legitimate customer and consumes investigator time.

**Logistic Regression** is selected as the best model. Representatively it reaches a PR‑AUC of ~0.93 and catches essentially all fraud in the test set (recall ≈ 1.0) at the default threshold, at the cost of precision. Because it is a linear model with standardized inputs, it is also the most **interpretable** — its coefficients directly reveal which features push a transaction toward fraud, which is exactly what the research question asks for.

**Decision threshold tuning** is the key evaluation step. A classifier outputs a fraud *probability*; the 0.5 cutoff is only a default. Sweeping the threshold and choosing the point that maximizes F1 moves the model to a balanced operating point (representatively ~0.98 precision at ~0.79 recall). In practice a bank that must not miss fraud would keep the threshold low (high recall), while a team with limited investigators would raise it (higher precision) — the model supports either choice.

**Model interpretation** answers the research question about patterns. A **global** analysis of the coefficients shows that a small set of PCA features (V14, V12, V10, V17, V4, V11 on the real data) carry almost all of the fraud signal. A **local** analysis of individual predictions breaks a single transaction's fraud score into per‑feature contributions and confirms that these same features drive individual fraud flags, while amount contributes more weakly. A detailed walkthrough of this interpretation is in Part II of the notebooks.

### Outline of project

- [Part I — Analysis & Modeling notebook](./CreditCardFraudDetection.ipynb)
- [Part II — Evaluation & Interpretation notebook](./CreditCardFraudEvaluation.ipynb)
- [Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — download `creditcard.csv` and place it in the same folder as the notebooks before running.

**To reproduce:** download `creditcard.csv` from the Kaggle link, put it in the data folder beside the src folder, open Part I and choose *Run → Run All Cells*, then do the same for Part II. All figures and metrics regenerate automatically.

### Contact and Further Information

Nam (Alex) Tran

Email: ktran945@gmail.com

[LinkedIn](https://www.linkedin.com/in/alexdtran82/)
