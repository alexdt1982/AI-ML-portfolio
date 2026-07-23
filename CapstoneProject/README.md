# Baseline Model: Logistic Regression

## Model Overview

A Logistic Regression classifier was selected as the baseline machine learning model for this project due to its simplicity, interpretability, and effectiveness for binary classification problems. The objective was to classify financial transactions as either legitimate or fraudulent using engineered transaction, customer, and behavioral features.

The model was trained using a stratified 80/20 train-test split to preserve the highly imbalanced class distribution present in the dataset. To mitigate the effects of class imbalance during training, the classifier was configured with `class_weight='balanced'`. Numerical features were standardized, while categorical features were one-hot encoded using a preprocessing pipeline.

## Evaluation Metrics

The model was evaluated using several performance metrics:

| Metric | Score |
|---------|------:|
| Accuracy | **87.76%** |
| Precision | **3.50%** |
| Recall | **75.95%** |
| F1 Score | **6.70%** |
| ROC-AUC | **91.17** |


## Key Findings

The exploratory data analysis revealed that the dataset is **highly imbalanced**, with fraudulent transactions representing less than 1% of all observations. Because of this imbalance, overall accuracy alone is not an appropriate measure of model performance. A classifier that predicts every transaction as legitimate would still achieve very high accuracy while failing to detect fraudulent activity.

Instead, greater emphasis was placed on precision, recall, F1-score, and ROC-AUC:

The confusion matrix further illustrates the trade-off between correctly identifying fraudulent transactions and minimizing false positive alerts.

## Conclusion

The Logistic Regression model provides a strong and interpretable baseline for fraud detection. While its performance is constrained by the severe class imbalance inherent in real-world financial transaction data, it establishes a meaningful benchmark for future comparison.

Overall, this baseline demonstrates the importance of evaluating fraud detection systems using metrics beyond accuracy and highlights the challenges posed by highly imbalanced datasets.

## Lessons Learned

Through this project, I gained practical experience with the complete machine learning workflow, including data preprocessing, feature engineering, exploratory data analysis, handling class imbalance, building preprocessing pipelines, and training a baseline classification model. One key takeaway is that high accuracy does not necessarily indicate a good fraud detection model; selecting appropriate evaluation metrics such as precision, recall, F1-score, and ROC-AUC is essential when working with imbalanced datasets. This project also reinforced the importance of thoughtful feature engineering and reproducible machine learning pipelines for building reliable predictive models.
