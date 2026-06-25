# Bank Marketing Classifier Comparison


Alex Tran

## Overview

This project compares the performance of four supervised machine learning classification algorithms on the UCI Bank Marketing dataset:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)

The objective is to predict whether a customer will subscribe to a term deposit based solely on customer demographic and financial information.

This project was completed as part of the Practical Application III assignment for the Machine Learning and AI Professional Certificate.

---

## Business Objective

Banks invest significant resources into telemarketing campaigns. Predicting which customers are most likely to subscribe to a term deposit allows the bank to:

- Improve marketing efficiency
- Reduce unnecessary customer contacts
- Increase campaign conversion rates
- Support data-driven marketing decisions

This project evaluates multiple classification models to determine which performs best for this prediction task.

---

## Dataset

Source:

- UCI Machine Learning Repository
- Portuguese Bank Marketing Dataset

Dataset Characteristics:

- 41,188 customer records
- 20 predictor variables
- Binary target variable:
  - **yes** – customer subscribed
  - **no** – customer did not subscribe

The project uses only the customer information features:

- age
- job
- marital
- education
- default
- housing
- loan

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## Project Workflow

### 1. Data Exploration

- Loaded dataset
- Examined feature types
- Checked for missing values
- Identified "unknown" categorical values
- Reviewed class imbalance

### 2. Feature Engineering

- Selected customer information features
- One-hot encoded categorical variables
- Prepared target labels
- Performed train/test split

### 3. Baseline Model

Established a majority-class baseline to determine the minimum performance benchmark.

### 4. Initial Model Comparison

Trained and evaluated:

- Logistic Regression
- KNN
- Decision Tree
- Support Vector Machine

Compared:

- Training time
- Training accuracy
- Test accuracy

### 5. Hyperparameter Tuning

Used GridSearchCV to tune:

- Logistic Regression
  - C
  - class_weight
- KNN
  - number of neighbors
  - weighting scheme
- Decision Tree
  - maximum depth
  - minimum samples split
- SVM
  - C
  - kernel
  - class_weight

Models were re-evaluated after tuning.

---

## Results

The notebook compares each classifier using:

- Training Time
- Training Accuracy
- Test Accuracy
- F1 Score

The results demonstrate the tradeoff between model complexity, training time, and predictive performance.



## Key Takeaways

- Logistic Regression provided a strong baseline while remaining fast and interpretable.
- Decision Trees achieved high training accuracy but were more susceptible to overfitting.
- KNN performance depended heavily on the number of neighbors selected.
- SVM required the longest training time due to its computational complexity.
- Hyperparameter tuning improved certain models while illustrating the tradeoffs between accuracy and model complexity.

---

## Future Improvements

Potential enhancements include:

- Incorporating campaign-related features
- Addressing class imbalance with SMOTE or undersampling
- Comparing ensemble methods such as Random Forest and Gradient Boosting
- Evaluating models using ROC-AUC and Precision-Recall curves
- Performing additional feature engineering



