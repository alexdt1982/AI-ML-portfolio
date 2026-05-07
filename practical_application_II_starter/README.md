# Used Car Price Analysis

## Overview

This project analyzes a dataset of used vehicles from Kaggle in order to identify the major factors that influence used car prices. The analysis follows the CRISP-DM framework and applies machine learning regression models to better understand depreciation trends and consumer valuation patterns in the used car market.

The primary goal of this project is to provide insights that could help a used car dealership improve pricing strategies and optimize inventory decisions.

---

# Business Understanding

From a business perspective, the goal of this project is to identify the key drivers of used car prices and determine which vehicle characteristics contribute most strongly to vehicle prices.

From a data science perspective, this problem can be treated as a supervised machine learning and regression task focused on predicting vehicle prices using features such as:
- vehicle year
- odometer mileage
- vehicle condition
- and other vehicle attributes

The final goal is to generate actionable insights that help dealerships make more informed inventory and pricing decisions.

---

# Data Understanding

The dataset contains information about used vehicles, including:
- price
- year
- odometer readings
- manufacturer
- fuel type
- transmission
- vehicle type
- and additional vehicle metadata

Initial exploration:
- inspecting dataset structure and data types,
- identifying missing values,
- analyzing summary statistics,
- detecting outliers and unrealistic values,
- and exploring relationships between variables.

---

# Data Preparation

Several preprocessing steps were performed before modeling:

- Removed duplicate entries
- Removed missing target values
- Filtered unrealistic price values
- Filtered unrealistic odometer readings
- Selected numerical features for regression modeling
- Applied scaling and preprocessing pipelines for machine learning workflows

The analysis primarily focused on numerical variables such as:
- `year`
- `odometer`

---

# Modeling

Several regression models were tested, including:

- Linear Regression
- Ridge Regression
- Decision Tree Regression
- Random Forest Regression

Cross-validation was used to compare model performance and evaluate generalization capability. Metrics such as:
- RMSE
- MAE
- and R²

were used to evaluate predictive accuracy.

---

# Evaluation

Among the tested models, the Decision Tree Regressor produced the strongest overall performance.

This suggests that used car pricing depends on nonlinear relationships between variables such as:
- vehicle age
- mileage
- and depreciation behavior

Unlike linear models, Decision Trees are able to capture more complex pricing patterns and conditional relationships within the data.

The analysis showed that:
- newer vehicles generally retain higher value,
- lower mileage vehicles tend to be priced significantly higher,
- and depreciation increases as mileage and age rise.

While the models produced meaningful results, future improvements could include:
- incorporating categorical variables,
- additional feature engineering,
- and more extensive hyperparameter tuning.

---

# Key Findings

## Major Drivers of Price

The strongest pricing factors identified were:
- vehicle year
- odometer mileage

### General Trends
- Newer vehicles tend to have higher prices
- Higher mileage vehicles tend to depreciate more heavily
- Nonlinear regression models performed better than simple linear models

---

# Business Recommendations

Based on the findings, the following recommendations are suggested for used car dealerships:

1. Prioritize newer vehicles with lower mileage when selecting inventory.

2. Monitor depreciation trends carefully when pricing older or high-mileage vehicles.

3. Use data-driven pricing strategies rather than relying solely on manual estimation.

4. Expand future analyses by incorporating additional vehicle characteristics such as manufacturer, condition, transmission type, and fuel type.

---



# Machine Learning Techniques

- Regression Modeling
- Cross-Validation
- Hyperparameter Tuning
- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)

---

# Conclusion

This project demonstrates how machine learning and data analysis techniques can be used to better understand the used car market and identify major pricing drivers. The findings provide meaningful insights into depreciation trends and consumer valuation behavior while also highlighting opportunities for future model improvement. 
Additional improvements could likely be achieved by incorporating categorical variables such as manufacturer, condition, and transmission type in future iterations of the project.
