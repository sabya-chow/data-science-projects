
# Credit Card Default Prediction Capstone Project

## Overview

This project aims to build a predictive model to assess the risk of credit card default for customers based on their payment history, purchase behavior, and other demographic and financial attributes. The primary goal is to help credit card companies identify customers at risk of default, improve risk management, and provide targeted customer interventions.

## Project Background

The credit card industry in India is expanding rapidly, with millions of new users adopting credit cards each year. However, with this growth comes the risk of default, which occurs when customers fail to meet the minimum payments on their credit card balances. Defaulting customers pose a significant financial risk to lenders, making it essential to accurately predict and manage these risks.

### Business Opportunity
By predicting the likelihood of default, credit card companies can:
- Identify high-risk customers and take preventive measures.
- Develop targeted interventions to reduce defaults.
- Improve overall portfolio health and profitability.
- Promote responsible lending practices.

### Social Opportunity
The project contributes to responsible lending practices by helping lenders make more informed decisions, ultimately improving customers' financial health and reducing defaults.

## Data Description

The dataset contains 99,978 customer records with 36 features, including demographic details, account balance information, payment history, and credit card purchase behavior.

### Key Features
- `userid`: Unique identifier for each customer.
- `default`: Target variable indicating whether a customer has defaulted (1 for default, 0 otherwise).
- `acct_amt_added_12_24m`: Total purchases made using the credit card between 24 to 12 months before the present date.
- `acct_days_in_dc_12_24m`: Total days the account stayed in debt collection status over the specified period.
- `max_paid_inv_0_12m`: Maximum credit card bill paid by the customer in the past year.
- `num_arch_dc_0_12m`: Number of archived purchases in debt collection status over the past year.
- Various other features related to customer behavior, merchant category, and transaction status.

## Project Workflow

1. **Problem Understanding and Data Preparation**
   - Problem definition and business opportunity exploration.
   - Data collection and descriptive statistics.
   - Data cleaning and preprocessing (missing value imputation, data type conversion, feature encoding, outlier treatment, etc.).
   
2. **Exploratory Data Analysis (EDA)**
   - Visual and statistical analysis of features.
   - Identification of key predictors for default.
   - Analysis of relationships and correlations.

3. **Model Building**
   - Initial modeling with various algorithms (Logistic Regression, Decision Trees, Random Forest, etc.).
   - Hyperparameter tuning and performance evaluation.
   - Addressing class imbalance using upsampling techniques.
   - Model optimization using GridSearchCV for hyperparameter tuning.

4. **Model Evaluation**
   - Assessment of model performance using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC curves.
   - Selection of the best-performing model based on business requirements and performance metrics.

5. **Model Interpretation**
   - Analysis of feature importance.
   - Business implications of key predictors.

## Key Results

- The **Random Forest** model with hyperparameter tuning emerged as the best-performing model with high accuracy and recall, effectively identifying customers at risk of default.
- Important predictors included:
  - `num_arch_ok_12_24m`: Number of archived purchases paid within the past 12-24 months.
  - `status_max_archived_0_6_months`: Maximum times the account was archived within the past 6 months.
  - `acct_days_in_rem_12_24m`: Days in reminder status within the past 12-24 months.

## Business Implications

- Customers with a high number of archived payments and frequent reminders are more likely to default.
- Strategies to mitigate default risk include targeted interventions for high-risk customers and incentive programs for consistent payers.

## Usage Instructions

### Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`


