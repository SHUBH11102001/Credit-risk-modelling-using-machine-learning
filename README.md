# Credit-risk-modelling-using-machine-learning

This project involves predicting approval statuses for bank loan applicants using machine learning models. It begins with loading and preprocessing two datasets, each containing thousands of rows and numerous features. Null values, represented by -99999, are handled by either removing the rows or columns as appropriate. The datasets are then merged based on a common identifier, 'PROSPECTID'. Feature selection includes identifying and retaining significant categorical and numerical features using chi-square tests, Variance Inflation Factor (VIF), and ANOVA tests. Categorical features are encoded using one-hot and ordinal encoding methods. Three machine learning models—Random Forest, XGBoost, and Decision Tree—are trained and evaluated on the data, with XGBoost showing superior performance. Hyperparameter tuning is conducted for XGBoost to further enhance its accuracy. The project's goal is to assist the bank in making informed decisions by accurately predicting the loan approval categories (P1, P2, P3, P4) for applicants based on their profile data.


Description of the features is provided in the excel sheet to have a better understanding of the features.

