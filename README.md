# Credit Risk Modelling Using Machine Learning

## Project Overview
This project aims to build a credit risk model using machine learning techniques on two datasets. The datasets contain various features related to prospects and their credit approval status. The primary objective is to predict whether a prospect will be approved for credit.

## Datasets
- **Dataset 1 (a1)**: Contains 26 columns and 51,336 rows.
- **Dataset 2 (a2)**: Contains 62 columns and 51,336 rows.

## Data Cleaning
- **Handling Missing Values**:
  - In `df1`, nulls are represented by -99999. We remove rows with this value in the `Age_Oldest_TL` column.
  - In `df2`, columns with more than 10,000 null values are dropped, and specific rows are removed where -99999 is present.
  
- **Merging DataFrames**: 
  - The two cleaned datasets are merged using an inner join on the `PROSPECTID` column.

## Feature Selection
1. **Categorical Features**: 
   - Identified categorical features: `MARITALSTATUS`, `EDUCATION`, `GENDER`, `last_prod_enq2`, and `first_prod_enq2`.
   - Used Chi-square tests to check the association between these categorical features and the target variable (`Approved_Flag`).

2. **Numerical Features**:
   - Calculated Variance Inflation Factor (VIF) to identify multicollinearity among numerical features.

3. **ANOVA**: 
   - Conducted ANOVA tests to determine the significance of numerical features in relation to the `Approved_Flag`.

4. **Final Features**:
   - The final feature set includes selected numerical features and all categorical features.

## Encoding Categorical Variables
- **Ordinal Encoding**: For the `EDUCATION` feature based on predefined ordinal values.
- **One-Hot Encoding**: Applied to `MARITALSTATUS`, `GENDER`, `last_prod_enq2`, and `first_prod_enq2`.

## Model Development
Three machine learning models were built to predict credit approval:
1. **Random Forest Classifier**:
   - Trained on the dataset and evaluated using accuracy, precision, recall, and F1 score.

2. **XGBoost Classifier**:
   - Implemented with multi-class classification. Similar evaluation metrics were calculated.

3. **Decision Tree Classifier**:
   - Built and evaluated against the same metrics as the previous models.

## Hyperparameter Tuning (Commented Out)
- A grid search approach was outlined for tuning hyperparameters of the XGBoost model, including parameters like `colsample_bytree`, `learning_rate`, `max_depth`, `alpha`, and `n_estimators`.

## Results
- Each model's accuracy and detailed classification report (precision, recall, F1 score) were printed.

## Conclusion
This project successfully built and evaluated multiple machine learning models for credit risk prediction, with XGBoost showing the best performance. 

