# Ames House Price Regression

An end-to-end machine learning project for predicting residential house sale prices using the Ames Housing dataset.

## Project Overview

This project develops a complete regression pipeline covering:

* Exploratory Data Analysis (EDA)
* Missing-value handling
* Feature engineering
* Categorical encoding and numerical scaling
* Baseline Linear Regression
* Ridge Regression with cross-validation
* Model evaluation using RMSE, MAE, and R²

The goal is to investigate the factors associated with house prices and build a regression model that can accurately predict `SalePrice`.

## Dataset

The project uses the **Ames Housing / House Prices: Advanced Regression Techniques** dataset from Kaggle.

* Training observations: 1,460
* Original features: 79 predictors
* Target variable: `SalePrice`

The Kaggle test set was not used because this project creates its own held-out test set from the training data for model evaluation.

## Project Workflow

### 1. Exploratory Data Analysis

The dataset was examined for:

* Dataset structure and data types
* Missing values
* Numerical feature distributions
* Outliers
* Correlations with `SalePrice`

The strongest numerical correlations with `SalePrice` included:

* `OverallQual`
* `GrLivArea`
* `GarageCars`
* `GarageArea`
* `TotalBsmtSF`

### 2. Data Cleaning and Feature Engineering

The following preprocessing steps were performed:

* Removed the `Id` column from the modeling data
* Handled categorical missing values representing absent features
* Filled `LotFrontage` using its median
* Filled `GarageYrBlt` and `MasVnrArea` with 0 where appropriate
* Filled the missing `Electrical` value using the most frequent category
* Verified that no missing values remained

Four additional features were created:

* `TotalSF` — combined basement and above-ground floor area
* `TotalBathrooms` — combined full and half bathrooms
* `HouseAge` — age of the house at the time of sale
* `RemodAge` — years since the most recent remodeling

Categorical variables were one-hot encoded and numerical variables were standardized using a preprocessing pipeline.

## Models

### Baseline Linear Regression

A multiple Linear Regression model was trained as the baseline.

### Ridge Regression

A Ridge Regression model was used to reduce coefficient instability caused by correlated features and the high-dimensional encoded feature space.

The regularization parameter `alpha` was tuned using **5-fold cross-validation**.

The selected value was:

```text
alpha = 10
```

## Model Results

Evaluation was performed on a held-out test set representing 20% of the dataset.

| Model             |      RMSE |       MAE |      R² |
| ----------------- | --------: | --------: | ------: |
| Linear Regression | 95,748.93 | 23,356.50 | -0.1952 |
| Ridge Regression  | 30,535.73 | 19,175.49 |  0.8784 |

Ridge Regression produced substantially more stable predictions than the baseline Linear Regression model on the held-out test set.

## Visualizations

The notebook includes:

* SalePrice distribution
* OverallQual vs SalePrice
* GrLivArea vs SalePrice
* Correlation analysis
* Actual vs predicted SalePrice
* Ridge regression residual analysis

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Google Colab
* GitHub

## How to Run

1. Open the notebook `Ames_House_Price_Regression.ipynb`.
2. Upload the Kaggle `train.csv` dataset to the Colab environment.
3. Run the notebook cells from top to bottom.
4. The notebook performs preprocessing, feature engineering, model training, cross-validation, and evaluation.

## Repository Structure

```text
Ames-House-Price-Regression/
│
├── Ames_House_Price_Regression.ipynb
└── README.md
```

## Key Takeaway

The project demonstrates a complete regression workflow from raw housing data through exploratory analysis, preprocessing, feature engineering, model training, hyperparameter tuning, and held-out test evaluation. Ridge Regression provided substantially better test-set performance than the baseline Linear Regression model.
