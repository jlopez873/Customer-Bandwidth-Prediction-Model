# Customer Bandwidth Usage Prediction Model

This repository contains the code and data files for predicting customer bandwidth usage for a telecom company using a multiple linear regression model.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
3. [Model Development](#model-development)
4. [Data Analysis](#data-analysis)
5. [Results and Implications](#results-and-implications)
6. [Recommendations](#recommendations)
7. [Running the Code](#running-the-code)

## Introduction
The goal of this project is to create a model that predicts customer bandwidth usage for a telecom company. We will use a dataset containing customer demographic information, usage habits, and other related variables. The project will go through data preparation, model development, data analysis, and interpretation of the results.

## Data Preparation
To prepare the data for analysis, we performed the following steps:
1. Clean and preprocess the data
2. Encode categorical variables
3. Create new columns for the data
4. Store the cleaned data in a CSV file

The prepared data can be found in the `churn_linear_regression.csv` file.

## Model Development
We developed an initial multiple linear regression model to predict customer bandwidth usage. Afterward, we performed step-forward feature selection, feature variance reduction, and testing for feature significance to create a reduced model. The reduced model contains nine independent variables and produces a lower root mean squared error (RMSE) than the initial model.

## Data Analysis
We analyzed the data by calculating the residual standard error and creating residual plots. The reduced model has an RMSE of 59.95 GB and explains 99.9% of the variance in the data.

## Results and Implications
Our reduced model has statistically significant independent variables and explains 99.9% of the variance in our data. The telecom company can use the model to predict future bandwidth usage and identify factors that influence customer bandwidth usage, such as customer age, household size, and subscription to streaming services.

## Recommendations
Based on the results of our analysis, we recommend that the telecom company focus on customer retention, especially for customers with children, higher monthly charges, and subscriptions to DSL and streaming services. Additionally, the company should consider ways to provide better service to older customers who use less bandwidth per year. These insights can be used to inform future marketing and pricing strategies.

## Running the Code
To run the code, follow these steps:

1. Download the `churn_linear_regression.csv` and `linear_regression_models.py` files.
2. Ensure you have Python 3 installed on your computer.
3. Install the necessary Python packages:
   ```
   pip install pandas numpy seaborn matplotlib sklearn statsmodels
   ```
4. Run the `linear_regression_models.py` file in your preferred Python environment or terminal:
   ```
   python linear_regression_models.py
   ```

After running the code, you should see the results of the initial and reduced models, along with the residual standard error and other relevant statistics.
