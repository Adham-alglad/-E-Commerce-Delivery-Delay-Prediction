# DelayGuard: E-Commerce Delivery Delay Prediction

## Project Overview

DelayGuard is a data science and machine learning project that predicts whether an e-commerce order is likely to be delivered late.

The project uses the Brazilian E-Commerce Public Dataset by Olist and applies a full data science workflow including data cleaning, exploratory data analysis, feature engineering, feature selection, model training, evaluation, and deployment using Streamlit.

## Problem Statement

Late delivery is a major operational issue in e-commerce because it affects customer satisfaction, logistics efficiency, and business reputation.

This project aims to build a binary classification model that predicts whether an order will be delivered after its estimated delivery date.

## Target Variable

- `0` = Delivered on time
- `1` = Delivered late

## Dataset

The project uses multiple related datasets, including:

- Orders
- Customers
- Payments
- Products
- Sellers
- Order Items
- Product Category Translation

The datasets were cleaned, merged, and aggregated so that each row in the final dataset represents one delivered order.

## Project Workflow

1. Project Understanding
2. Data Loading and Initial Inspection
3. Data Cleaning
4. Target Creation
5. Data Merging and Aggregation
6. Exploratory Data Analysis
7. Feature Engineering
8. Feature Selection using Mutual Information
9. Model Training and Comparison
10. Hyperparameter Tuning
11. Threshold Tuning
12. Final Evaluation
13. Streamlit Web App Deployment

## Models Compared

The following models were tested:

- Logistic Regression
- Decision Tree
- Random Forest

The final selected model was a tuned Random Forest classifier.

## Final Model

The final model was selected based on precision, recall, F1-score, ROC-AUC, and average precision, because the target variable was highly imbalanced.

## Web Application

The project includes a Streamlit web application with:

- A prediction page
- A model training journey page
- An EDA dashboard page

## How to Run the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
