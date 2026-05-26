# Olist-sentiment-prediction
Repository for COMP90051 Project, Group 55


# Run this for requirements...
```
pip install -r requirements.txt
```

# Research Question
To what extent can customer review sentiment be inferred from structured behavioural and transactional signals—such as order timing, payment behaviour, freight ratios, and product diversity—without using any textual review information?

# Dataset
We plan to use the Olist Brazilian E-commerce dataset, which consists of multiple relational tables (orders, payments, products, customers, and reviews).
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data 

# Experimental Design

Three models with different complexity levels were compared:

1. Logistic Regression (low complexity)
2. Random Forest (medium complexity)
3. Neural Oblivious Decision Ensembles (NODE, ICLR 2020) (high complexity)

Nested cross-validation with hyperparameter tuning was implemented from scratch to compare model performance.


# Feature Engineering

Feature engineering was performed in two stages:

1. Pre-split features:
   - Temporal and delivery features
   - Freight and pricing features
   - Product-related features

2. Behavioural representation features:
   - Frustration index
   - Financial burden index
   - Consistency index

# Folder Structure
- **Data/**
  - Raw dataset files
- **Notebooks/**
  - **feature_engineering/**
    - Data cleaning, joining, and feature creation notebooks
  - **runs/**
    - Run nested cross-validation notebooks
- **Scripts/**
  - **cross_validation/**
    - Nested cross-validation implementation
  - **feature_engineering/**
    - Feature generation scripts
  - **models/**
    - Model class definitions and wrappers
    - **node_lib/**
      - Core NODE architecture and related modules

- **Results/**
  - Saved metrics and generated figures

# How to Run

1. Place the Olist dataset files inside the **Data/** folder.

2. Run feature generation scripts:

   **Scripts/feature_engineering/**
   - `create_pre_split_features.py`

3. Run the model notebooks:

   **Notebooks/runs/**
   - `run_cv_LR.ipynb`
   - `run_cv_RF.ipynb`
   - `run_cv_NODE.ipynb`

4. Generated metrics and figures will be saved in:

   **Results/**

