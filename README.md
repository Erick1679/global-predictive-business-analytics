# Global Predictive Business Analytics Platform

## Project Overview

The Global Predictive Business Analytics Platform is an end-to-end analytics solution that combines data engineering, business intelligence, and machine learning to transform business data into actionable insights.

The platform simulates a global retail business environment and provides analytics capabilities across:

- Revenue performance
- Customer behaviour
- Customer churn prediction
- Customer lifetime value
- Product profitability
- Inventory operations
- Revenue forecasting


## Business Problem

Businesses generate large volumes of data but often struggle to answer important questions:

- Which customers are likely to leave?
- Which customers provide the highest value?
- What revenue can be expected in the future?
- Which products drive profitability?
- Where are inventory risks developing?


## Project Objectives

This project aims to:

- Build an end-to-end analytics pipeline
- Generate and process business datasets
- Create predictive machine learning models
- Develop an interactive business intelligence dashboard
- Produce automated analytical reports


## Solution Architecture


## Dataset Overview

The platform generates realistic enterprise datasets:

| Dataset | Records |
|---|---:|
| Customers | 15,000 |
| Products | 1,000 |
| Transactions | 120,000 |
| Marketing Data | 150,000+ |
| Support Records | 5,000 |
| Inventory Records | 10,000 |


## Machine Learning Models

### Customer Churn Prediction

Purpose:

Identify customers with a high probability of leaving.

Models:

- Logistic Regression
- Random Forest Classifier

Evaluation:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC


### Revenue Forecasting

Purpose:

Predict future revenue trends.

Model:

- Random Forest Regression

Evaluation:

- MAE
- RMSE
- MAPE


### Customer Lifetime Value

Purpose:

Estimate customer long-term value.

Outputs:

- CLV scores
- Customer segmentation
- Value ranking


## Dashboard Modules

The Streamlit dashboard includes:

### Executive Overview

- Revenue KPIs
- Profit analysis
- Customer health indicators


### Revenue Forecasting

- Actual vs predicted revenue
- Future revenue projections
- Forecast evaluation


### Customer Intelligence

- Churn risk analysis
- Customer prioritization
- Retention insights


### Customer Lifetime Value

- Customer segmentation
- Value contribution analysis


### Product Analytics

- Product performance
- Revenue ranking
- Profitability analysis


### Operations & Inventory

- Stock monitoring
- Warehouse analysis
- Inventory risk


### Model Performance

- Machine learning evaluation
- Forecast metrics


### Data Quality

- Dataset monitoring
- Pipeline health


## Technology Stack

Programming:

- Python

Data Processing:

- Pandas
- NumPy

Machine Learning:

- Scikit-learn

Visualization:

- Plotly

Dashboard:

- Streamlit

Reporting:

- Excel
- PowerPoint


## How To Run

Install dependencies:

```bash
pip install -r requirements.txt

Activate environment:
source .venv/bin/activate

Generate data:
python -m src.data_generator

Clean data:
python -m src.cleaner

Train models:
python -m src.churn_model
python -m src.forecasting
python -m src.clv_model

Launch dashboard:
streamlit run app.py