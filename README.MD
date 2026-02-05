# Bitcoin Market Sentiment vs Trader Performance

## Project Overview
This project analyzes how **Bitcoin market sentiment (Fear & Greed Index)** influences **trader behavior and performance** using real trading data from Hyperliquid. The study merges sentiment data with historical trade activity, extracts behavioral patterns, and builds a machine learning model to predict trade outcomes. A Streamlit application is also developed for real-time prediction.

---

## Objective

The goals of this project are:

- Analyze how trader performance varies across sentiment regimes  
- Identify behavioral changes driven by Fear vs Greed  
- Build a machine learning model to predict trade outcomes  
- Extract actionable trading insights  
- Deploy a real-time prediction app using Streamlit  

---

## Datasets

### 1. Fear & Greed Index
Daily Bitcoin market sentiment indicator.

**Columns**
- `date`
- `classification` (Extreme Fear → Extreme Greed)
- `value`

Sentiment encoding:

| Sentiment | Score |
|----------|-------|
| Extreme Fear | 0 |
| Fear | 1 |
| Neutral | 2 |
| Greed | 3 |
| Extreme Greed | 4 |

---

### 2. Hyperliquid Historical Trading Data
Real trade-level dataset representing trader activity.

**Key Columns**
- Account  
- Execution Price  
- Trade Direction (Buy/Sell)  
- Trade Size (USD)  
- Closed PnL  
- Timestamp  

---

## Data Cleaning & Preprocessing

Steps performed:

- Converted timestamps to datetime format  
- Aggregated trades to daily level  
- Merged trader dataset with sentiment using trade date  
- Verified merge alignment and handled missing values  
- Encoded sentiment categories numerically  

This ensures proper alignment between **market context and trader activity**.

---

## Feature Engineering

### Trade-Level Features
- `PnL_bin` → Profit / Loss classification  
- `Side_Binary` → Buy = 1, Sell = 0  
- `Normalized_PnL` → PnL scaled by trade size  

### Behavioral / Temporal (Lag) Features
Lag features prevent data leakage and simulate real-world trading:

- Previous trade PnL (`pnl_lag1`)  
- Previous trade size (`size_lag1`)  
- Previous sentiment (`sent_lag1`)  

These capture **behavioral momentum and contextual memory**.

---

## Exploratory Analysis

Behavioral patterns across sentiment regimes:

- Traders perform best during **Greed / Extreme Greed**
- Fear periods show lower profitability and higher losses
- Buy-side trades dominate during Greed
- Traders take larger risks during Greed
- Traders become conservative during Fear

This confirms that **market sentiment significantly influences trader behavior**.

---

## Machine Learning Model

### Problem Definition
Predict trade outcome:

- 0 → Loss  
- 1 → Neutral  
- 2 → Profit  

### Model Approach
Ensemble-based classification:

- Random Forest  
- Gradient Boosting  
- XGBoost  

### Modeling Principles
- Time-aware train/test split  
- Lag features used → prevents data leakage  
- Class imbalance handled  
- Feature importance analyzed  

---

## Model Performance

**Accuracy ≈ 90%**

Balanced precision and recall across classes.

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

Key observations:
- Profit predicted strongest  
- Some confusion between Neutral and Profit  
- Loss prediction hardest (expected in financial markets)  

---

### Feature Importance

![Feature Importance](images/feature_importance.png)

Key insights:

- Previous trade outcome (`pnl_lag1`) is strongest predictor → **behavioral momentum**
- Trade direction important
- Sentiment contributes moderately
- Behavioral features dominate predictive signal  

Conclusion:
> Trader behavior is the primary driver; sentiment acts as contextual signal.

---

## Streamlit App

A Streamlit application was built to enable **real-time trade outcome prediction**.

### What the App Does
Predicts whether a trade will result in:
- Loss
- Neutral
- Profit

### Inputs
- Trade direction  
- Trade size  
- Trade impact  
- Market sentiment  
- Previous trade outcome  

### Outputs
- Predicted class  
- Confidence probabilities  
- Probability visualization  

---

## How to Run the Project

### Install Dependencies 
pip install -r requirements.txt

### Run the script
streamlit run app.py
