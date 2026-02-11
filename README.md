⚡ Energy Advanced Time Series Prediction
High-Frequency Electricity Price Forecasting Using Statistical & Deep Learning Models
🔗 Repository

https://github.com/Aadip-Thapaliya/EATSP-Energy-Advance-Time-Series-Prediction

📌 Project Overview

This project investigates Day-Ahead Electricity Price Forecasting (DE-LU market) using advanced time series modeling techniques on high-frequency (15-minute resolution) energy market data.

Electricity prices are highly volatile and difficult to predict due to:

Strong daily & weekly seasonality

Extreme price spikes

Heavy-tailed distribution

Nonlinear interactions between load, renewables, and generation

Structural market shifts

To address this, we compare:

📈 Statistical Models (SARIMAX)

🌲 Gradient Boosting Models (XGBoost, LightGBM)

📊 Additive Model (Prophet)

🧠 Deep Learning (LSTM)

The goal is to determine which modeling paradigm best captures complex electricity market dynamics.

🎯 Task Type

Multivariate Time Series Regression

📊 Dataset Information

Source: Energy Charts – Fraunhofer ISE
https://www.energy-charts.info

Region: Germany–Luxembourg (DE-LU)
Frequency: 15-minute intervals
Time Range: 2020–2025
Total Observations: 210,433
Raw Features: 23
Engineered Features: 59

Target Variable

Day Ahead Auction Price (EUR/MWh)

Range: approx. -125 to 400 EUR/MWh

Highly volatile

Heavy-tailed (kurtosis ≈ 14)

Exhibits strong intraday & weekly seasonality

🔍 Exploratory Data Analysis

Performed analyses:

Missing value detection & cleaning

Correlation heatmap

Seasonal decomposition

Autocorrelation (ACF/PACF)

Rolling statistics

Principal Component Analysis (PCA)

Outlier detection (Z-score method)

Distribution analysis

Key Findings

Strong daily (96-step) seasonality

Strong weekly (672-step) seasonality

Residual load strongly correlated with price

Renewable generation negatively correlated with price

Extreme spikes dominate error metrics

🛠 Feature Engineering

Created additional predictive signals:

Lag features

Rolling averages (24h, 168h)

Hour / weekday / month features

Renewable aggregation

Residual load derived metrics

Temporal cyclic encodings

Total engineered features: 59

🧪 Modeling Approaches
1️⃣ SARIMAX (Baseline)

Captures:

Autoregression

Seasonality

Exogenous variables

Used as interpretable benchmark.

2️⃣ XGBoost

Gradient boosted decision trees optimized for regression.

3️⃣ LightGBM

Histogram-based gradient boosting framework.

4️⃣ Prophet

Additive forecasting model designed for strong seasonality.

5️⃣ LSTM (Deep Learning Model)

Architecture:

128 LSTM units

Swish activation

Lookback window

Dense output layer

Adam optimizer

MSE loss

Designed to capture nonlinear long-range temporal dependencies.

📈 Model Performance Comparison
Model	RMSE (EUR/MWh)
SARIMAX	37.74
XGBoost	36.86
LightGBM	40.56
Prophet	85.19
LSTM	18.19
🏆 Best Model

Model: LSTM
RMSE: 18.19 EUR/MWh
MAE: 13.08 EUR/MWh
R²: 0.8836

Improvement Over Baseline

SARIMAX RMSE: 37.74

LSTM RMSE: 18.19

➡ ~52% reduction in prediction error

🧠 Key Insights
Most Important Features

Residual Load

Total Load

Renewable Generation Share

Wind Generation

Solar Generation

Model Strengths

Captures nonlinear relationships

Learns temporal dependencies

Handles volatility better than linear models

Strong generalization on high-frequency data

Model Limitations

Computationally expensive

Hyperparameter sensitive

Assumes availability of future exogenous variables

No probabilistic uncertainty modeling implemented

💼 Practical / Business Impact

Accurate electricity price forecasting enables:

Improved trading strategy optimization

Reduced financial risk in volatile markets

Better renewable integration planning

Enhanced grid operation efficiency

A 50% reduction in forecasting error can significantly improve trading margins and hedging strategies.

📂 Project Structure
EATSP-Energy-Advance-Time-Series-Prediction/
│
├── 0_LiteratureReview/
├── 1_DatasetCharacteristics/
│   └── exploratory_data_analysis.ipynb
│
├── 2_BaselineModel/
│   └── baseline_model.ipynb
│
├── 3_Model/
│   └── model_definition_evaluation/
│
├── 4_Presentation/
│   └── README.md
│
├── CoverImage/
│   └── cover_image.png
│
└── README.md

💾 Saved Artifacts

Trained XGBoost model (.pkl)

Trained LSTM model

Prophet model

Feature scalers

Future prediction CSV files

Visualization outputs

🔮 Future Improvements

Transformer-based time-series models

Attention mechanisms

Bayesian hyperparameter optimization

Probabilistic forecasting

Real-time deployment (API)

Automated retraining pipeline

Concept drift monitoring

🛠 Tech Stack

Python

Pandas / NumPy

Scikit-learn

Statsmodels

XGBoost

LightGBM

TensorFlow / Keras

Prophet

Matplotlib / Seaborn

🖼 Cover Image

📜 License

This project uses publicly available data from Fraunhofer ISE Energy Charts.
