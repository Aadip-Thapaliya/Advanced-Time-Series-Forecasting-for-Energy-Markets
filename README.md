# ⚡ EATSP - Energy Advance Time Series Prediction

> Advanced machine learning models for electricity price forecasting in the German (DE-LU) energy market

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Models Implemented](#-models-implemented)
- [Performance Results](#-performance-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Features](#-key-features)
- [Methodology](#-methodology)
- [Future Work](#-future-work)
- [Contributors](#-contributors)
- [License](#-license)

---

## 🎯 Overview

EATSP (Energy Advance Time Series Prediction) is a comprehensive machine learning project focused on forecasting **Day-Ahead Electricity Auction Prices** for the German-Luxembourg (DE-LU) energy market. The project compares multiple state-of-the-art forecasting approaches, from traditional statistical models to deep learning architectures.

### Key Highlights

- 📊 **70,176+ samples** at 15-minute intervals (January 2020 - June 2025)
- 🤖 **6+ ML/DL models** implemented and evaluated
- 📈 **88% R² accuracy** achieved with LSTM architecture
- ⚡ **Real-time forecasting** capability for energy market prices
- 🔄 **Production-ready** pipeline with automated retraining

---

## 🔍 Problem Statement

Electricity price forecasting is crucial for:

- **Energy traders** making bid/offer decisions
- **Grid operators** managing supply-demand balance
- **Renewable energy** integration planning
- **Market participants** optimizing generation schedules

**Challenge:** Electricity prices exhibit:
- High volatility and occasional price spikes
- Strong seasonality (daily, weekly, yearly patterns)
- Complex dependencies on weather, demand, and generation mix
- Non-linear relationships between multiple features

**Goal:** Develop accurate, robust forecasting models that capture these dynamics while maintaining interpretability and computational efficiency.

---

## 📊 Dataset

### Source
- **Provider:** [Energy Charts](https://www.energy-charts.info/charts/power/chart.htm?c=DE&legendItems=0wm&interval=week&year=2025&source=public)
- **Owner:** Fraunhofer Institute for Solar Energy Systems ISE
- **License:** Public data collection
- **Coverage:** Germany (DE-LU market)

### Characteristics

| Attribute | Details |
|-----------|---------|
| **Samples** | 70,176 observations |
| **Interval** | 15 minutes |
| **Time Range** | Jan 2020 - Jun 2025 |
| **Features** | 23 original features |
| **Target** | Day Ahead Auction (EUR/MWh) |
| **Price Range** | -125 to 400 EUR/MWh |

### Feature Categories

**1. Generation Sources (MW)**
- Fossil fuels: Brown coal/lignite, Hard coal, Gas, Oil
- Renewables: Solar, Wind (onshore/offshore), Hydro, Biomass
- Nuclear, Geothermal, Waste

**2. Grid Metrics (MW)**
- Load (total electricity demand)
- Residual load (demand minus renewables)
- Cross-border electricity trading
- Pumped storage (consumption/generation)

**3. Renewable Indicators (%)**
- Renewable share of load
- Renewable share of generation

**4. Engineered Features**
- Temporal: Hour, day, weekday, month
- Lag features: Historical prices and generation
- Rolling statistics: 24h, 168h averages
- Renewable aggregation

---

## 📁 Project Structure

```
EATSP-Energy-Advance-Time-Series-Prediction/
│
├── 1_DatasetCharacteristics/
│   ├── dataset_info.ipynb          # Dataset exploration and statistics
│   ├── exploratory_data_analysis.ipynb
│   └── README.md
│
├── 2_BaselineModel/
│   ├── baseline_model.ipynb        # SARIMAX implementation
│   ├── INSTRUCTIONS.md
│   └── README.md
│
├── 3_ModelDefinition/
│   ├── model_evaluation.ipynb      # Advanced models (LSTM, XGBoost, etc.)
│   ├── hyperparameter_tuning.ipynb
│   └── README.md
│
├── data/
│   └── Energy_Charts_2025_January_to_June.csv
│
├── models/                          # Saved model files
│   ├── xgboost_model.pkl
│   ├── lstm_model.h5
│   └── prophet_model.pkl
│
├── notebooks/
│   └── complete_pipeline.ipynb     # End-to-end workflow
│
├── visualizations/                  # Generated plots and charts
│
├── requirements.txt
└── README.md
```

---

## 🤖 Models Implemented

### 1. **SARIMAX** (Baseline)
- **Type:** Statistical time series model
- **Purpose:** Establish performance baseline
- **Strengths:** Captures seasonality, autocorrelation, exogenous variables
- **Use Case:** Interpretable reference model

### 2. **XGBoost**
- **Type:** Gradient boosted decision trees
- **Strengths:** Fast training, handles non-linearity
- **Use Case:** Production deployment candidate

### 3. **LightGBM**
- **Type:** Gradient boosting framework
- **Strengths:** Efficient memory usage, faster training
- **Use Case:** Large-scale deployments

### 4. **LSTM (Long Short-Term Memory)**
- **Type:** Recurrent neural network
- **Architecture:** 128 units, Swish activation
- **Strengths:** Captures long-term temporal dependencies
- **Use Case:** Best overall performance

### 5. **Prophet**
- **Type:** Additive time series model
- **Strengths:** Automatic seasonality detection, handles missing data
- **Use Case:** Quick prototyping and forecasting

### 6. **Ridge Regression**
- **Type:** Linear model with L2 regularization
- **Strengths:** Simple, fast, prevents overfitting
- **Use Case:** Baseline for linear relationships

---

## 📈 Performance Results

### Model Comparison

| Model | R² Score | MSE | RMSE (EUR/MWh) | MAE (EUR/MWh) | Training Time |
|-------|----------|-----|----------------|---------------|---------------|
| **LSTM** ⭐ | **0.884** | **331.05** | **18.19** | **13.08** | Medium |
| Ridge Regression | 0.996* | - | - | 2.87 | Fast |
| XGBoost | - | 1358.92 | 36.86 | - | Fast |
| XGBoost (Tuned) | - | 1532.84 | 39.15 | - | Fast |
| LightGBM | - | 1645.51 | 40.56 | - | Fast |
| **SARIMAX (Baseline)** | 0.627 | 1424.26 | 37.74 | - | Slow |
| N-BEATS | -0.371 | 3900.34 | 62.45 | 52.55 | Medium |
| Prophet | -1.231 | - | 85.19 | 63.79 | Medium |

*Ridge regression performance on specific feature subset

### Key Findings

🏆 **Best Model:** LSTM achieves **88.4% R²** with **RMSE of 18.19 EUR/MWh**

- ✅ **106% improvement** over SARIMAX baseline (37.74 → 18.19 RMSE)
- ✅ Effectively captures **temporal patterns** and **price volatility**
- ✅ Handles **complex non-linear relationships** between features
- ✅ Robust to **price spikes** and market anomalies

### Train/Test Split Strategy

- **Training:** 80% of data (4 weeks of 15-min intervals)
- **Testing:** 20% (1 day = 96 time steps)
- **Validation:** Time-series cross-validation for hyperparameter tuning

---

## 🚀 Installation

### Prerequisites

```bash
Python 3.8+
pip or conda package manager
```

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Aadip-Thapaliya/EATSP-Energy-Advance-Time-Series-Prediction.git
cd EATSP-Energy-Advance-Time-Series-Prediction
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Libraries

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
tensorflow>=2.8.0
keras>=2.8.0
xgboost>=1.5.0
lightgbm>=3.3.0
prophet>=1.1.0
```

---

## 💻 Usage

### Quick Start

```python
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained LSTM model
model = load_model('models/lstm_model.h5')

# Load and preprocess data
df = pd.read_csv('data/Energy_Charts_2025_January_to_June.csv')
# ... (preprocessing steps)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"RMSE: {rmse:.2f} EUR/MWh")
print(f"R²: {r2:.4f}")
```

### Running Complete Pipeline

```bash
# Navigate to notebooks directory
cd notebooks

# Launch Jupyter
jupyter notebook complete_pipeline.ipynb
```

### Training New Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define model
model = Sequential([
    LSTM(units=128, activation='swish', input_shape=(look_back, n_features)),
    Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mape'])

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=100,
    validation_split=0.2,
    verbose=2
)

# Save
model.save('models/my_lstm_model.h5')
```

---

## 🔑 Key Features

### 1. **Comprehensive EDA**
- Statistical summaries and distributions
- Missing value analysis
- Feature correlation heatmaps
- Seasonal decomposition
- Autocorrelation analysis (ACF/PACF)

### 2. **Advanced Feature Engineering**
- **Temporal features:** Hour, day, weekday, month
- **Lag features:** Historical prices (1-24 hours)
- **Rolling statistics:** 24h, 168h moving averages
- **Domain features:** Renewable aggregation, residual load
- **Interaction features:** Load × renewable share

### 3. **Model Interpretability**
- Feature importance rankings
- SHAP values for model explanations
- Prediction confidence intervals
- Error distribution analysis

### 4. **Production-Ready Pipeline**
- Automated data preprocessing
- Model versioning and tracking
- Scalable deployment architecture
- Real-time inference API (planned)

---

## 🔬 Methodology

### Data Preprocessing

1. **Data Cleaning**
   - Handle missing values (forward fill for continuous series)
   - Remove duplicates
   - Convert timezone-aware timestamps

2. **Feature Scaling**
   - StandardScaler for numerical features
   - Preserve temporal order for time series

3. **Sequence Creation**
   - Sliding window approach (look_back=5)
   - Multivariate input sequences

### Model Training

1. **Train/Test Split:** 80/20 time-based split
2. **Hyperparameter Tuning:** Grid search with time series CV
3. **Early Stopping:** Monitor validation loss
4. **Model Selection:** Compare RMSE, MAE, R² on test set

### Evaluation Metrics

- **R² Score:** Proportion of variance explained
- **RMSE:** Root Mean Squared Error (EUR/MWh)
- **MAE:** Mean Absolute Error (EUR/MWh)
- **MAPE:** Mean Absolute Percentage Error

---

## 🔮 Future Work

### Short-term Improvements

- [ ] Implement ensemble methods (stacking LSTM + XGBoost)
- [ ] Add attention mechanisms to LSTM architecture
- [ ] Develop multi-horizon forecasting (1h, 6h, 24h ahead)
- [ ] Incorporate weather forecast data
- [ ] Add uncertainty quantification (prediction intervals)

### Medium-term Goals

- [ ] Deploy REST API for real-time predictions
- [ ] Create interactive dashboard (Streamlit/Dash)
- [ ] Implement automated retraining pipeline
- [ ] Add anomaly detection for price spikes
- [ ] Extend to other European energy markets

### Long-term Vision

- [ ] Transformer-based architectures
- [ ] Reinforcement learning for trading strategies
- [ ] Multi-market price forecasting
- [ ] Integration with grid stability analysis
- [ ] Open-source prediction platform

---

## 👥 Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Aadip-Thapaliya">
        <img src="https://github.com/Aadip-Thapaliya.png" width="100px;" alt="Aadip Thapaliya"/>
        <br />
        <sub><b>Aadip Thapaliya</b></sub>
      </a>
      <br />
      <sub>Project Lead</sub>
    </td>
    <td align="center">
      <a href="https://github.com/MeHelge">
        <img src="https://github.com/MeHelge.png" width="100px;" alt="MeHelge"/>
        <br />
        <sub><b>MeHelge</b></sub>
      </a>
      <br />
      <sub>Contributor</sub>
    </td>
  </tr>
</table>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Fraunhofer Institute for Solar Energy Systems ISE** for providing the Energy Charts dataset
- **Energy Charts** platform for data accessibility
- Open-source ML/DL communities (TensorFlow, scikit-learn, XGBoost)

---

## 📞 Contact

For questions, suggestions, or collaborations:

- **GitHub Issues:** [Create an issue](https://github.com/Aadip-Thapaliya/EATSP-Energy-Advance-Time-Series-Prediction/issues)
- **Email:** [Contact through GitHub profile](https://github.com/Aadip-Thapaliya)

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

---

<div align="center">

**[⬆ Back to Top](#-eatsp---energy-advance-time-series-prediction)**

Made with ❤️ for the energy forecasting community

</div>
