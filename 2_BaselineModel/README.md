# Baseline Model

**[Notebook](baseline_model.ipynb)**

## Baseline Model Results

### Model Selection
- **Baseline Model Type:** SARIMAX (Seasonal ARIMA with Exogenous Variables)
- **Rationale:** We selected SARIMAX because it captures seasonality, autocorrelation, and the influence of external drivers such as residual load and renewable generation. This makes it a suitable and interpretable baseline for electricity price forecasting.

### Model Performance
- **Evaluation Metric:** MSE, RMSE
- **Performance Score:** MSE: 1424.25, RMSE: 37.73 EUR/MWh

### Evaluation Methodology
- **Data Split:** Train/Test = 4 weeks / 1 day time steps 
- **Evaluation Metrics:** MSE/RMSE to quantify average forecast error in price units

### Metric Practical Relevance
MSE supports model comparison but is less intuitive due to squared units.
RMSE expresses the average error in €/MWh, making it directly interpretable for market participants. An RMSE of ~46 €/MWh means the baseline is moderately accurate but struggles with volatility and price spikes.


## Next Steps
This baseline model serves as a reference point for evaluating more sophisticated models in the [Model Definition and Evaluation](../3_Model/README.md) phase.
