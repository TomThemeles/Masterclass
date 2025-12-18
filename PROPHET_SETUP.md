# Prophet Time Series Forecasting - Setup Guide

## 📋 Overview

This directory contains a complete implementation of Facebook's Prophet time series forecasting model for retail sales prediction. The implementation includes comprehensive pre-processing, data preparation, model setup, training, and evaluation.

## 🎯 What's Included

### 1. Jupyter Notebook (`model_prophet.ipynb`)
A comprehensive, interactive notebook that demonstrates the complete Prophet workflow:
- **Step-by-step execution** with explanations
- **Data loading and inspection**
- **Data pre-processing** (date parsing, missing value handling)
- **Data aggregation** to daily time series
- **Data quality checks**
- **Train/test split** (temporal split for time series)
- **Prophet model configuration** with optimal parameters
- **Model training and forecasting**
- **Performance evaluation** with multiple metrics (MAE, RMSE, MAPE)
- **Comprehensive visualizations** (forecast plots, component analysis)

**Best for:** 
- Learning and understanding the process
- Experimentation and parameter tuning
- Visual exploration of results
- Google Colab or Jupyter environments

### 2. Python Script (`prophet_forecasting.py`)
A production-ready, modular Python script that automates the entire pipeline:
- **Modular functions** for each step
- **Complete automation** - run end-to-end with one command
- **Clear console output** with progress indicators
- **Error handling** and validation
- **Results saved automatically** (CSV files for predictions and metrics)

**Best for:**
- Automated execution
- Batch processing
- Integration into workflows
- Command-line usage

## 🚀 Quick Start

### Option 1: Using the Jupyter Notebook (Recommended for Learning)

1. **Upload to Google Colab:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload `model_prophet.ipynb`
   - Upload `train.csv` data file

2. **Run the notebook:**
   ```python
   # Execute cells sequentially from top to bottom
   # Follow the instructions in each section
   ```

3. **View results:**
   - Metrics displayed in notebook
   - Visualizations shown inline
   - Download output files if needed

### Option 2: Using the Python Script (Recommended for Production)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the script:**
   ```bash
   python3 prophet_forecasting.py
   ```

3. **Check outputs:**
   - `prophet_predictions.csv` - Forecast results
   - `prophet_metrics.csv` - Performance metrics

## 📊 Data Requirements

### Input Data Format
- **File:** `train.csv`
- **Required columns:**
  - `date`: Date of sales (any date format)
  - `unit_sales`: Number of units sold
  - Other columns: `store_nbr`, `item_nbr`, `onpromotion`, etc.

### Prophet Format
The scripts automatically convert data to Prophet's required format:
- `ds`: Date column (datetime)
- `y`: Target variable (numeric)

## ⚙️ Model Configuration

### Default Parameters
```python
Prophet(
    growth='linear',                    # Linear growth trend
    changepoint_prior_scale=0.05,       # Trend flexibility
    changepoint_range=0.8,              # 80% of data for changepoints
    n_changepoints=25,                  # Number of changepoints
    
    yearly_seasonality=True,            # Annual patterns
    weekly_seasonality=True,            # Day-of-week patterns
    daily_seasonality=False,            # Not needed for daily data
    seasonality_mode='multiplicative',  # Multiplicative seasonality
    
    interval_width=0.95                 # 95% confidence intervals
)
```

### Custom Seasonality
- **Monthly seasonality** added (period=30.5 days)
- Captures mid-term seasonal patterns

### Why These Parameters?
- **Linear growth:** Suitable for retail sales without obvious saturation
- **Changepoint scale (0.05):** Moderate flexibility to detect trend changes
- **Multiplicative seasonality:** Better for data where seasonal effects scale with trend
- **Yearly/Weekly/Monthly:** Captures all major seasonal patterns in retail

## 📈 Model Performance

The model is evaluated on a test set (20% of data) using three metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| **MAE** | Mean Absolute Error - Average prediction error | Lower is better |
| **RMSE** | Root Mean Squared Error - Penalizes large errors | Lower is better |
| **MAPE** | Mean Absolute Percentage Error - Error as % | <20% is good |

### Expected Performance
Based on the sample run:
- **MAE:** ~280 units
- **RMSE:** ~495 units
- **MAPE:** ~16% (Good accuracy!)

## 📁 Output Files

### 1. `prophet_predictions.csv`
Contains predictions for the test period:
```csv
ds,actual,yhat,yhat_lower,yhat_upper
2016-09-12,1040.0,1614.82,948.31,2224.06
2016-09-13,1323.0,1506.52,893.16,2131.20
...
```

Columns:
- `ds`: Date
- `actual`: Actual sales values
- `yhat`: Predicted sales
- `yhat_lower`: Lower bound (95% CI)
- `yhat_upper`: Upper bound (95% CI)

### 2. `prophet_metrics.csv`
Contains performance metrics:
```csv
Model,MAE,RMSE,MAPE,Training_Time_Seconds
Prophet,282.31,495.63,16.31,0.17
```

## 🔧 Customization

### Adjusting Parameters

**Make the model more flexible (capture more changes):**
```python
model = Prophet(
    changepoint_prior_scale=0.1,  # Increase from 0.05
    seasonality_prior_scale=15.0  # Increase from 10.0
)
```

**Make the model more conservative (smoother):**
```python
model = Prophet(
    changepoint_prior_scale=0.01,  # Decrease from 0.05
    seasonality_prior_scale=5.0    # Decrease from 10.0
)
```

### Adding Holidays
```python
# Define Ecuadorian holidays
holidays = pd.DataFrame({
    'holiday': 'christmas',
    'ds': pd.to_datetime(['2013-12-25', '2014-12-25', '2015-12-25']),
    'lower_window': 0,
    'upper_window': 1,
})

model = Prophet(holidays=holidays)
```

### Adding External Regressors
```python
# Add promotions as a regressor
model = Prophet()
model.add_regressor('total_promotions')

# Prepare data with regressor
prophet_data['total_promotions'] = daily_sales['total_promotions']
```

## 🎓 Learning Resources

### Understanding Prophet
- [Official Prophet Documentation](https://facebook.github.io/prophet/)
- [Prophet Paper](https://peerj.com/preprints/3190/)
- Prophet is designed for business forecasting with:
  - Strong seasonal effects
  - Multiple seasons of historical data
  - Missing data and outliers
  - Historical trend changes

### Key Concepts

**1. Trend:** Long-term increase or decrease in sales

**2. Seasonality:** 
- Yearly (annual patterns)
- Weekly (weekday vs weekend)
- Monthly (monthly cycles)

**3. Changepoints:** Points where trend changes occur

**4. Uncertainty:** Confidence intervals showing prediction range

## 🐛 Troubleshooting

### Issue: MAPE shows very large value
**Cause:** Zero values in the data cause division by zero  
**Solution:** The script now filters out zero values before calculating MAPE

### Issue: Training is slow
**Cause:** Large dataset or many changepoints  
**Solution:** 
- Use a subset of data for testing
- Reduce `n_changepoints`
- Set `mcmc_samples=0` (default, faster)

### Issue: Poor forecast accuracy
**Solutions:**
- Tune `changepoint_prior_scale`
- Add relevant holidays
- Add external regressors (promotions, oil prices)
- Use longer training period
- Check for data quality issues

### Issue: Import error for prophet
**Solution:**
```bash
pip install prophet
# Or if on Mac with M1/M2:
conda install -c conda-forge prophet
```

## 📊 Comparison with Other Models

| Feature | Prophet | ARIMA | Exponential Smoothing |
|---------|---------|-------|---------------------|
| Ease of use | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| Handles missing data | ✅ Yes | ❌ No | ❌ No |
| Seasonality | Multiple | Limited | Yes |
| Interpretability | High | Medium | High |
| Training speed | Fast | Slow | Fast |
| External regressors | ✅ Yes | Limited | ❌ No |

## 🚀 Next Steps

1. **Compare models:** Run ARIMA and Exponential Smoothing notebooks
2. **Tune parameters:** Experiment with different configurations
3. **Add features:** Include holidays, promotions, external variables
4. **Deploy:** Use for production forecasting
5. **Monitor:** Track forecast accuracy over time

## 📝 Notes

- **Data size:** 300,000 records aggregated to 1,680 daily observations
- **Training time:** ~0.2 seconds on standard hardware
- **Memory usage:** ~40MB for full dataset
- **Python version:** 3.8+ recommended
- **Prophet version:** 1.1.4+

## 💡 Tips

1. **Start simple:** Use default parameters first
2. **Visualize components:** Use `model.plot_components()` to understand patterns
3. **Check residuals:** Look for patterns in prediction errors
4. **Use cross-validation:** For more robust evaluation (see Prophet docs)
5. **Document changes:** Keep track of parameter changes and their effects

## 🤝 Contributing

To improve this implementation:
1. Test with different datasets
2. Experiment with parameter combinations
3. Add more visualizations
4. Implement cross-validation
5. Add automated hyperparameter tuning

## 📄 License

This implementation is part of the Masterschool Time Series Forecasting project.

---

**Author:** Masterschool Project  
**Last Updated:** December 2025  
**Version:** 1.0
