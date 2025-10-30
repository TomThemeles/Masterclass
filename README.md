# 🛒 Retail Sales Forecasting Project

## 📋 Project Overview

This project analyzes retail sales data from Ecuadorian stores and builds predictive models to forecast future sales. The goal is to compare different time series forecasting approaches and recommend the best model for business use in inventory planning, staffing optimization, and promotion scheduling.

**Author:** [Your Name]  
**Date Completed:** October 2025  
**Assignment:** Masterschool Time Series Forecasting Modeling Project

---

## 🎯 Business Problem and Context

### Why Sales Forecasting Matters
Retail planners need accurate sales forecasts to:
- **Optimize inventory levels** - Avoid stockouts and reduce excess inventory costs
- **Plan staffing** - Ensure adequate coverage during peak periods
- **Schedule promotions** - Time marketing campaigns for maximum impact
- **Allocate resources** - Make data-driven decisions about store investments

### What Makes This Challenging?
- **Seasonality** - Weekly, monthly, and yearly patterns affect sales
- **Holidays and Events** - Special events cause irregular spikes
- **Promotions** - Promotional periods create non-standard patterns
- **External Factors** - Economic conditions (oil prices) influence consumer behavior
- **Multiple Products** - Different items have different demand patterns

### Business Impact
A 10% improvement in forecast accuracy can lead to:
- 5% reduction in inventory costs
- 2-3% increase in sales (fewer stockouts)
- Better customer satisfaction
- Improved operational efficiency

---

## 📊 Data Overview

### Datasets Used
1. **train.csv**

2. **Additional datasets**

### Key Statistics

### Data Limitations
- Some missing values in certain periods
- Promotional data not available for all items
- External factors (weather, competitors) not included
- Oil price data has some gaps

---

## 🔬 Methodology

### Approach
1. **Data Preparation**
   - Aggregated sales by date

2. **Exploratory Data Analysis**
   - Analyzed trends and seasonality

3. **Feature Engineering**
   - Lag features (7-day, 14-day averages)

4. **Model Development**
   - Trained three different forecasting models

### Models Tested
1. **Prophet** - Facebook's forecasting tool designed for business time series
2. **Exponential Smoothing (Holt-Winters)** - Statistical method for trend and seasonality
3. **ARIMA** - Classic autoregressive integrated moving average

### Evaluation Metrics
- **MAE (Mean Absolute Error)** - Average magnitude of errors
- **RMSE (Root Mean Squared Error)** - Penalizes large errors more heavily
- **MAPE (Mean Absolute Percentage Error)** - Error as a percentage

**Lower values are better for all metrics.**

---

## 🏆 Model Comparison

### Performance Summary


### 🥇 Best Model: Prophet

**Why Prophet Won:**
1. **Lowest error rates** across all three metrics (MAE, RMSE, MAPE)
2. **Handles seasonality automatically** - Captures both weekly and yearly patterns
3. **Robust to missing data** - Works well even with gaps
4. **Business-friendly** - Interpretable components (trend, seasonality, holidays)
5. **Reasonable training time** - Not too slow for regular retraining
6. **Easy to extend** - Can easily add holidays, events, and external regressors


### Model Complexity Assessment

| Model | Ease of Use | Tuning Required | Interpretability | Best For |
|-------|-------------|-----------------|------------------|----------|
| Prophet | ⭐⭐⭐⭐⭐ Easy | Minimal | High | Business forecasting |
| Exponential Smoothing | ⭐⭐⭐⭐ Moderate | Low | High | Baseline comparisons |
| ARIMA | ⭐⭐ Complex | High | Medium | Statistical analysis |

---

## 🚀 Setup Instructions

### Prerequisites
- Google Colab account (recommended) or Python 3.8+
- Google Drive (for data storage)
- Basic knowledge of Python and Jupyter notebooks

### Installation

1. **Clone this repository**
```bash
git clone https://github.com/yourusername/retail-sales-forecasting.git
cd retail-sales-forecasting
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Download the data**
   - Download `train.csv` from [provide data source link]
   - Place it in the `data/` folder
   - Or upload to Google Colab when running notebooks

### Running in Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload the notebook you want to run from the `notebooks/` folder
3. Upload `train.csv` using the file upload button 📁 in Colab
4. Run all cells sequentially

## 📁 Repository Structure

```
retail-sales-forecasting/
│
├── data/                          # Data files
│   └── sample_output.csv         # Sample predictions
│
├── notebooks/                     # Jupyter/Colab notebooks
│   ├── data_prep.ipynb           # Data preparation and EDA
│   ├── model_prophet.ipynb       # Prophet model
│   ├── model_exponential_smoothing.ipynb  # Exponential Smoothing
│   └── model_arima.ipynb         # ARIMA model
│
├── visualizations/                # Key plots and figures
│   ├── sales_trends.png          # Time series plots
│   ├── seasonal_patterns.png     # Seasonality analysis
│   ├── model_comparison.png      # Model performance comparison
│   └── prophet_forecast.png      # Best model predictions
│
├── README.md                      # This file - project documentation
├── requirements.txt               # Python package dependencies
└── .gitignore                    # Files to ignore in git

```

### File Descriptions

- **notebooks/** - All Jupyter/Colab notebooks for the project
- **data/** - Dataset storage (note: large files should use Git LFS)
- **visualizations/** - Key plots and analysis figures
- **README.md** - Comprehensive project documentation
- **requirements.txt** - All Python packages needed

---

## 🛠️ Technical Details

### Computing Requirements
- **Memory**: 8GB RAM recommended (4GB minimum)
- **Processing**: Standard CPU sufficient for Prophet and Exponential Smoothing
- **GPU**: Recommended for ARIMA on full dataset (use T4 in Colab)

### Dependencies
- Python 3.8+
- pandas, numpy - Data manipulation
- matplotlib, seaborn - Visualization
- prophet - Facebook's forecasting tool
- statsmodels - Statistical models (ARIMA, Exponential Smoothing)
- scikit-learn - Evaluation metrics

See `requirements.txt` for complete list with versions.

---

