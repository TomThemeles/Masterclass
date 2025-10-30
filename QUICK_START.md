# Quick Start Guide

## Setup

1. **Create GitHub repository** named `retail-sales-forecasting`
2. **Upload these files** to your repository
3. **Download train.csv** (don't upload to GitHub - too large)

## Running the Project

### Step 1: Data Preparation
- Open `data_prep.ipynb` in Google Colab
- Upload `train.csv`
- Complete the TODO sections
- Run all cells

### Step 2: Run Models
Complete each model notebook:
- `model_prophet.ipynb`
- `model_exponential_smoothing.ipynb`
- `model_arima.ipynb`

### Step 3: Compare Results
Fill in the table in README.md with your results.

## Tips

- **Start with Prophet** - Easiest model
- **Use weekly data for ARIMA** - Much faster
- **Document your decisions** - Add comments explaining why
- **Save visualizations** - Put in visualizations/ folder

## Evaluation Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = mean_absolute_percentage_error(y_true, y_pred) * 100
