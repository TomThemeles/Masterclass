#!/usr/bin/env python3
"""
Prophet Time Series Forecasting Model
Pre-processing, Preparation, and Setup Script

This script demonstrates the complete workflow for setting up and running
a Prophet time series forecasting model on retail sales data.

Author: Masterclass Project
Date: December 2025
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def load_and_inspect_data(file_path='train.csv'):
    """
    Load and inspect the raw data
    
    Args:
        file_path (str): Path to the training data CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    print('='*70)
    print('📂 STEP 1: LOAD AND INSPECT DATA')
    print('='*70)
    
    print(f'\nLoading data from {file_path}...')
    df = pd.read_csv(file_path)
    
    print(f'✅ Data loaded successfully!')
    print(f'   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns')
    print(f'   Columns: {list(df.columns)}')
    print(f'   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB')
    
    return df


def preprocess_data(df):
    """
    Pre-process data: parse dates, handle missing values, check data quality
    
    Args:
        df (pd.DataFrame): Raw dataframe
        
    Returns:
        pd.DataFrame: Pre-processed dataframe
    """
    print('\n' + '='*70)
    print('🧹 STEP 2: PRE-PROCESS DATA')
    print('='*70)
    
    # Parse dates
    print('\n📅 Converting date column to datetime...')
    df['date'] = pd.to_datetime(df['date'])
    print(f'   Date range: {df["date"].min().strftime("%Y-%m-%d")} to {df["date"].max().strftime("%Y-%m-%d")}')
    print(f'   Total days: {(df["date"].max() - df["date"].min()).days} days')
    print(f'   Unique dates: {df["date"].nunique()}')
    
    # Check for missing values
    print('\n🔍 Checking for missing values...')
    missing_count = df['unit_sales'].isnull().sum()
    if missing_count > 0:
        print(f'   ⚠️ Found {missing_count} missing values in unit_sales')
        print('   🔧 Filling with 0 (assuming no sales)')
        df['unit_sales'] = df['unit_sales'].fillna(0)
    else:
        print('   ✅ No missing values in unit_sales')
    
    # Check for negative sales
    negative_count = (df['unit_sales'] < 0).sum()
    print(f'\n🔍 Negative sales (returns): {negative_count:,} records ({negative_count/len(df)*100:.2f}%)')
    if negative_count > 0:
        print('   💡 Keeping negative values (represent returns)')
    
    return df


def aggregate_to_daily(df):
    """
    Aggregate data by date for time series forecasting
    
    Args:
        df (pd.DataFrame): Pre-processed dataframe
        
    Returns:
        pd.DataFrame: Daily aggregated data in Prophet format (ds, y)
    """
    print('\n' + '='*70)
    print('📊 STEP 3: AGGREGATE TO DAILY TIME SERIES')
    print('='*70)
    
    print('\nAggregating sales by date...')
    daily_sales = df.groupby('date').agg({
        'unit_sales': 'sum',
        'onpromotion': 'sum',
        'store_nbr': 'nunique',
        'item_nbr': 'nunique'
    }).reset_index()
    
    daily_sales.columns = ['date', 'total_sales', 'total_promotions', 'num_stores', 'num_items']
    
    print(f'✅ Aggregated to {len(daily_sales):,} daily records')
    
    # Format for Prophet (ds, y)
    print('\n📋 Formatting for Prophet (ds, y columns)...')
    prophet_data = daily_sales[['date', 'total_sales']].copy()
    prophet_data.columns = ['ds', 'y']
    prophet_data = prophet_data.sort_values('ds').reset_index(drop=True)
    
    # Check for missing dates
    date_range = pd.date_range(start=prophet_data['ds'].min(), 
                               end=prophet_data['ds'].max(), 
                               freq='D')
    missing_dates = set(date_range) - set(prophet_data['ds'])
    
    if len(missing_dates) > 0:
        print(f'   ⚠️ Found {len(missing_dates)} missing dates')
        print('   🔧 Filling missing dates with 0 sales')
        
        complete_dates = pd.DataFrame({'ds': date_range})
        prophet_data = complete_dates.merge(prophet_data, on='ds', how='left')
        prophet_data['y'] = prophet_data['y'].fillna(0)
    else:
        print('   ✅ No missing dates - time series is complete')
    
    # Display statistics
    print(f'\n📊 Daily Sales Statistics:')
    print(f'   Mean:   {prophet_data["y"].mean():>12,.2f} units')
    print(f'   Median: {prophet_data["y"].median():>12,.2f} units')
    print(f'   Std:    {prophet_data["y"].std():>12,.2f} units')
    print(f'   Min:    {prophet_data["y"].min():>12,.0f} units')
    print(f'   Max:    {prophet_data["y"].max():>12,.0f} units')
    
    return prophet_data


def train_test_split(data, split_ratio=0.8):
    """
    Create train/test split for time series (temporal split)
    
    Args:
        data (pd.DataFrame): Prophet-formatted data
        split_ratio (float): Ratio for training data (default: 0.8)
        
    Returns:
        tuple: (train_data, test_data)
    """
    print('\n' + '='*70)
    print('🔀 STEP 4: TRAIN/TEST SPLIT')
    print('='*70)
    
    split_index = int(len(data) * split_ratio)
    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()
    
    print(f'\n📊 Split ratio: {split_ratio*100:.0f}% / {(1-split_ratio)*100:.0f}%')
    print(f'   Total records:   {len(data):>6,} days')
    print(f'   Training set:    {len(train_data):>6,} days ({len(train_data)/len(data)*100:.1f}%)')
    print(f'   Test set:        {len(test_data):>6,} days ({len(test_data)/len(data)*100:.1f}%)')
    
    print(f'\n📅 Date Ranges:')
    print(f'   Training: {train_data["ds"].min().strftime("%Y-%m-%d")} to {train_data["ds"].max().strftime("%Y-%m-%d")}')
    print(f'   Test:     {test_data["ds"].min().strftime("%Y-%m-%d")} to {test_data["ds"].max().strftime("%Y-%m-%d")}')
    
    return train_data, test_data


def initialize_prophet_model():
    """
    Initialize and configure Prophet model with optimal parameters
    
    Returns:
        Prophet: Configured Prophet model
    """
    print('\n' + '='*70)
    print('⚙️ STEP 5: INITIALIZE PROPHET MODEL')
    print('='*70)
    
    print('\nConfiguring Prophet model...')
    model = Prophet(
        # Growth parameters
        growth='linear',
        
        # Changepoint parameters (detect trend changes)
        changepoint_prior_scale=0.05,
        changepoint_range=0.8,
        n_changepoints=25,
        
        # Seasonality parameters
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        seasonality_prior_scale=10.0,
        
        # Uncertainty parameters
        interval_width=0.95,
        uncertainty_samples=1000
    )
    
    # Add custom monthly seasonality
    model.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=5
    )
    
    print('✅ Prophet model configured!')
    print(f'\n📋 Configuration:')
    print(f'   Growth:              {model.growth}')
    print(f'   Seasonality mode:    {model.seasonality_mode}')
    print(f'   Yearly seasonality:  {model.yearly_seasonality}')
    print(f'   Weekly seasonality:  {model.weekly_seasonality}')
    print(f'   Monthly seasonality: ✓ (custom, period=30.5 days)')
    print(f'   Changepoint scale:   {model.changepoint_prior_scale}')
    print(f'   Seasonality scale:   {model.seasonality_prior_scale}')
    print(f'   Confidence interval: {model.interval_width*100}%')
    
    return model


def train_model(model, train_data):
    """
    Train the Prophet model on training data
    
    Args:
        model (Prophet): Configured Prophet model
        train_data (pd.DataFrame): Training data
        
    Returns:
        tuple: (trained_model, training_time)
    """
    print('\n' + '='*70)
    print('🎓 STEP 6: TRAIN PROPHET MODEL')
    print('='*70)
    
    print(f'\nTraining on {len(train_data):,} days of data...')
    print('This may take a few moments...')
    
    start_time = time.time()
    model.fit(train_data)
    training_time = time.time() - start_time
    
    print(f'\n✅ Model trained successfully!')
    print(f'⏱️  Training time: {training_time:.2f} seconds')
    
    return model, training_time


def generate_forecast(model, periods):
    """
    Generate forecast for future periods
    
    Args:
        model (Prophet): Trained Prophet model
        periods (int): Number of periods to forecast
        
    Returns:
        pd.DataFrame: Forecast dataframe
    """
    print('\n' + '='*70)
    print('🔮 STEP 7: GENERATE FORECAST')
    print('='*70)
    
    print(f'\nCreating future dataframe for {periods} days...')
    future = model.make_future_dataframe(periods=periods, freq='D')
    
    print(f'Generating predictions...')
    forecast = model.predict(future)
    
    print(f'✅ Forecast generated!')
    print(f'   Forecast shape: {forecast.shape}')
    print(f'   Date range: {forecast["ds"].min().strftime("%Y-%m-%d")} to {forecast["ds"].max().strftime("%Y-%m-%d")}')
    
    return forecast


def evaluate_model(forecast, test_data):
    """
    Evaluate model performance on test data
    
    Args:
        forecast (pd.DataFrame): Forecast dataframe
        test_data (pd.DataFrame): Test data
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print('\n' + '='*70)
    print('📊 STEP 8: EVALUATE MODEL PERFORMANCE')
    print('='*70)
    
    # Extract test period predictions
    test_forecast = forecast[forecast['ds'] >= test_data['ds'].min()].copy()
    test_results = test_forecast.merge(test_data, on='ds', how='inner')
    test_results.columns = [col if col != 'y' else 'actual' for col in test_results.columns]
    
    # Calculate metrics
    y_true = test_results['actual'].values
    y_pred = test_results['yhat'].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate MAPE manually to handle zeros
    # Filter out zero values to avoid division by zero
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = float('inf')
    
    mean_actual = y_true.mean()
    mae_percentage = (mae / mean_actual) * 100 if mean_actual != 0 else float('inf')
    
    # Display metrics
    print('\n' + '='*60)
    print('📊 MODEL PERFORMANCE METRICS')
    print('='*60)
    print(f'Mean Absolute Error (MAE):        {mae:>12,.2f} units')
    print(f'Root Mean Squared Error (RMSE):   {rmse:>12,.2f} units')
    print(f'Mean Absolute Percentage Error:   {mape:>12.2f}%')
    print(f'\nMAE as % of mean sales:           {mae_percentage:>12.2f}%')
    print(f'Mean actual sales:                {mean_actual:>12,.2f} units')
    print('='*60)
    
    # Interpretation
    print('\n💡 Interpretation:')
    print(f'   On average, predictions are off by {mae:,.0f} units ({mape:.1f}%)')
    if mape < 10:
        print('   ✅ Excellent forecast accuracy (<10% MAPE)')
    elif mape < 20:
        print('   ✅ Good forecast accuracy (10-20% MAPE)')
    elif mape < 30:
        print('   ⚠️  Acceptable forecast accuracy (20-30% MAPE)')
    else:
        print('   ⚠️  Consider model improvement (>30% MAPE)')
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'MAE_percentage': mae_percentage,
        'mean_actual': mean_actual
    }
    
    return metrics, test_results


def plot_forecast(model, forecast, test_data, save_path=None):
    """
    Create visualization of forecast vs actual
    
    Args:
        model (Prophet): Trained model
        forecast (pd.DataFrame): Forecast data
        test_data (pd.DataFrame): Test data
        save_path (str): Optional path to save figure
    """
    print('\n' + '='*70)
    print('📈 STEP 9: VISUALIZE RESULTS')
    print('='*70)
    
    # Extract test predictions
    test_forecast = forecast[forecast['ds'] >= test_data['ds'].min()].copy()
    test_results = test_forecast.merge(test_data, on='ds', how='inner')
    test_results.columns = [col if col != 'y' else 'actual' for col in test_results.columns]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Test period forecast vs actual
    ax1 = axes[0]
    ax1.plot(test_results['ds'], test_results['actual'], 
             color='steelblue', linewidth=2, label='Actual Sales', marker='o', markersize=4)
    ax1.plot(test_results['ds'], test_results['yhat'], 
             color='coral', linewidth=2, label='Prophet Forecast', linestyle='--', marker='s', markersize=4)
    ax1.fill_between(test_results['ds'], 
                     test_results['yhat_lower'], 
                     test_results['yhat_upper'],
                     alpha=0.2, color='coral', label='95% Confidence Interval')
    
    ax1.set_title('🔮 Prophet Forecast vs Actual Sales (Test Period)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Total Daily Sales (units)', fontsize=12)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Full timeline with components
    ax2 = axes[1]
    ax2.plot(forecast['ds'], forecast['yhat'], 
             color='darkgreen', linewidth=1.5, label='Forecast', alpha=0.8)
    ax2.fill_between(forecast['ds'], 
                     forecast['yhat_lower'], 
                     forecast['yhat_upper'],
                     alpha=0.2, color='darkgreen', label='Uncertainty')
    
    ax2.set_title('📈 Full Forecast Timeline', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Total Daily Sales (units)', fontsize=12)
    ax2.legend(fontsize=12, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'   ✅ Figure saved to {save_path}')
    else:
        print('   ℹ️  Figure displayed (not saved)')
    
    plt.close()


def main():
    """
    Main execution function - runs the complete Prophet forecasting pipeline
    """
    print('\n' + '='*70)
    print('🚀 PROPHET TIME SERIES FORECASTING - COMPLETE PIPELINE')
    print('='*70)
    print('\nThis script demonstrates:')
    print('  ✓ Data loading and pre-processing')
    print('  ✓ Time series preparation')
    print('  ✓ Prophet model setup and configuration')
    print('  ✓ Model training')
    print('  ✓ Forecast generation')
    print('  ✓ Performance evaluation')
    print('  ✓ Results visualization')
    print('='*70)
    
    try:
        # Execute pipeline
        df = load_and_inspect_data('train.csv')
        df = preprocess_data(df)
        prophet_data = aggregate_to_daily(df)
        train_data, test_data = train_test_split(prophet_data, split_ratio=0.8)
        model = initialize_prophet_model()
        model, training_time = train_model(model, train_data)
        forecast = generate_forecast(model, periods=len(test_data))
        metrics, test_results = evaluate_model(forecast, test_data)
        
        # Save results
        print('\n' + '='*70)
        print('💾 SAVING RESULTS')
        print('='*70)
        
        # Save predictions
        output_file = 'prophet_predictions.csv'
        test_results[['ds', 'actual', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(
            output_file, index=False
        )
        print(f'\n✅ Predictions saved to {output_file}')
        
        # Save metrics
        metrics_df = pd.DataFrame({
            'Model': ['Prophet'],
            'MAE': [metrics['MAE']],
            'RMSE': [metrics['RMSE']],
            'MAPE': [metrics['MAPE']],
            'Training_Time_Seconds': [training_time]
        })
        metrics_df.to_csv('prophet_metrics.csv', index=False)
        print(f'✅ Metrics saved to prophet_metrics.csv')
        
        # Final summary
        print('\n' + '='*70)
        print('🎯 PIPELINE COMPLETED SUCCESSFULLY!')
        print('='*70)
        print('\n✅ All steps completed:')
        print('   1. ✓ Data loaded and inspected')
        print('   2. ✓ Data pre-processed and validated')
        print('   3. ✓ Daily time series aggregated')
        print('   4. ✓ Train/test split created')
        print('   5. ✓ Prophet model initialized')
        print('   6. ✓ Model trained')
        print('   7. ✓ Forecasts generated')
        print('   8. ✓ Performance evaluated')
        print('   9. ✓ Results saved')
        
        print(f'\n📊 Final Performance:')
        print(f'   MAE:  {metrics["MAE"]:,.2f} units')
        print(f'   RMSE: {metrics["RMSE"]:,.2f} units')
        print(f'   MAPE: {metrics["MAPE"]:.2f}%')
        print(f'   Training time: {training_time:.2f} seconds')
        
        print('\n' + '='*70)
        
        return model, forecast, metrics
        
    except Exception as e:
        print(f'\n❌ Error occurred: {str(e)}')
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == '__main__':
    model, forecast, metrics = main()
