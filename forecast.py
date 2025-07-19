import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
from utils import calculate_metrics
warnings.filterwarnings("ignore")

PLOT_FOLDER = "static/plots"

def analyze_product_categories(df):
    print("\n--- Analyzing Sales by Product Category ---")
    df_indexed = df.set_index('transaction_date')
    category_sales = df_indexed.groupby('category_id')['quantity_sold'].resample('MS').sum()
    category_map = {101: 'Electronics', 201: 'Home Goods', 301: 'Clothing', 401: 'Books', 501: 'Food'}
    plt.style.use('seaborn-v0_8-whitegrid')

    # Seasonal line
    fig, ax = plt.subplots(figsize=(15, 7))
    plot_data = category_sales.unstack(level=0)
    plot_data.rename(columns=category_map, inplace=True)
    plot_data.plot(ax=ax, marker='o', linestyle='--')
    ax.set_title('Monthly Sales by Product Category')
    plt.tight_layout()
    plt.savefig(f'{PLOT_FOLDER}/category_seasonal_plot.png')
    plt.close()

    # Bar chart
    total_sales = df_indexed.groupby('category_id')['quantity_sold'].sum().sort_values(ascending=False)
    total_sales.index = total_sales.index.map(category_map)
    fig, ax = plt.subplots(figsize=(10, 6))
    total_sales.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Total Sales by Product Category')
    plt.tight_layout()
    plt.savefig(f'{PLOT_FOLDER}/category_total_sales_plot.png')
    plt.close()

def create_seasonal_forecast(df):
    print("\n--- Generating Overall Sales Forecast ---")
    monthly_sales = df.set_index('transaction_date')['quantity_sold'].resample('MS').sum()
    train_data = monthly_sales[:-12]
    test_data = monthly_sales[-12:]
    model = sm.tsa.statespace.SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
    pred = model.get_prediction(start=test_data.index.min(), end=test_data.index.max())
    forecast_values = pred.predicted_mean
    pred_ci = pred.conf_int()

    # Plot forecast
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(monthly_sales.index, monthly_sales, label='Historical Sales')
    ax.plot(test_data.index, test_data, label='Actual Sales', color='royalblue', marker='o')
    ax.plot(forecast_values.index, forecast_values, label='Forecast', color='forestgreen', linestyle='--')
    ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='g', alpha=.2)
    ax.set_title('Sales Forecast vs Actuals')
    plt.tight_layout()
    plt.savefig(f'{PLOT_FOLDER}/final_seasonal_forecast_plot.png')
    plt.close()

    metrics = calculate_metrics(test_data, forecast_values)
    return forecast_values, metrics
