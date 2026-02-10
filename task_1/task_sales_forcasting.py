# ===================== IMPORTS =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet


# ===================== LOAD DATA =====================
file_path = "data/raw/archive (2)/Sample - Superstore.csv"

df = pd.read_csv(file_path, encoding="latin1")

print(df.head())
print("Dataset shape:", df.shape)

# Save clean UTF-8 copy
df.to_csv("data/raw/Sample - Superstore.csv", index=False)
print("Clean UTF-8 version saved")


# ===================== DATA CLEANING =====================
# Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Sort by date
df = df.sort_values('Order Date')

# Keep only required columns
df_clean = df[['Order Date', 'Sales', 'Region', 'Category']]

# Create Month column
df_clean['Month'] = df_clean['Order Date'].dt.to_period('M')


# ===================== MONTHLY AGGREGATION =====================
monthly_sales = (
    df_clean
    .groupby('Month')['Sales']
    .sum()
    .reset_index()
)

monthly_sales['Month'] = monthly_sales['Month'].dt.to_timestamp()

print("\nMonthly Sales:")
print(monthly_sales.head())

# Save processed data
monthly_sales.to_csv("data/processed/monthly_sales.csv", index=False)
print("Monthly sales data saved successfully")


# ===================== REGION-WISE AGGREGATION =====================
region_monthly_sales = (
    df_clean
    .groupby(['Month', 'Region'])['Sales']
    .sum()
    .reset_index()
)

print("\nRegions:", region_monthly_sales['Region'].unique())


# ===================== EDA: MONTHLY SALES TREND =====================
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['Month'], monthly_sales['Sales'], linewidth=2)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()


# ===================== EDA: REGION-WISE SALES TREND =====================
plt.figure(figsize=(12, 6))

for region in region_monthly_sales['Region'].unique():
    region_data = region_monthly_sales[region_monthly_sales['Region'] == region]
    plt.plot(
        region_data['Month'].dt.to_timestamp(),
        region_data['Sales'],
        marker='o',
        label=region
    )

plt.title("Region-wise Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()


# ===================== EDA: OVERALL VS REGION =====================
plt.figure(figsize=(12, 6))

# Overall sales
plt.plot(
    monthly_sales['Month'],
    monthly_sales['Sales'],
    color='black',
    linewidth=3,
    label='Overall Sales'
)

# Region-wise sales
for region in region_monthly_sales['Region'].unique():
    region_data = region_monthly_sales[region_monthly_sales['Region'] == region]
    plt.plot(
        region_data['Month'].dt.to_timestamp(),
        region_data['Sales'],
        linestyle='--',
        marker='o',
        label=region
    )

plt.title("Overall vs Region-wise Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()


# ===================== PROPHET FORECASTING =====================
print("\n--- STARTING PROPHET FORECASTING ---\n")

# Prepare data for Prophet
prophet_df = monthly_sales.rename(
    columns={
        'Month': 'ds',
        'Sales': 'y'
    }
)

print(prophet_df.head())

# Initialize Prophet model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

# Fit model
model.fit(prophet_df)

# Create future dataframe (next 6 months)
future = model.make_future_dataframe(
    periods=6,
    freq='MS'
)

# Predict
forecast = model.predict(future)

print("\nForecast for next 6 months:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6))


# ===================== PROPHET FORECAST PLOT =====================
model.plot(forecast)
plt.title("Sales Forecast using Prophet (Next 6 Months)")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()
plt.close()


# ===================== PROPHET COMPONENTS =====================
model.plot_components(forecast)
plt.show()
plt.close()


# ===================== END SCRIPT =====================
input("Press Enter to close all plots and exit...")
