# Moving Average Calculation

import pandas as pd

# Step 1: Read the CSV file
df = pd.read_csv('data/modified_sales_data.csv')

# Step 2: Display the first few rows of the DataFrame to understand its structure
print("Original Data:")
print(df.head())

# Step 3: Calculate the 3-month moving average for the 'Order Quantity' column
df['Moving Average'] = df['Order Quantity'].rolling(window=3).mean()

# Step 4: Remove rows with NaN values in 'Moving Average'
df = df.dropna(subset=['Moving Average'])

# Step 5: Create a 'Year-Month' feature by concatenating 'Year' and 'Month'
df['Year-Month'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)

# Step 6: Display the resulting DataFrame
print("\nData with Moving Average and Year-Month:")
print(df)

# Step 7: Export the DataFrame to a CSV file
csv_file_path = 'moving_average.csv'
df.to_csv(csv_file_path, index=False)  # Export without row indices

print(f"Data exported to {csv_file_path}")
