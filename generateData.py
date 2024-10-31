import pandas as pd
import numpy as np
import random

# Furniture categories list
furniture_categories = ["Table", "Chair", "Sofa", "Cabinet", "Bed", "Desk"]

# Range of years and months
years = [2021, 2022, 2023]
months = list(range(1, 13))

# Create dummy data
num_entries = 500
data = {
    "Year": [random.choice(years) for _ in range(num_entries)],
    "Month": [random.choice(months) for _ in range(num_entries)],
    "Furniture Category": [random.choice(furniture_categories) for _ in range(num_entries)],
    "Order Quantity": [random.randint(1, 50) for _ in range(num_entries)]  # Set random quantity per order
}

# Convert to DataFrame
df_dummy_data = pd.DataFrame(data)

# Display the first 5 rows
print(df_dummy_data.head())  # Show the output

# Export to CSV
df_dummy_data.to_csv('dummy_data.csv', index=False)


