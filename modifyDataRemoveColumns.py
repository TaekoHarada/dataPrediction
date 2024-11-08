import pandas as pd
import random

# Load the data from a CSV file
# data = pd.read_csv('data/salesdata.csv')
data = pd.read_csv('data/combined_salesdata.csv')

# Remove the specified columns
columns_to_remove = ['Store', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
data = data.drop(columns=columns_to_remove)

# Add the 'Category' column with random values from the specified list
categories = ["Table", "Chair", "Sofa", "Cabinet", "Bed", "Desk"]
data['Category'] = [random.choice(categories) for _ in range(len(data))]

# Add the 'Order Quantity' column with random integers between 0 and 100
data['Order Quantity'] = [random.randint(0, 100) for _ in range(len(data))]


# Display the first few rows of the modified DataFrame
print("Modified Data:")
print(data.head())

# Optionally, you can save the modified data to a new CSV file
data.to_csv('data/modified_sales_data.csv', index=False)
