import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# データの読み込み
data = pd.read_csv('data/modified_sales_data.csv')

# 年、月の列を追加
# Add 'Year', 'Month', and 'Day of Year' columns
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', dayfirst=True)
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['DayOfYear'] = data['Date'].dt.dayofyear  # 1-365
print(data.head())

# 特徴量（Features）とターゲット変数（Target Variable）を定義
X = data[['DayOfYear','Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday']].copy()
X['IsHoliday'] = X['IsHoliday'].astype(int)
X = X.fillna(X.mean())  # 欠損値を補完

y = data['Order Quantity']

# Min year
min_year = data['Year'].min()
print(f"Min Year: {min_year}")

# 最大の年を取得
max_year = data['Year'].max()
print(f"Max Year: {max_year}")

# データを分割
train_data = data[data['Year'] < max_year]
test_data = data[data['Year'] == max_year]

X_train = train_data[['DayOfYear','Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday']].copy()
y_train = train_data['Order Quantity']
X_test = test_data[['DayOfYear','Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday']].copy()
y_test = test_data['Order Quantity']

# 欠損値の補完
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# モデルの作成とトレーニング
model = LinearRegression()
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# MSE 計算
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")




# ---- Predict sales for the next year ----
# Prepare data for the next year
next_year = max_year + 1
next_year_days = pd.date_range(start=f'{next_year}-01-01', end=f'{next_year}-12-31')

# Create a new DataFrame with features for the next year
future_data = pd.DataFrame({
    'DayOfYear': next_year_days.dayofyear,
    'Temperature': X_train['Temperature'].mean(),  # Using the mean of Temperature as a placeholder
    'Fuel_Price': X_train['Fuel_Price'].mean(),    # Using the mean of Fuel_Price
    'CPI': X_train['CPI'].mean(),                  # Using the mean of CPI
    'Unemployment': X_train['Unemployment'].mean(),# Using the mean of Unemployment
    'IsHoliday': next_year_days.map(lambda x: 1 if x.weekday() in [5, 6] else 0)  # Mark weekends as holidays
})

# Make predictions for the next year
future_predictions = model.predict(future_data)

# Print the predicted sales for each day in the next year
print("\nPredicted Sales for Next Year:")
for date, prediction in zip(next_year_days, future_predictions):
    print(f"{date.date()}: Predicted Order Quantity = {prediction:.2f}")