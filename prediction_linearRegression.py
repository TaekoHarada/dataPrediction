import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

# データの読み込み
data = pd.read_csv('data/modified_sales_data.csv')

# Add 'Year', 'Month', and 'Day of Year' columns
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', dayfirst=True)
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['DayOfYear'] = data['Date'].dt.dayofyear  # 1-365
print(data.head())

# user selected features Get from HTTP request
user_selected_features = ["Year","Month","DayOfYear", 'Unemployment',"IsHoliday"] 

#Target
target = 'Order Quantity'

# 使用する特徴量のリストを作成
all_features = ["Year","Month","DayOfYear", "Temperature", "Fuel_Price", "CPI", "Unemployment", "IsHoliday"]

# null（選択されていない）特徴量を除外
selected_features = [feature for feature in all_features if feature in user_selected_features]

# csvから読み込んだデータ(DataFrame)を、選択された特徴と”ターゲット”を列として作成
features = data[selected_features].copy()

print("選択された特徴量:", selected_features)
print(features)


if 'IsHoliday' in selected_features:
    data['IsHoliday'] = data['IsHoliday'].astype(int)

# data = data.fillna(data.mean())  # 欠損値を補完


# Min year
min_year = data['Year'].min()
print(f"Min Year: {min_year}")

# 最大の年を取得
max_year = data['Year'].max()
print(f"Max Year: {max_year}")

# データを分割
train_data = data[data['Year'] < max_year]
test_data = data[data['Year'] == max_year]

X_train = train_data[selected_features].copy()
y_train = train_data[target]
X_test = test_data[selected_features].copy()
y_test = test_data[target]

# 欠損値の補完
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())


print("X_train", X_train)
print("y_train", y_train)

# model = LinearRegression()
# たくさんの木を並べて安定した結果を出す
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# learning_rate: 学習のスピードや慎重さ（大きいと早く学習するが、不安定になることもある）。デフォルトは0.1

# 1本の木でうまくいかなかった部分を次の木で少しずつ修正していく
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)  # learning_rateは学習率

model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# MSE 計算
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")




# ---- Predict sales for the next year ----

# Combine training and test data to use for the future prediction
full_data = pd.concat([X_train, X_test])

# Prepare data for the next year
next_year = max_year + 1
next_year_days = pd.date_range(start=f'{next_year}-01-01', end=f'{next_year}-12-31')

# Initialize the future_data DataFrame
future_data = pd.DataFrame()

# Dynamically add columns to future_data based on user-selected features
if 'Year' in selected_features:
    future_data['Year'] = next_year_days.year

if 'Month' in selected_features:
    future_data['Month'] = next_year_days.month

if 'DayOfYear' in selected_features:
    future_data['DayOfYear'] = next_year_days.dayofyear

if 'Temperature' in selected_features:
    future_data['Temperature'] = full_data['Temperature'].mean()  # Using the mean as a placeholder

if 'Fuel_Price' in selected_features:
    future_data['Fuel_Price'] = full_data['Fuel_Price'].mean()  # Using the mean as a placeholder

if 'CPI' in selected_features:
    future_data['CPI'] = full_data['CPI'].mean()  # Using the mean as a placeholder

if 'Unemployment' in selected_features:
    future_data['Unemployment'] = full_data['Unemployment'].mean()  # Using the mean as a placeholder

if 'IsHoliday' in selected_features:
    future_data['IsHoliday'] = next_year_days.map(lambda x: 1 if x.weekday() in [5, 6] else 0)  # Mark weekends as holidays

# Vefify if user selected features are in future_data
print(future_data)

# Make predictions for the next year
future_predictions = model.predict(future_data)

# Convert future_predictions to a DataFrame
future_predictions_df = pd.DataFrame({
    'Date': next_year_days,
    'Predicted Order Quantity': future_predictions
})

# Export predicted data to a CSV file
csv_file_path = 'data/future_predictions.csv'
future_predictions_df.to_csv(csv_file_path, index=False)  # Export without row indices
