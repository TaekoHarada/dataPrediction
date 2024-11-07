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

# ユーザーが送信した特徴量の名前のリスト（nullの特徴量は含まれていない）
user_selected_features = ["Date","Year","Month","DayOfYear",'Order Quantity',"Temperature", "CPI", "IsHoliday"]  # 例: ユーザーが選択した特徴量

# 使用する特徴量のリストを作成
all_features = ["DayOfYear", "Temperature", "Fuel_Price", "CPI", "Unemployment", "IsHoliday"]

# null（選択されていない）特徴量を除外
selected_features = [feature for feature in all_features if feature in user_selected_features]

# csvから読み込んだデータ(DataFrame)を、選択された特徴量を列として作成
features = data[selected_features].copy()

print("選択された特徴量:", selected_features)
print(features)


# 特徴量（Features）とターゲット変数（Target Variable）を定義
X = data[selected_features].copy()

if 'IsHoliday' in user_selected_features:
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

# Convert future_predictions to a DataFrame
future_predictions_df = pd.DataFrame({
    'Date': next_year_days,
    'Predicted Order Quantity': future_predictions
})

# Export predicted data to a CSV file
csv_file_path = 'data/future_predictions.csv'
future_predictions_df.to_csv(csv_file_path, index=False)  # Export without row indices
