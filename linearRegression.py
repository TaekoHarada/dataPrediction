import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# データの読み込み
data = pd.read_csv('moving_average.csv')

# 'Year' 列を使って2021年と2022年のデータをトレーニング用に、2023年のデータをテスト用に分割
train_data = data[data['Year'] < 2023]
test_data = data[data['Year'] == 2023]

# 特徴量とターゲット変数の定義
X_train = train_data[['Year-Month', 'Moving Average']].copy()
y_train = train_data['Order Quantity']
X_test = test_data[['Year-Month', 'Moving Average']].copy()
y_test = test_data['Order Quantity']

# 'Year-Month' を数値形式に変換
X_train['Year-Month'] = pd.to_datetime(X_train['Year-Month']).dt.to_period('M').astype(int)
X_test['Year-Month'] = pd.to_datetime(X_test['Year-Month']).dt.to_period('M').astype(int)

# モデルのトレーニング
model = LinearRegression()
model.fit(X_train, y_train)

# テストデータでの予測
y_pred = model.predict(X_test)

# モデルの評価
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
