import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# データの読み込み
data = pd.read_csv('data/moving_average.csv')

# 2018年から2023年のデータのみを抽出
historical_data = data[(data['Year'] <= 2023)]

# 月ごとの移動平均を計算
monthly_moving_avg = historical_data.groupby('Month')['Moving Average'].mean()
print("Moving avg: ",monthly_moving_avg)

# 次の年の 'Year-Month' を準備
next_year = 2024
months = range(1, 13)  # 1月から12月まで

# 次の年のデータフレームを作成し、月ごとの移動平均を設定
future_data = pd.DataFrame({
    'Year-Month': [f"{next_year}-{str(month).zfill(2)}" for month in months],
    'Moving Average': [monthly_moving_avg[month] for month in months]  # 各月に対応する月の平均移動平均を使用
})

# 'Year-Month' を数値形式（YYYYMM形式）に変換
future_data['Year-Month'] = pd.to_datetime(future_data['Year-Month']).dt.to_period('M').astype(int)

# モデルのトレーニング
# 特徴量とターゲット変数の定義
X = historical_data[['Year-Month', 'Moving Average']].copy()
y = historical_data['Order Quantity']

# 'Year-Month' を数値形式に変換
X['Year-Month'] = pd.to_datetime(X['Year-Month']).dt.to_period('M').astype(int)

# モデルの作成とトレーニング
model = LinearRegression()
model.fit(X, y)

# 未来データでの予測
future_predictions = model.predict(future_data)

# 予測結果の表示
for month, prediction in zip(months, future_predictions):
    print(f"{next_year}-{str(month).zfill(2)}: Predicted Order Quantity = {prediction:.2f}")
