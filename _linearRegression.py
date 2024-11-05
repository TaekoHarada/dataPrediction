import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('moving_average.csv')

# Explore dataset
print(data.head())

# ステップ 1: 特徴量とターゲット変数を定義
X = data[['Year-Month', 'Moving Average']].copy()
y = data['Order Quantity']

### Year-Monthを数値に変換（例えば、YYYYMM形式）
# 'Year-Month' 列を日付形式に変換
date_column = pd.to_datetime(X['Year-Month'])

# 日付形式を「年月のみの周期（Period）」に変換
year_month_period = date_column.dt.to_period('M')

# 'Year-Month' 列を YYYYMM 形式の整数に変換
X['Year-Month'] = year_month_period.astype(int)


# ステップ 2: データをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ステップ 3: 線形回帰モデルをトレーニング
model = LinearRegression()
model.fit(X_train, y_train)

# ステップ 4: テストセットでの予測
y_pred = model.predict(X_test)

# ステップ 5: モデルの評価　Mean Squared Error 
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
