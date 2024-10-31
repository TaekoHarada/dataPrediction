import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ステップ 1: 特徴量とターゲット変数を定義
X = df[['Year-Month', 'Moving Average']]
y = df['Order Quantity']

# Year-Monthを数値に変換（例えば、YYYYMM形式）
X['Year-Month'] = pd.to_datetime(X['Year-Month']).dt.to_period('M').astype(int)

# ステップ 2: データをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ステップ 3: 線形回帰モデルをトレーニング
model = LinearRegression()
model.fit(X_train, y_train)

# ステップ 4: テストセットでの予測
y_pred = model.predict(X_test)

# ステップ 5: モデルの評価
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
