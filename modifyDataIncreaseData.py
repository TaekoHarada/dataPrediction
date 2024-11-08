import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 元のデータをCSVファイルから読み込む
input_file_path = 'data/salesdata.csv'  # 元のデータのCSVファイルパス
data = pd.read_csv(input_file_path)

# 'Date'列をdatetime型に変換してから "dd/mm/yyyy" 形式にフォーマット
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', dayfirst=True)
data['Date'] = data['Date'].dt.strftime('%d/%m/%Y')  # 日付を "dd/mm/yyyy" 形式にフォーマット

# 昨日の日付を取得
yesterday = datetime.now() - timedelta(days=1)

# 追加するダミーデータの日付範囲を作成
start_date = pd.to_datetime(data['Date'].max(), format='%d/%m/%Y') + timedelta(days=1)
date_range = pd.date_range(start=start_date, end=yesterday)

# ダミーデータを作成
dummy_data = pd.DataFrame({
    'Store': np.random.choice(data['Store'].unique(), size=len(date_range)),
    'Date': date_range.strftime('%d/%m/%Y'),  # 日付を "dd/mm/yyyy" 形式にフォーマット
    'Temperature': np.random.uniform(30, 90, size=len(date_range)),  # 30～90の範囲でランダムに生成
    'Fuel_Price': np.random.uniform(2, 4, size=len(date_range)),  # 2～4の範囲でランダムに生成
    'MarkDown1': np.random.choice([np.nan, np.random.uniform(1000, 20000)], size=len(date_range)),
    'MarkDown2': np.random.choice([np.nan, np.random.uniform(500, 10000)], size=len(date_range)),
    'MarkDown3': np.random.choice([np.nan, np.random.uniform(10, 500)], size=len(date_range)),
    'MarkDown4': np.random.choice([np.nan, np.random.uniform(50, 2000)], size=len(date_range)),
    'MarkDown5': np.random.choice([np.nan, np.random.uniform(1000, 5000)], size=len(date_range)),
    'CPI': np.random.uniform(190, 220, size=len(date_range)),  # 190～220の範囲でランダムに生成
    'Unemployment': np.random.uniform(5, 10, size=len(date_range)),  # 5～10の範囲でランダムに生成
    'IsHoliday': np.random.choice([False, True], size=len(date_range), p=[0.9, 0.1])  # 90%がFalse、10%がTrue
})

# 元のデータとダミーデータを結合
combined_data = pd.concat([data, dummy_data])

# 結合したデータを新しいCSVファイルに出力
output_file_path = 'data/combined_salesdata.csv'  # 出力するCSVファイルパス
combined_data.to_csv(output_file_path, index=False)

print(f"データが {output_file_path} に保存されました。")
