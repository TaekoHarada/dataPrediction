import pandas as pd
import numpy as np
import random

# 家具カテゴリのリスト
furniture_categories = ["Table", "Chair", "Sofa", "Cabinet", "Bed", "Desk"]

# 年と月の範囲
years = [2018, 2019, 2020, 2021, 2022, 2023]
months = list(range(1, 13))

# すべての年・月・家具カテゴリの組み合わせを作成
all_combinations = pd.DataFrame([(year, month, category) for year in years for month in months for category in furniture_categories],
                                columns=["Year", "Month", "Furniture Category"])

# ランダムな注文数量を生成し、結合する
num_entries = len(all_combinations)  # 全組み合わせの数
order_quantities = [random.randint(1, 50) for _ in range(num_entries)]  # 1〜50のランダムな数量
all_combinations['Order Quantity'] = order_quantities

# DataFrameを確認
print(all_combinations.head())  # 最初の5行を表示

# CSVに出力
all_combinations.to_csv('dummy_data.csv', index=False)
