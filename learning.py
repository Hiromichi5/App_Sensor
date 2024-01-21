# learning.py

# 機械学習によって、分類器を作成

import pandas as pd
import glob

# CSVファイルのパスを取得
file_paths = glob.glob('Train/*.csv')

# すべてのCSVファイルを1つのデータフレームにまとめる
all_data = pd.DataFrame()

for file_path in file_paths:
    # CSVファイルを読み込む
    df = pd.read_csv(file_path)

    # データフレームにファイル名の情報を追加
    filename = file_path.split('/')[-1]
    df['File'] = filename

    # 全体のデータフレームに追加
    all_data = pd.concat([all_data, df], ignore_index=True)

# データフレームを出力
print(all_data)

# CSVファイルに保存する場合は以下のようにします
# all_data.to_csv('merged_data.csv', index=False)
