# learning.py

# 機械学習によって、分類器を作成

import pandas as pd
import glob
import sys
import os

# CSVファイルを読み込む
df_thre = pd.read_csv('threshold.csv')

# 'None'またはNaN値をデフォルト値に置き換える
df_thre['threshold'].fillna(0, inplace=True)
list_mark = ['マル1','マル2','マル3']
list_thre = [0.5,1.0,1.0]
print(df_thre)
# for mark ,threshold in zip(df_thre['マーク'],df_thre['threshold']):
for mark ,threshold in zip(list_mark,list_thre):
    print(mark,",",threshold)
    file_path = os.path.join('static','img',mark,'threshold.csv')
    df_tmp = pd.read_csv(file_path)
    tmp = df_tmp[df_tmp['threshold'] == threshold]
    # df_thre[mark]['start'] == tmp['start']
    # df_thre[mark]['end'] == tmp['end']
    # 開始時間と終了時間の列をマージ
    if not tmp.empty:
        df_thre[mark + '_start'] = tmp['start'].values[0]
        df_thre[mark + '_end'] = tmp['end'].values[0]
    else:
        # 対応するデータがない場合はNoneまたは適切なデフォルト値を設定
        df_thre[mark + '_start'] = None
        df_thre[mark + '_end'] = None
print(df_thre)
sys.exit()

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
