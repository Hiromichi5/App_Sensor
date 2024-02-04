# learning.py

# 機械学習によって、分類器を作成

import pandas as pd
import numpy as np
import glob
import sys
import os
import math
import matplotlib.pyplot as plt
import feature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle  
start = []
end = []

# 設定した閾値による抽出すべき時間をまとめたdf_threを作成
# CSVファイルを読み込む
df_thre = pd.read_csv('threshold.csv')
print(df_thre)
for mark ,threshold in zip(df_thre['マーク'],df_thre['threshold']):
    if math.isnan(threshold):
        print(str(mark)+'のthresholdがNaNです。')
        start.append(np.nan)
        end.append(np.nan)
        # start, endカラムにnanを追加
    else:
        # start, endカラムに時間を追加
        file = os.path.join('static','img',str(mark),'threshold.csv')
        df_tmp = pd.read_csv(file)
        print(mark,",",threshold)
        s = df_tmp[df_tmp['threshold'] == threshold]['start'].values[0]
        e = df_tmp[df_tmp['threshold'] == threshold]['end'].values[0]
        start.append(s)
        end.append(e)
df_thre['start'] = start
df_thre['end'] = end

# 特徴量
df = feature.feature_create(df_thre)
# 特徴量とクラスラベルの分離
X_train= df.drop(columns=["label","move"])  # 特徴量
y_train= df["label"]  # クラスラベル
# 標準化
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns, index=X_train.index)

# マルとバツ　それぞれのクラスター中心を算出
maru_rows = X_train_std[X_train_std.index.str.contains('マル')]
# batu_rows = X_train_std[X_train_std.index.str.contains('バツ')]
print(maru_rows)
# print(batu_rows)

class_center = maru_rows.mean()
print(class_center)
# print(class_center)
sys.exit()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# モデルをファイルに保存
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# # モデルのロード
# with open('model.pkl', 'rb') as model_file:
#     loaded_model = pickle.load(model_file)

# predicted = loaded_model.predict(X)

# X_test = 
# X_test = scaler.transform(X_test)

# クラスター中心との距離

# y_pred = model.predict(X_test)

# 特徴量とクラスラベルの分離
# df.columns = df.columns.astype(str)
# X = df.drop(columns=["label"])  # 特徴量
# y = df["label"]  # クラスラベル
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)



