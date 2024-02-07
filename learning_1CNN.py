# learning_1CNN.py

# 1次元CNNによって、分類器を作成

import pandas as pd
import numpy as np
import glob
import sys
import os
import math
import matplotlib.pyplot as plt
import feature
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from joblib import dump
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D, Input, Reshape

start = []
end = []

# 設定した閾値による抽出すべき時間をまとめたdf_threを作成
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
interval_maru, interval_batu = [], []

label = []

def get_data():
    # 特徴量
    global label
    train = []
    list = ['マル1','マル2','マル3','マル4','マル5','マル6','マル7','マル8','マル9','マル10',
            'バツ1','バツ2','バツ3','バツ4','バツ5','バツ6','バツ7','バツ8','バツ9','バツ10']
    window = []
    merged_train = pd.DataFrame()
    for mark in list:
        file = os.path.join('acc_train','acc_data_training_'+mark+'.csv')
        df_tmp = pd.read_csv(file)
        df_tmp['Timestamp'] = pd.to_datetime(df_tmp['Timestamp'])
        start = df_thre[df_thre['マーク'] == mark]['start'].values[0]
        start = pd.to_datetime(start)
        end = df_thre[df_thre['マーク'] == mark]['end'].values[0]
        end = pd.to_datetime(end)
        df = pd.DataFrame()
        df = df_tmp[(df_tmp['Timestamp'] >= start) & (df_tmp['Timestamp'] <= end)][['X','Y','Z']]
        train.append(df)
        merged_train = pd.concat([merged_train, df], ignore_index=True)
        window.append(len(df))
        if 'マル' in mark:
            label.append(0)
        elif 'バツ' in mark:
            label.append(1)
        else:
            label.append(np.nan)
    padding_size = np.max(window)

    # スケーリング
    scaler = StandardScaler()
    scaler.fit(merged_train)  # まるとばつの全サンプルを含むデータセットにフィットさせる
    # scaled_samples = scaler.transform(merged_train)  # スケーリングを適用
    # print(scaled_samples)
    # sys.exit()
    # ゼロパディング
    train_padding = []
    for df in train:
        df_std = pd.DataFrame(scaler.transform(df),columns=df.columns) # スケーリング
        diff_size = padding_size - len(df_std)
        padding_df = pd.DataFrame([[0, 0, 0]] * diff_size, columns=["X", "Y", "Z"])
        # データフレームを結合する
        merged_df = pd.concat([df_std, padding_df], ignore_index=True)
        train_padding.append(merged_df)
    
    print("window = ",window)
    input_data = np.array(train_padding)
    print(input_data)
    print(input_data.shape)
    return input_data




if __name__ == "__main__":
    data = get_data()
    X_train, X_test, y_train, y_test = train_test_split(data, np.array(label), test_size=0.2, random_state=42)

    input_dim = X_train.shape[1] * X_train.shape[2]  # 特徴量の数 * タイムステップ数
    encoding_dim = 32  # エンコーディングの次元

    # オートエンコーダーの定義
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    flattened = Flatten()(input_layer)  # 入力を平滑化
    encoded = Dense(encoding_dim, activation='relu')(flattened)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    decoded = Reshape((X_train.shape[1], X_train.shape[2]))(decoded)  # 出力層の形状を元の入力データと同じにする
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # オートエンコーダーのトレーニング
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

    # 再構築誤差の計算
    reconstruction_error = np.mean(np.power(X_train - autoencoder.predict(X_train), 2), axis=1)

    # 閾値の設定(99%分位数）
    threshold = np.quantile(reconstruction_error, 0.99)

    print(f"設定された閾値: {threshold}")
    # テストデータ
    reconstructed_data = autoencoder.predict(X_test)
    reconstruction_errors = np.mean(np.power(X_test - reconstructed_data, 2), axis=1)
    anomalies = (reconstruction_errors > threshold).astype(int)

    print(anomalies)

    sys.exit()


    # モデルの定義
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(65, 3)), # 畳み込み層（入力層含む）
        MaxPooling1D(pool_size=2), # プーリング層
        Dropout(0.5),
        Conv1D(filters=128, kernel_size=3, activation='relu'), # 2つ目の畳み込み層
        MaxPooling1D(pool_size=2),# プーリング層
        Dropout(0.5),
        Flatten(), # 平滑化層
        Dense(100, activation='relu'), # 全結合層
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # 出力層: シグモイド活性化関数を使用
    ])

    # モデルのコンパイル: バイナリクロスエントロピーとAdamオプティマイザを使用
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # モデルのサマリー表示
    model.summary()

    # モデルのトレーニング
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # テストデータに対する予測値を計算
    y_pred = model.predict(X_test)
    print(y_pred)
    y_pred = np.round(y_pred).astype(int)  # 予測確率を0または1のクラスラベルに変換

    # F1スコアを計算
    f1 = f1_score(y_test, y_pred)
    print('F1 Score:', f1)
