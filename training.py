# training.py

# 丸とバツを交互に書いてトレーニングデータを取得

from __future__ import print_function
from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import *
from time import sleep, time
import csv
from datetime import datetime, timedelta
import pytz
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import sys

class State:
    def __init__(self, device, session):
        self.device = device
        self.samples = 0
        self.callback = FnVoid_VoidP_DataP(self.data_handler)
        self.session = session
        self.csv_writer = None

    def data_handler(self, ctx, data):
        # データハンドラー：データを表示し、サンプル数を更新
        acc_data = parse_value(data)
        print("%s -> {x : %.3f, y : %.3f, z : %.3f}" % (self.device.address, acc_data.x, acc_data.y, acc_data.z))
        
        # 現在のUTC時間を取得
        current_time_utc = datetime.utcfromtimestamp(time()).replace(tzinfo=pytz.utc)

        # 日本時間に変換
        japan_tz = pytz.timezone('Asia/Tokyo')
        current_time = current_time_utc.astimezone(japan_tz)
        
        # CSV ファイルにデータを書き込む
        self.csv_writer.writerow([self.device.address, self.samples, current_time, acc_data.x, acc_data.y, acc_data.z])
        
        self.samples += 1

# トレーニングデータ取得
def get_training():

    # 引数が指定されていない場合や、引数が 1 から 5 のいずれでもない場合はエラーメッセージを表示して終了
    valid_arguments = {'1', '2', '3', '4', '5'}
    if len(sys.argv) != 2 or sys.argv[1] not in valid_arguments:
        print("Usage: python script_name.py 1|2|3|4|5")
        sys.exit(1)

    # MAC アドレスを選択
    mac_options = ["EF:7D:EE:22:15:8B", # 1
                "F9:41:F3:3B:21:24", # 2
                "ED:60:85:42:09:A9", # 3
                "EB:7A:92:A7:79:D0", # 4
                "EB:0E:8D:A0:F6:8C"] # 5

    mac = mac_options[int(sys.argv[1]) - 1]
    print("MACアドレス : ", mac)

    session = 1  # 初回セッション
    states = []  # State インスタンスのリストを初期化

    try:
        # フォルダ名
        folder_name = "Train"
        # Train フォルダが存在しない場合は作成
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # 接続
        d = MetaWear(mac)
        d.connect()
        print("Connected to " + d.address + " over " + ("USB" if d.usb.is_connected else "BLE"))
        states.append(State(d, session))  # ここで State インスタンスを追加

        print("Configuring device")
        # BLEの設定
        libmetawear.mbl_mw_settings_set_connection_parameters(states[0].device.board, 7.5, 7.5, 0, 6000)  # ここで states[0] を使う
        sleep(1.5)
        # 加速度センサの設定
        libmetawear.mbl_mw_acc_set_odr(states[0].device.board, 100.0)
        libmetawear.mbl_mw_acc_set_range(states[0].device.board, 16.0)
        libmetawear.mbl_mw_acc_write_acceleration_config(states[0].device.board)

        # セッションごとに繰り返し
        mark_array = ["バツ","マル"]
        while session <= 10:
            num = round(session/2+0.1)
            mark = mark_array[session%2]
            csv_filename = os.path.join(folder_name, f'acc_data_training_{mark}{num}.csv')
            with open(csv_filename, mode='w', newline='') as file:
                # CSV ファイルを開いてヘッダーを書き込む
                csv_header = ["Device Address", "Sample Number", "Timestamp", "X", "Y", "Z"]
                writer = csv.writer(file)
                writer.writerow(csv_header)
                states[0].samples = 0  # セッションごとにサンプル数をリセット
                states[0].csv_writer = writer
                sleep(0.5)
                key_pressed = input(f"\nトレーニングデータ取得  {mark}{num} (y/n): ").lower()

                if key_pressed == 'n':
                    # 'n' が入力されたら前回のデータを削除して新しいセッションを始める
                    session -= 1
                    raise KeyboardInterrupt

                elif key_pressed == 'y':
                    # 加速度の取得とサブスク
                    signal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(states[0].device.board)
                    libmetawear.mbl_mw_datasignal_subscribe(signal, None, states[0].callback)
                    # 加速度の開始
                    libmetawear.mbl_mw_acc_enable_acceleration_sampling(states[0].device.board)
                    libmetawear.mbl_mw_acc_start(states[0].device.board)

                    # 待機
                    sleep(2.0)  # 5秒間待機する例

                    # 加速度の停止
                    libmetawear.mbl_mw_acc_stop(states[0].device.board)
                    libmetawear.mbl_mw_acc_disable_acceleration_sampling(states[0].device.board)

                    print(f"Data saved to {csv_filename}")
                    session += 1
                    states[0].csv_writer = None  # ファイルをクローズ
    except KeyboardInterrupt:
        print(f"Session {session} aborted. Data not saved.")
    finally:
        # 全体の終了処理
        if states:
            if states[0].csv_writer:
                states[0].csv_writer.close()
            # 購読解除
            signal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(states[0].device.board)
            libmetawear.mbl_mw_datasignal_unsubscribe(signal)
            # 切断
            libmetawear.mbl_mw_debug_disconnect(states[0].device.board)
            
# 加速度データをグラフで表示する関数
def plot_acc_data(csv_file, threshold):
    # CSVファイルからデータを読み込む
    data = pd.read_csv(csv_file)

    # タイムスタンプ列をdatetimeオブジェクトに変換
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

    # タイムスタンプでエラーが発生した行を削除
    data = data.dropna(subset=['Timestamp'])

    # 最初のタイムスタンプを基準として経過時間（秒）を計算
    start_time = data['Timestamp'].iloc[0]
    data['Elapsed'] = (data['Timestamp'] - start_time).dt.total_seconds()

    # 加速度の絶対値を計算
    data['abs_acc'] = np.sqrt(data['X']**2 + data['Y']**2 + data['Z']**2)

    # 動きがあったかどうかのフラグを設定（閾値を超える動きをしたとき）
    data['movement'] = data['abs_acc'] > threshold

    # グラフの描画
    plt.figure(figsize=(10, 6))
    plt.plot(data['Elapsed'], data['X'], label='X軸')
    plt.plot(data['Elapsed'], data['Y'], label='Y軸')
    plt.plot(data['Elapsed'], data['Z'], label='Z軸')

    # 動きが検出された区間を検索
    movements = data['movement']
    movement_starts = data['Elapsed'][movements & ~movements.shift(1).fillna(False)]
    movement_ends = data['Elapsed'][movements & ~movements.shift(-1).fillna(False)]

    # 動きの最初と最後の位置を取得
    if not movement_starts.empty and not movement_ends.empty:
        first_movement_start = movement_starts.iloc[0]
        last_movement_end = movement_ends.iloc[-1]

        # 静止状態と動きがあった状態の間で背景色を変える
        plt.axvspan(0, first_movement_start, color='blue', alpha=0.1, label='静止')
        plt.axvspan(first_movement_start, last_movement_end, color='orange', alpha=0.3, label='運動')
        plt.axvspan(last_movement_end, data['Elapsed'].iloc[-1], color='blue', alpha=0.1)
    else:
        # 動きが検出されなかった場合、全体を静止状態とする
        plt.axvspan(0, data['Elapsed'].iloc[-1], color='blue', alpha=0.1, label='静止')

    # X軸の範囲を0から2秒までに設定
    plt.xlim(0, 2)

    # ラベルとタイトルの設定
    plt.xlabel('時間(秒)')
    plt.ylabel('加速度(mG)')
    plt.title('閾値:' + str(threshold) )
    plt.legend()

    # グラフの表示
    plt.show()

if __name__ == "__main__":
    # get_training()
    list = [0.5,1.0,1.5,2.0,2.5,3.0]
    for i in list:
        plot_acc_data('Train/acc_data_training_マル2.csv',threshold=i)
    # list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    # for i in list:
    #     plot_acc_data('Train/simulated_acceleration_data.csv',threshold=i)
