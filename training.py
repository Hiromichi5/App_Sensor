# training.py

# 丸とバツを交互に書いてトレーニングデータを取得

# from __future__ import print_function
# from mbientlab.metawear import MetaWear, libmetawear, parse_value
# from mbientlab.metawear.cbindings import *
from settings import NUM_SAMPLES,MAC_ADRESS,WAITING_TIME
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

    mac = MAC_ADRESS[int(sys.argv[1]) - 1]
    print("MACアドレス : ", mac)

    session = 1  # 初回セッション
    states = []  # State インスタンスのリストを初期化

    try:
        # フォルダ名
        folder_name = "acc_train"
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
        while session <= NUM_SAMPLES:
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
                    sleep(WAITING_TIME) 

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

# データの確認      
def check_data():
    # フォルダ名
    folder_name = "Train"
    # フォルダ内の全てのCSVファイルのリストを取得
    files = [file for file in os.listdir(folder_name) if file.endswith('.csv')]
    for file in files:
        # CSVファイルのフルパスを作成
        file_path = os.path.join(folder_name, file)
        # データを読み込み、リストに追加
        df = pd.read_csv(file_path)
        print(file,":",len(df))

if __name__ == "__main__":
    # get_training()
    print("--- 保存結果 ---")
    check_data()