# test.py

# 採点の有無、マルバツの有無を推定

from __future__ import print_function
from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import *

from settings import NUM_SAMPLES,MAC_ADRESS,TEST_TIME,STOP_FRAME,MOVE_FRAME,STEP_SIZE,EFFICIENCY_STEP 
from settings import EFFICIENCY_FRAME,EFFICIENCY_TIME,EFFICIENCY_NEAR_FRAME,EFFICIENCY_STEP_TIME
import threading
from time import sleep, time
import csv
from datetime import datetime, timedelta
import pytz
import warnings
import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Tkinter バックエンドを使用
import matplotlib.pyplot as plt
import japanize_matplotlib
import sys
import pickle
from joblib import load
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # WARNINGとINFOを非表示にする

if not os.path.exists(os.path.join('acc_test')):
    os.makedirs(os.path.join('acc_test'))

# # 更新する配列を作成する
# data = np.random.rand(10)

# # プロットの初期化
# fig, ax = plt.subplots()
# line, = ax.plot(np.arange(len(data)), data)
counter_sliding, counter_graph= 0, 0
x, y, z, xyz = [], [], [], []
pred = []

# 警告を無視する
warnings.filterwarnings('ignore')

# モデルのロード
with open(os.path.join('Model_App','model','Autoencoder','autoencoderマル.pkl'), 'rb') as model_file:
    autoencoder_maru = pickle.load(model_file)
with open(os.path.join('Model_App','model','Autoencoder','autoencoderバツ.pkl'), 'rb') as model_file:
    autoencoder_batu = pickle.load(model_file)
    
# スケーラーをロード
scaler = load(os.path.join('Model_App','model','Autoencoder','scaler_autoencoder.joblib'))
df = pd.read_csv(os.path.join('Model_App','model','Autoencoder','threshold.csv'))
threshold_auto_maru = df['threshold_auto_maru'][0]
threshold_auto_batu = df['threshold_auto_batu'][0]
threshold_move = df['threshold_move'][0]
padding_size_maru = df['padding_size_maru'][0]
padding_size_batu = df['padding_size_batu'][0]
window_size = max([padding_size_maru,padding_size_batu])

# 作業効率リスト
Efficiency = []
Time = [EFFICIENCY_TIME]
t = EFFICIENCY_TIME
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1]})
fig.canvas.draw_idle()
plt.pause(0.1) 
plt.show(block=False) 
# plt.show(block=False)
# fig.canvas.draw_idle()
# plt.pause(0.1) 

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
        # print("%s -> {x : %.3f, y : %.3f, z : %.3f}" % (self.device.address, acc_data.x, acc_data.y, acc_data.z))
        
        # 現在のUTC時間を取得
        current_time_utc = datetime.utcfromtimestamp(time()).replace(tzinfo=pytz.utc)

        # 日本時間に変換
        japan_tz = pytz.timezone('Asia/Tokyo')
        current_time = current_time_utc.astimezone(japan_tz)
        
        # CSV ファイルにデータを書き込む
        self.csv_writer.writerow([self.device.address, self.samples, current_time, acc_data.x, acc_data.y, acc_data.z])
        if len(x) > window_size:
            global counter_graph, counter_sliding
            x.pop(0)
            y.pop(0)
            z.pop(0)
            xyz.pop(0)
            counter_sliding += 1
            counter_graph += 1
            if counter_sliding == STEP_SIZE:
                result = classify(x,y,z,xyz)
                pred.append(result)
                print(result)
                counter_sliding = 0
            if counter_graph == EFFICIENCY_STEP:
                # computation_thread = threading.Thread(target=update_graph, args=(result,))
                # computation_thread.start()
                update_graph(result)
        x.append(acc_data.x)
        y.append(acc_data.y)
        z.append(acc_data.z)
        xyz.append(np.sqrt(acc_data.x**2+acc_data.y**2+acc_data.z**2))
        
        self.samples += 1

def update_graph(result):# resultが300以上になったらstart
    global t

    num_scoring = count_clusters(result[-EFFICIENCY_FRAME:])
    Efficiency.append(num_scoring)
    mean = np.mean(Efficiency)

    # グラフ作成
    ax1.clear()
    ax1.plot(Time, Efficiency, color='b', label='作業効率')
    ax1.axhline(y=mean, color='g', linestyle='--', label='平均')
    ax1.legend()
    ax1.set_title('作業効率の変化')
    ax1.set_xlabel('時間（秒）')

    t += EFFICIENCY_STEP_TIME
    Time.append(t)


    ax2.clear() 
    ax2.text(0, 0.8, "評価：", size=30)
    mean_near = np.mean(Efficiency[-EFFICIENCY_NEAR_FRAME:]) if len(Efficiency)>=EFFICIENCY_NEAR_FRAME else 0
    E_current = Efficiency[-1]
    if mean_near != 0:
        if mean < E_current:
            if mean_near < E_current:
                ax2.text(0, 0.8, "　　　Excellent", size=30, color='r')
            else:
                ax2.text(0, 0.8, "　　　Good", size=30, color='y')
        else:
            if mean_near < E_current:
                ax2.text(0, 0.8, "　　　Poor", size=30, color='g')
            else:
                ax2.text(0, 0.8, "　　　Bad", size=30, color='b')
    else:
        ax2.text(0, 0.8, "　　　---", size=30)
    ax2.text(0, 0.6, "作業効率：" + str(E_current), size=30)
    ax2.text(0, 0.4, "直近の平均：" + str(mean_near), size=30)
    ax2.text(0, 0.2, "全体の平均：" + (str(round(mean, 2)) ), size=30)
    ax2.axis("off")

    fig.canvas.draw_idle()
    plt.pause(0.001)  # グラフを再描画

def count_clusters(arr):
    cluster_count = 0
    zero_count = 0
    in_cluster = False
    for num in arr:
        if num == '○' or num == '×' or num == '○×':
            if zero_count >= 5 or not in_cluster:
                cluster_count += 1
                in_cluster = True
            zero_count = 0   # 1を見つけたら、ゼロカウントをリセット
        else:
            zero_count += 1
            if zero_count >= 5:
                in_cluster = False  # もし0が5回以上続いたら、塊の終わりとマーク
    return cluster_count

def classify(x,y,z,xyz):
    # 入力データ作成
    data = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
    df_std_maru = scaler.transform(data[0:int(padding_size_maru)])
    df_std_batu = scaler.transform(data[0:int(padding_size_batu)])
    df_tmp_maru = np.array([df_std_maru])
    df_tmp_batu = np.array([df_std_batu])
    # 静止検出
    move = []
    for i in range(STOP_FRAME):
        move.append(abs(xyz[i+1]-xyz[i]) > threshold_move)
    if sum(move) > MOVE_FRAME:
        # マル異常検知
        reconstruction_errors = np.mean(np.power(df_tmp_maru - autoencoder_maru.predict(df_tmp_maru,verbose=0), 2), axis=1)
        anomalies = (reconstruction_errors > threshold_auto_maru).astype(int) # 異常検知 → 1, 正常 → 0
        judge_maru = 0 if sum(anomalies[0]) == 0 else 1
        judge_maru = int(sum(anomalies[0]) != 0)

        # バツ異常検知
        reconstruction_errors = np.mean(np.power(df_tmp_batu - autoencoder_batu.predict(df_tmp_batu,verbose=0), 2), axis=1)
        anomalies = (reconstruction_errors > threshold_auto_batu).astype(int) # 異常検知 → 1, 正常 → 0
        judge_batu = int(sum(anomalies[0]) != 0)
        judge = judge_maru + judge_batu

        if judge == 0: # マルかつバツ
            # return 1
            return '○×' 
        elif judge == 1: # マルあるいはバツ
            if judge_maru == 1:
                # return 1
                return '○'
            elif judge_batu ==1:
                # return 1
                return '×'
            else:
                return 'Error1'
        elif judge == 2: # 異常検知
            # return 0
            return '異常'
        else:
            return 'Error2'
    else:
        # return 0
        return '静止'

    
# テストデータ取得
def test():

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
        folder_name = "acc_test"
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
        csv_filename = os.path.join(folder_name, f'acc_data_test.csv')

        with open(csv_filename, mode='w', newline='') as file:
            # CSV ファイルを開いてヘッダーを書き込む
            csv_header = ["Device Address", "Sample Number", "Timestamp", "X", "Y", "Z"]
            writer = csv.writer(file)
            writer.writerow(csv_header)
            states[0].samples = 0  # セッションごとにサンプル数をリセット
            states[0].csv_writer = writer
            sleep(0.5)
            key_pressed = input(f"\nテスト開始(y/n): ").lower()

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
                sleep(TEST_TIME) 

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
                file.close()
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
    test()
    # check_data()