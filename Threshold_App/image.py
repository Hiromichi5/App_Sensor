import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import japanize_matplotlib
import os
import sys

# 現在のスクリプトのディレクトリを取得
current_directory = os.path.dirname(__file__)
# 親ディレクトリのパスを取得
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
# 親ディレクトリをモジュール検索パスに追加
sys.path.append(parent_directory)
from settings import MIN_THRE,MAX_THRE,STEP_THRE,NUM_SAMPLES

# グラフを保存する関数
def save_img(csv_file,mark,num):
    thre_list = [round(x, 2) for x in list(np.arange(MIN_THRE, MAX_THRE+STEP_THRE, STEP_THRE))]
    dict = [] # 閾値に対する始終
    data = pd.read_csv(csv_file) # CSVファイルからデータを読み込む
    data = data.iloc[5:] # 初め5データを削除

    for threshold in thre_list:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce') # タイムスタンプ列をdatetimeオブジェクトに変換
        data = data.dropna(subset=['Timestamp']) # タイムスタンプでエラーが発生した行を削除
        start_time = data['Timestamp'].iloc[0] # 最初のタイムスタンプを基準として経過時間（秒）を計算
        data['Elapsed'] = (data['Timestamp'] - start_time).dt.total_seconds() # 0秒から始まるように変更

        # 加速度の絶対値を計算
        data['abs_acc'] = np.sqrt(data['X']**2 + data['Y']**2 + data['Z']**2)
        # 加速度の変化を計算
        data['acc_change'] = abs(data['abs_acc'].diff())
        # 加速度変化に基づいて 'movement' フラグを設定する
        data['movement'] = data['acc_change'] > threshold

        # グラフの描画
        plt.figure(figsize=(10, 6))
        plt.plot(data['Elapsed'], data['X'], label='X軸')
        plt.plot(data['Elapsed'], data['Y'], label='Y軸')
        plt.plot(data['Elapsed'], data['Z'], label='Z軸')

        # 動きが検出された区間を検索
        movements = data['movement']
        movement_starts = data['Elapsed'][movements & ~movements.shift(1).fillna(False)]
        movement_ends = data['Elapsed'][movements & ~movements.shift(-1).fillna(False)]
        time_starts = data['Timestamp'][movements & ~movements.shift(1).fillna(False)]
        time_ends = data['Timestamp'][movements & ~movements.shift(-1).fillna(False)]

        # 動きの最初と最後の位置を取得
        if not movement_starts.empty and not movement_ends.empty:
            first_movement_start = movement_starts.iloc[0]
            last_movement_end = movement_ends.iloc[-1]
            first = time_starts.iloc[0]
            last = time_ends.iloc[-1]
            #ファイル書き込み
            dict.append({"threshold":threshold,"start":first,"end":last})
            # 静止状態と動きがあった状態の間で背景色を変える
            plt.axvspan(0, first_movement_start, color='blue', alpha=0.1, label='静止')
            plt.axvspan(first_movement_start, last_movement_end, color='orange', alpha=0.3, label='運動')
            plt.axvspan(last_movement_end, data['Elapsed'].iloc[-1], color='blue', alpha=0.1)
        else:
            # 動きが検出されなかった場合、全体を静止状態とする
            plt.axvspan(0, data['Elapsed'].iloc[-1], color='blue', alpha=0.1, label='静止')
            dict.append({"threshold":threshold,"start":None,"end":None})


        # X軸の範囲を0から2秒までに設定
        plt.xlim(0, 2)

        # ラベルとタイトルの設定
        plt.xlabel('時間(秒)')
        plt.ylabel('加速度(mG)')
        plt.title(str(mark)+str(num)+'回目 閾値:' + str(threshold) )
        plt.legend()

        if not os.path.exists('static/img/'+str(mark)+str(num)):
            os.makedirs('static/img/'+str(mark)+str(num))
        file_name = 'static/img/{}{}/{}{}_{}.png'.format(mark, num, mark, num, "{:.2f}".format(threshold))
        plt.savefig(file_name)
        plt.close()
        
    #　閾値に対するstart,end位置のファイルを保存
    csv_file = 'static/img/'+str(mark)+str(num)+'/threshold.csv'
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["threshold", "start", "end"])
        writer.writeheader()
        for row in dict:
            writer.writerow(row)

if __name__ == "__main__":
    #　全てのデータに対して画像作成・保存
    for session in range(1,NUM_SAMPLES+1):
        mark_array = ["バツ","マル"]
        num = round(session/2+0.1)
        mark = mark_array[session%2]
        csv_file = os.path.join(parent_directory,'acc_train','acc_data_training_'+str(mark)+str(num)+'.csv')
        print(str(mark)+str(num)+'画像生成中')
        save_img(csv_file,mark,num)
    

    # mark  = 'バツ'
    # num = 1
    # csv_file = 'acc_train/acc_data_training_'+str(mark)+str(num)+'.csv'
    # print(str(mark)+str(num)+'画像生成中')
    # save_img(csv_file,mark,num)
    # # plot_acc_data(csv_file,1.0)