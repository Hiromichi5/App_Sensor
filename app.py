# app.py

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import japanize_matplotlib

app = Flask(__name__)

@app.route('/')
def index():
    # index.htmlをレンダリングする
    return render_template('index.html')

@app.route('/update_graph', methods=['POST'])
def update_graph():
    # フロントエンドから送信された閾値を取得
    threshold = float(request.form['threshold'])
    
    # 加速度データをグラフに表示する関数を実行
    img = io.BytesIO()  # グラフの画像を保持するバイナリストリーム
    plot_acc_data('Train/acc_data_training_マル2.csv', threshold, img)
    
    # グラフの画像をBase64にエンコードしてHTMLに埋め込む
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return jsonify({'image_url': 'data:image/png;base64,{}'.format(plot_url)})

# 加速度データをグラフで表示する関数
def plot_acc_data(csv_file, threshold, img):
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

    # グラフをバイナリストリームに保存
    plt.savefig(img, format='png')
    plt.close()

if __name__ == '__main__':
    app.run(debug=True)
