from flask import Flask, render_template, request, jsonify
import os
import csv
import sys
import image
# 現在のスクリプトのディレクトリを取得
current_directory = os.path.dirname(__file__)
# 親ディレクトリのパスを取得
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
# 親ディレクトリをモジュール検索パスに追加
sys.path.append(parent_directory)
from settings import MIN_THRE,MAX_THRE,STEP_THRE,NUM_SAMPLES

app = Flask(__name__)
mark = None
number = None

@app.route('/')
def index():
    # index.htmlをレンダリングする
    return render_template('index.html',slider_min=MIN_THRE, 
                           slider_max=MAX_THRE, 
                           slider_step=STEP_THRE, 
                           slider_value=MIN_THRE)

@app.route('/update_graph', methods=['POST'])
def update_graph():
    # フロントエンドから送信された閾値を取得
    global mark,number
    threshold = request.form['threshold']
    number = request.form.get('Number')
    mark = request.form.get('mark')
    # 対応する閾値の画像を取得
    #image_url = f'static/img/マル{round_number}/マル{round_number}_{threshold}.png'
    image_url = f'static/img/{mark}{number}/{mark}{number}_{threshold}.png'
    # 対応する閾値の画像を辞書から取得
    # image_url = 'static/img/マル1/マル1_'+str(threshold)+'.png'
    # グラフの画像のURLをレスポンスとして返す
    return jsonify({'image_url': image_url})

@app.route('/save_threshold', methods=['POST'])
def save_threshold():
    global mark, number
    threshold = request.form['threshold']
    round_label = f'{mark}{number}'

    # CSVファイルを読み込んでリストに保存
    with open('threshold.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)

    # 対応する行を更新
    for row in data:
        if row[0] == round_label:
            row[1] = threshold
            break

    # 更新されたデータでCSVファイルを書き込み
    with open('threshold.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    return jsonify({'status': 'success'})

@app.route('/get_thresholds', methods=['GET'])
def get_thresholds():
    thresholds = []
    with open('threshold.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            # 各行が少なくとも2つの要素を持っていることを確認
            if len(row) >= 2:
                # "None" という文字列をそのまま使用
                value = row[1] if row[1] != "None" else None
                thresholds.append(f'{row[0]}:{value}')
            else:
                # 不足している要素を持つ行に対する処理
                pass

    return jsonify(thresholds)

def create_directory():
    if not os.path.exists('static/img'):
        os.makedirs('static/img')

if __name__ == '__main__':
    create_directory()
    #　全てのデータに対して画像作成・保存
    for session in range(1,NUM_SAMPLES+1):
        mark_array = ["バツ","マル"]
        num = round(session/2+0.1)
        mark = mark_array[session%2]
        csv_file = os.path.join(parent_directory,'acc_train','acc_data_training_'+str(mark)+str(num)+'.csv')
        print(str(mark)+str(num)+'画像生成中')
        image.save_img(csv_file,mark,num)
    app.run(debug=True, port=8080, use_reloader=False)
