from settings import EFFICIENCY_FRAME,EFFICIENCY_TIME,EFFICIENCY_NEAR_FRAME,EFFICIENCY_STEP_TIME
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# 作業効率リスト
Efficiency = []
Time = [EFFICIENCY_TIME]
time = EFFICIENCY_TIME
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1]})
fig.canvas.draw_idle()
plt.pause(0.1) 

def update_graph(result):# resultが300以上になったらstart
    global time 
    num_scoring = count_clusters(result[-EFFICIENCY_FRAME:])
    Efficiency.append(num_scoring)
    mean = np.mean(Efficiency)
    # グラフ作成
    ax1.plot(Time, Efficiency, color='b', label='作業効率')
    time += EFFICIENCY_STEP_TIME
    Time.append(time)
    ax1.axhline(y=mean, color='g', linestyle='--', label='平均')
    ax1.legend()
    ax1.set_title('作業効率の変化')
    ax1.set_xlabel('時間（秒）')

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
    plt.pause(0.1)  # グラフを再描画

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

plt.show(block=False)  # 最初に1回だけグラフを非ブロッキングモードで表示


