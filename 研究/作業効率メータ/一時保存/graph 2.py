import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import japanize_matplotlib

f = open('true_out.csv', 'r')
data = f.read().split(", ")
data = list(map(int, data))
f.close()

T = 60  # 1分あたりの衝突回数
f_sampling = 0.05  # 1サンプル取得の時間
h = 20  # サンプル周波数
update_time = 30  # 更新する時間単位
e = 0
mean_100 = 0

hist_range = int(T / f_sampling)  # ヒストグラムにする幅
start = 0
end = hist_range

update_range = int(update_time / f_sampling)
hist = []

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1]})
    ani = FuncAnimation(fig, plot_hist, fargs=(ax1, ax2), interval=f_sampling * 1000, save_count=len(data))  # Set interval in milliseconds
    plt.show()

def plot_hist(i, ax1, ax2):
    global start, end, e, mean_100
    if i == 0:
        e = sum(data[i:i+hist_range])
    else:
        plus_begin = int((i-1) * 0.25 / f_sampling) + hist_range
        plus = sum(data[plus_begin:plus_begin+5])
        minus = sum(data[start:start+5])
        end = end + 5
        start = start + 5
        e = e - minus + plus 
    hist.append(e)

    time_axis = np.arange(0, (len(hist) - 1) * f_sampling, f_sampling)
    
    # Left subplot (histogram)
    ax1.clear()
    ax1.plot(time_axis, hist[:-1], color='b', label='作業効率')
    ax1.axhline(y=np.mean(hist), color='g', linestyle='--', label='平均')
    ax1.legend()
    ax1.set_title('作業効率の変化')
    ax1.set_xlabel('時間（秒）')
    ax1.set_xticks(np.arange(0, (len(hist) - 1) * f_sampling + 1, 10))
    ax1.set_xlim(0, (len(hist) - 1) * f_sampling)
    ax1.set_xticklabels(np.arange(0, (len(hist) - 1) * f_sampling + 1, 10))

    # Right subplot (text display)
    ax2.clear()
    ax2.text(0, 0.8, "評価：", size=30)
    if len(hist) > 100:
        if mean_100 < e:
            ax2.text(0, 0.8, "　　　Excellent", size=30, color='r')
        elif mean_100 >= e and np.mean(hist) < e:
            ax2.text(0, 0.8, "　　　Good", size=30, color='y')
        else:
            ax2.text(0, 0.8, "　　　Poor", size=30, color='g')
    else:
        ax2.text(0, 0.8, "　　　---", size=30)
    ax2.text(0, 0.6, "作業効率：" + str(e), size=30)
    if len(hist) > 100:
        ax2.text(0, 0.4, "直近の平均：" + str(mean_100), size=30)
    else:
        ax2.text(0, 0.4, "直近の平均：---", size=30)
    ax2.text(0, 0.2, "全体の平均：" + (str(round(mean_100, 2)) if mean_100 is not None else "---"), size=30)
    ax2.axis("off")

if __name__ == "__main__":
    main()
