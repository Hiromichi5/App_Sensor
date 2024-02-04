import csv
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from matplotlib.animation import FuncAnimation

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
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    ani = FuncAnimation(fig, plot_hist, fargs=(ax,), interval=f_sampling * 1000)  # Set interval in milliseconds
    plt.show()

def plot_hist(i, ax):
    global start, end, e, mean_100
    if i == 0:
        e = sum(data[i:i+hist_range])
    else:
        plus_begin = int((i-1) * 0.25 / f_sampling) + hist_range
        plus = sum(data[plus_begin:plus_begin+5])
        plus = sum(data[end:end+5])
        minus = sum(data[start:start+5])
        end = end + 5
        start = start + 5
        e = e - minus + plus 
    hist.append(e)

    time_axis = np.arange(0, len(hist) * f_sampling, f_sampling)
    
    ax.clear()
    ax.plot(time_axis, hist, color='b', label='作業効率')
    ax.axhline(y=np.mean(hist), color='g', linestyle='--', label='平均')
    ax.legend()
    ax.set_title('作業効率の変化')
    ax.set_xlabel('時間（秒）')
    ax.set_xticks(np.arange(0, len(hist) * f_sampling + 1, 10))  # Set ticks every 10 seconds
    ax.set_xlim(0, len(hist) * f_sampling)
    ax.set_xticklabels(np.arange(0, len(hist) * f_sampling + 1, 10))  # Display seconds as labels

if __name__ == "__main__":
    main()
