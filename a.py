import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 更新する配列を作成する
data = np.random.rand(10)

# プロットの初期化
fig, ax = plt.subplots()
line, = ax.plot(np.arange(len(data)), data)

# 更新する関数を定義する
def update(frame):
    data[frame] = np.random.rand()  # データを更新する
    line.set_ydata(data)  # ラインを更新する
    return line,

# アニメーションを作成する
ani = FuncAnimation(fig, update, frames=len(data), blit=True, interval=200)  # フレームレートを調整する

plt.show()