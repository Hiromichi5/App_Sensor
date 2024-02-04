import matplotlib
matplotlib.use('TkAgg')  # Tkinterをバックエンドとして使用

import face
import graph
import threading
import time


def program1():
    face.main()

def program2():
    graph.main()


if __name__ == "__main__":
    # スレッドを生成してそれぞれのプログラムを実行
    thread1 = threading.Thread(target=program1)
    thread2 = threading.Thread(target=program2)

    # スレッドを開始    
    thread1.start()
    thread2.start()

    # 両方のスレッドが終了するまで待機
    thread1.join()
    thread2.join()
