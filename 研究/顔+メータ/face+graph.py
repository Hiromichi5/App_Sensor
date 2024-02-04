#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np
import multiprocessing
import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # これを追加
from matplotlib.animation import FuncAnimation
import japanize_matplotlib
import threading

# 以下にコードを挿入

f = open('true_out.csv', 'r')
data = f.read().split(", ")
data = list(map(int, data))
f.close()

T = 60
f_sampling = 0.05
h = 20
update_time = 30
e = 0
mean_100 = 0

hist_range = int(T / f_sampling)
start = 0
end = hist_range

update_range = int(update_time / f_sampling)
hist = []

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
    mean = np.mean(hist)
    time_axis = np.arange(0, (len(hist) - 1) * f_sampling, f_sampling)

    # Left subplot (histogram)
    ax1.clear()
    time_axis = np.arange(0, len(hist) * f_sampling, f_sampling)
    ax1.plot(time_axis, hist, color='b', label='作業効率')  
    ax1.axhline(y=mean, color='g', linestyle='--', label='平均')
    ax1.legend()
    ax1.set_title('作業効率の変化')
    ax1.set_xlabel('時間（秒）')
    ax1.set_xticks(np.arange(0, len(hist) * f_sampling, 10))  
    ax1.set_xlim(0, len(hist) * f_sampling)
    ax1.set_xticklabels(np.arange(0, len(hist) * f_sampling, 10))

    # Right subplot (text display)
    ax2.clear()
    ax2.text(0, 0.8, "評価：", size=30)
    mean_100 = np.mean(hist[(len(hist)-100):])
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
    ax2.text(0, 0.2, "全体の平均：" + (str(round(mean, 2)) if mean is not None else "---"), size=30)
    ax2.axis("off")

def main():
    output_file = open("output.txt", "a")  # テキストファイルを開く
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1]})

    # スレッド1: グラフ表示
    def graph_thread():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1]})
        ani = FuncAnimation(fig, plot_hist, fargs=(ax1, ax2), frames=len(data), interval=f_sampling * 1000)
        plt.show()

    # スレッド2: カメラ表示
    def camera_thread():
        # カメラの設定
        DEVICE_ID = 0
        capture = cv2.VideoCapture(DEVICE_ID)
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)

        while True:
            ret, frame = capture.read()
            frame = imutils.resize(frame, width=1000)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            image_points = None

            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

                image_points = np.array([
                    tuple(shape[30]),
                    tuple(shape[21]),
                    tuple(shape[22]),
                    tuple(shape[39]),
                    tuple(shape[42]),
                    tuple(shape[31]),
                    tuple(shape[35]),
                    tuple(shape[48]),
                    tuple(shape[54]),
                    tuple(shape[57]),
                    tuple(shape[8]),
                ], dtype='double')

            if len(rects) > 0:
                model_points = np.array([
                    (0.0, 0.0, 0.0),
                    (-30.0, -125.0, -30.0),
                    (30.0, -125.0, -30.0),
                    (-60.0, -70.0, -60.0),
                    (60.0, -70.0, -60.0),
                    (-40.0, 40.0, -50.0),
                    (40.0, 40.0, -50.0),
                    (-70.0, 130.0, -100.0),
                    (70.0, 130.0, -100.0),
                    (0.0, 158.0, -10.0),
                    (0.0, 250.0, -50.0)
                ])
                size = frame.shape
                focal_length = size[1]
                center = (size[1] // 2, size[0] // 2)

                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype='double')

                dist_coeffs = np.zeros((4, 1))

                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                              dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
                mat = np.hstack((rotation_matrix, translation_vector))

                (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)
                yaw, pitch, roll = eulerAngles[1], eulerAngles[0], eulerAngles[2]

                dt_now = str(datetime.datetime.now())
                output_line = f"Time {dt_now}, yaw {int(yaw)}, pitch {int(pitch)}, roll {int(roll)}"
                print(output_line)
                output_file.write(output_line + "\n")

                cv2.putText(frame, 'Time : ' + dt_now, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                cv2.putText(frame, 'yaw : ' + str(int(yaw)), (20, 35), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                cv2.putText(frame, 'pitch : ' + str(int(pitch)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                cv2.putText(frame, 'roll : ' + str(int(roll)), (20, 65), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

                (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector,
                                                          translation_vector, camera_matrix, dist_coeffs)

                for p in image_points:
                    cv2.drawMarker(frame, (int(p[0]), int(p[1])), (0.0, 1.409845, 255),
                                   markerType=cv2.MARKER_CROSS, thickness=1)

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                cv2.arrowedLine(frame, p1, p2, (255, 0, 0), 2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        output_file.close()
        capture.release()
        cv2.destroyAllWindows()

    # スレッドの開始
    thread_graph = threading.Thread(target=graph_thread)
    thread_camera = threading.Thread(target=camera_thread)

    thread_graph.start()
    thread_camera.start()

def graph_process():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1]})
    ani = FuncAnimation(fig, plot_hist, fargs=(ax1, ax2), frames=len(data), interval=f_sampling * 1000)
    plt.show()
if __name__ == "__main__":
    # グラフプロセスを開始
    graph_process = multiprocessing.Process(target=graph_process)
    graph_process.start()

    # カメラスレッドはメインスレッドで続行
    camera_thread()