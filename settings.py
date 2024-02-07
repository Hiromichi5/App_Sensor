# settings.py
# 定数設定ファイル

# サンプル数 マル5,バツ5の場合10
NUM_SAMPLES = 20

# MAC アドレス
MAC_ADRESS = ["EF:7D:EE:22:15:8B", # 1
            "F9:41:F3:3B:21:24", # 2
            "ED:60:85:42:09:A9", # 3
            "EB:7A:92:A7:79:D0", # 4
            "EB:0E:8D:A0:F6:8C"] # 5

# 閾値
MIN_THRE = 0.01
MAX_THRE = 0.2
STEP_THRE = 0.02

# 待機時間
WAITING_TIME = 2.0
TEST_TIME = 3000

# ウィンドウサイズ
WINDOW_SIZE = 50 # 0.5秒

# ファイル構成

# acc_train - acc_data_training_マル1.csv
#           - acc_data_training_マル2.csv

# acc_train_extract - acc_data_training_ex_マル1.csv
#                   - acc_data_training_ex_マル2.csv

# static - img - マル1 - threshold.csv
#                     - マル1_0.5.png
#                     - マル1_1.0.png
#              - マル2 - threshold.csv
#                     - マル2_0.5.png
#                     - マル2_1.0.png