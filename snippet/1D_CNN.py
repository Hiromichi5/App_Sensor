import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D

# n_timesteps : シーケンスの長さ（時間ステップの数）
# n_features : 特徴の数（3軸加速度センサーの場合は3）
# n_classes : 分類するクラスの数

# モデルの定義
model = Sequential()

# 畳み込み層
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
# プーリング層
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))

# 畳み込み層
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
# プーリング層
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))

# 平滑化層
model.add(Flatten())

# 全結合層
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
# 出力層
model.add(Dense(n_classes, activation='softmax'))

# モデルのコンパイル
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# モデルのサマリー表示
model.summary()
