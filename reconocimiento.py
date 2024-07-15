import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Dense(128, input_dim=42, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.load_weights('./weights4c.h5')

def normalize_keypoints(kp):
  for i in range(0, len(kp)):
      non_zero_x = kp[i][0:21][kp[i][0:21] != 0]
      non_zero_y = kp[i][21:42][kp[i][21:42] != 0]

      min_val = [np.min(non_zero_x), np.min(non_zero_y)]
      max_val = [np.max(non_zero_x), np.max(non_zero_y)]

      kp[i][0:21] = (kp[i][0:21] - min_val[0]) / (max_val[0] - min_val[0])

      kp[i][21:42] = (kp[i][21:42] - min_val[1]) / (max_val[1] - min_val[1])
  return kp


def recon(keypoints):
    keypoints = keypoints.reshape(1, -1)
    keypoints = normalize_keypoints(keypoints)
    sign = model.predict(keypoints, verbose = 0)
    return np.argmax(sign)