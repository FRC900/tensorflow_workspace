import serial
import time
from tensorflow import keras
import numpy as np

detector = keras.models.load_model("detect_adv.h5")

s = serial.Serial("COM4", 9600)
while True:
    m = s.read_until(b"\r\n")
    m = np.array(list(map(float, m.decode().split(","))))
    if(len(m) == 6):
        m = m - np.mean(m)
        m = m/m.std()
        m = detector(np.expand_dims(m, 0))
        m = np.argmax(m)
        print(["Blue", "Red", "Yellow", "Green"][m])
        del m