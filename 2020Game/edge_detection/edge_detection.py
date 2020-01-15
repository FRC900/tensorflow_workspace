import cv2
import numpy as np
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255.0
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    theta =  127.5*(np.arctan2(sobelY, sobelX)/np.pi + 1)
    cv2.imshow("frame", np.uint8(np.sqrt(sobelX**2 + sobelY**2)*255))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break