import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('HSV Detector')
cv2.createTrackbar('LH','HSV Detector',0,255,nothing)
cv2.createTrackbar('LS','HSV Detector',0,255,nothing)
cv2.createTrackbar('LV','HSV Detector',0,255,nothing)
cv2.createTrackbar('UH','HSV Detector',255,255,nothing)
cv2.createTrackbar('US','HSV Detector',255,255,nothing)
cv2.createTrackbar('UV','HSV Detector',255,255,nothing)

while True:
    ret,frame = cap.read()
    if not ret:
        break
    frame2 = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    lh = cv2.getTrackbarPos('LH','HSV Detector')
    ls = cv2.getTrackbarPos('LS','HSV Detector')
    lv = cv2.getTrackbarPos('LV','HSV Detector')
    uh = cv2.getTrackbarPos('UH','HSV Detector')
    us = cv2.getTrackbarPos('US','HSV Detector')
    uv = cv2.getTrackbarPos('UV','HSV Detector')

    lb = np.array([lh,ls,lv])
    ub = np.array([uh,us,uv])

    mask = cv2.inRange(frame2,lb,ub)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (3,3))
    res = cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow('original',frame)
    cv2.imshow('mask' , mask)
    cv2.imshow('masked',res)

cap.release()
cv2.destroyAllWindows()
