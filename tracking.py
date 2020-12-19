import prediction
import cv2 as cv2
import numpy as np
import time

model = prediction.model()
cap = cv2.VideoCapture(0)

pen_img = cv2.resize(cv2.imread('pen.png' ,1) , (50,50))
eraser_img = cv2.resize(cv2.imread('earaser.jpg',1),(50,50))
predict_img  = cv2.resize(cv2.imread('gun-pointer.png',1),(50,50))

# Kernerl for cleaning of th images.
kernel = np.ones((5,5) , np.uint8)

# For contour we need to filter noise
noiseth = 500 # Threshold noise values.

#We will draw on a canvas and then merge with the main.
canvas = None

#Creating a background subtractor.
backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

#Threshold for background
background_threshold = 600

#Switch for determinations.
switch = 'pen'
switch_new = 'noshot'

# Time for chanigng the button.
last_switch = time.time()
last_switch_predict = time.time()

# Coordinates for starting . 
x1,y1 = 0,0 

while True:
    ret ,frame = cap.read()
    frame = cv2.flip(frame , 1)

    # Assigning a blackscreen into the canvas.
    if canvas is None:
        canvas = np.zeros_like(frame)
    
    #Background Subtractoe at top_left and top_right.
    top_left = frame[0:50 , 0:50]
    top_right = frame[0:50 , 500:550]
    fgmask = backgroundobject.apply(top_left)
    fgmask_new = backgroundobject.apply(top_right)

    # Disrupter analysis.(analysis of hand behind photo)
    switch_thresh = np.sum(fgmask==255)
    switch_thresh_new = np.sum(fgmask_new==255)

    # Cnditioning.
    if switch_thresh > background_threshold and (time.time()-last_switch) > 1:
        # updating last time.
        last_switch = time.time()

        #updating values.
        if switch == 'pen':
            switch = 'eraser'
        else:
            switch = 'pen'
    
    if switch_thresh_new > background_threshold and (time.time() - last_switch_predict) > 1:
        last_switch_predict = time.time()

        if switch_new == 'noshot':
            switch_new = 'takeshot'
        else:
            switch_new = 'takeshot'
    
    # Converting BGR to HSV for working.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red colour ranges.
    lower_range = np.array([99,107,106])
    upper_range = np.array([255,255,255])

    # Creating the mask
    mask = cv2.inRange(hsv , lower_range , upper_range)

    # Erosion for eating white sapces.
    mask = cv2.erode(mask,kernel,iterations=2)

    # Dilation to expand the erosion.
    mask = cv2.dilate(mask , kernel , iterations=2)

    # Finding the contours.
    contours , hierarchy = cv2.findContours(mask , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)

    # Now determinig how contours and it should be more than threshold.
    if contours and cv2.contourArea(max(contours , key = cv2.contourArea)) > noiseth:
        # getting the maximum contour
        c = max(contours , key = cv2.contourArea)

        # getting the coordinates of contour.
        x,y,w,h = cv2.boundingRect(c)

        #Drawing the box around it.
        cv2.rectangle(frame , (x,y) , (x+w , y+h) , (0,25,255),2)

        #getting the co-ordinates for the canvas.
        x2,y2 = x,y

        if x1 == 0 and y1 == 0:
            x1,y1 = x2,y2 # Starting Conditions and dissapearence.
        else:
            if switch == 'pen':
                canvas = cv2.line(canvas , (x1,y1) , (x2,y2) , (255,0,255) , 10)
            else:
                cv2.circle(canvas , (x2,y2) , 20, (0,0,0) , -1)
        x1,y1 = x2,y2
    
    else:
        x1 , y1 = 0 , 0
    
    #Smooth Drawing.(**)
    _ , mask = cv2.threshold(cv2.cvtColor (canvas , cv2.COLOR_BGR2GRAY) , 20 , 255 ,cv2.THRESH_BINARY)
    foreground = cv2.bitwise_and(canvas , canvas , mask=mask)
    background = cv2.bitwise_and(frame , frame , mask = cv2.bitwise_not(mask))

    #Frame adding
    frame = cv2.add(foreground,background)

    #Changing Script.
    if switch != 'pen':
        cv2.circle(frame, (x1, y1), 20, (255,255,255), -1)
        frame[0: 50, 0: 50] = eraser_img
    else: 
        frame[0: 50, 0: 50] = pen_img
    
    if switch_new != 'noshot':
        frame[0:50 , 500:550] = predict_img
        roi = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi,(28,28),interpolation = cv2.INTER_AREA)
        roi = np.array(roi)
        answer = model.predictions(roi)
        print(answer)
        #print('shot acquired')
        switch_new = 'noshot'
    else:
        frame[0:50 , 500:550] = predict_img

    cv2.imshow('screen' , frame)
    cv2.imshow('canvas' , canvas)

    k = cv2.waitKey(5) & 0xFF

    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
