import cv2
import numpy as np
import time
import HandDetectionModule as hd

detector = hd.HandDetector(maxHands = 2)
pTime = 0
FingerCount = 0
TipIds = [4,8,12,16,20]
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    img = detector.findHand(img, draw=True)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    lmlist = detector.findPosition(img, draw=False)
    #print(lmlist)
    if len(lmlist) != 0:
        fingers = []
        # For thumb
        if lmlist[TipIds[0]][1] > lmlist[TipIds[0]-2][1]:
            fingers.append(1)
    # For 4 fingers
        for id in range(1,len(TipIds)): 
            if lmlist[TipIds[id]][2] < lmlist[TipIds[id]-1][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        FingerCount = fingers.count(1)
    if FingerCount == 0:
        cv2.putText(img, "Take your right hand up", (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    else:
        cv2.putText(img, f'Finger Count: {str(int(FingerCount))}', (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,255), 3)
    cv2.putText(img, f'FPS: {str(int(fps))}', (30,210), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()
    
