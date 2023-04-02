import cv2
import mediapipe as mp
import time
import HandDetectionModule as htm
pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.HandDetector()
while True:
    success, img = cap.read()  # read the image 
    # detector.findHand(img)
    detector.findHand(img, draw=False)# to not draw the landmarks but still detect them and return coordinates
    # lmList = detector.findPosition(img)
    lmList = detector.findPosition(img, draw=False) # to not draw the landmarks
    if len(lmList) != 0:
        print(lmList[4])
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,0), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()