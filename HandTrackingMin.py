# Hand tracking Module using mediapipe
import cv2 
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2) # create an object of Hands class It only uses RGB images
mpDraw = mp.solutions.drawing_utils # to draw the landmarks of the hand

pTime = 0
cTime = 0
while True:
    success, img = cap.read()  # read the image 
    #img = cv2.flip(img, 1)  # flip the image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# convert the image to RGB
    results = hands.process(imgRGB) # process the image
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: # loop through all the hands
            for id, lm in enumerate(handLms.landmark):
                # print(id, " ",  lm)
                # Finding position in the form of Pixel
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id,' ',cx,' ', cy)
                # if id==12:
                cv2.circle(img, (cx,cy), 25,(255,0,255),-1)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # draw the landmarks of the hand refer line 10
    

    # calculating FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,0), 2)


    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
