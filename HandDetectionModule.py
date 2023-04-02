import cv2 
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode = False, maxHands = 2):
        self.mode = mode 
        self.maxHands = maxHands

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands) # create an object of Hands class It only uses RGB images
        self.mpDraw = mp.solutions.drawing_utils # to draw the landmarks of the hand

    def findHand(self,img, draw=True):
        #img = cv2.flip(img, 1)  # flip the image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# convert the image to RGB
        self.results = self.hands.process(imgRGB) # process the image
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, " ",  lm)
                # Finding position in the form of Pixel
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id,' ',cx,' ', cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 8,(255,0,255),cv2.FILLED)
        return lmList
    
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()  # read the image 
        detector.findHand(img)
        lmList = detector.findPosition(img)
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

if __name__ == "__main__":
    main()
