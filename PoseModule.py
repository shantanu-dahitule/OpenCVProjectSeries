import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=False, trackCon=False):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):

# mpDraw = mp.solutions.drawing_utils
# mpPose = mp.solutions.pose
# pose = mpPose.Pose()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
    #print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
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
    cap = cv2.VideoCapture(0)
    pTime=0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=True) # to  draw the landmarks
        print(lmList)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

        

if __name__ == "__main__":
    main()