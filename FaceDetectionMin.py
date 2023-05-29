import cv2 
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpFace = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFace.FaceDetection(0.75)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            bboxC = detection.location_data.relative_bounding_box
            ih,iw,ic = img.shape
            bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih),\
                int(bboxC.width*iw), int(bboxC.height*ih)
            cv2.rectangle(img, bbox, (255,0,255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(20,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()