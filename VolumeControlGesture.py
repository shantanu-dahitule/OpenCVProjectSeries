import cv2
import time
import numpy as np
#####################
import HandDetectionModule as hd
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
#####################

detector = hd.HandDetector(maxHands = 1)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
VolRange = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-20.0, None)

minVol = VolRange[0]
maxVol = VolRange[1]
BarVol = 400
VolCent = 0
cap = cv2.VideoCapture(0)
pTime = 0
while True:
    success, img = cap.read()
    img = detector.findHand(img, draw=False)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist)!=0:
        x1,y1 = lmlist[4][1], lmlist[4][2]
        x2,y2 = lmlist[8][1], lmlist[8][2]

        cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255,0,255), cv2.FILLED)

        cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 3)
        cv2.circle(img, ((x1+x2)//2,(y1+y2)//2), 15, (255,0,255), cv2.FILLED)
        lt = []
        length = np.hypot(x2-x1,y2-y2)
        # print(length)
        # Hand range 40 - 300
        # Volume Range -65 - 0
        vol = np.interp(length,[50,300],[minVol,maxVol])
        if length>50 and length<300: 
            BarVol = np.interp(length,[50,300],[400,150])
            VolCent = np.interp(length,[50,300],[0,100])
            volume.SetMasterVolumeLevel(vol, None)
        print(length," - ",vol)
        if length<50:
            cv2.circle(img, ((x1+x2)//2,(y1+y2)//2), 15, (0,255,0), cv2.FILLED)
    
    cv2.rectangle(img, (50,150), (85,400), (0,255,0), 3)
    cv2.rectangle(img, (50,int(BarVol)), (85,400), (0,255,0), cv2.FILLED)
    cv2.putText(img, f"{str(int(VolCent))}%", (40,140), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 3)
    cv2.putText(img, f"FPS: {str(int(fps))}", (40,70), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
                break

cap.release()
cv2.destroyAllWindows()