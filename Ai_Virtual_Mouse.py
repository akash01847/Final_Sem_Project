import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui


def virtual_mouse():
    wCam = 640
    hCam = 480
    frameR = 50
    smoothening = 50

    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    cap.set(10, 160)
    detector = htm.handDetector(maxHands=1)
    wScr, hScr = pyautogui.size()
    print(wScr, hScr)

    while True:
        # find the landmark
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        # prevent abortion of program through failsafe
        pyautogui.FAILSAFE = False

        # get the tip of index and middle finger.
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            # check which finger is up.
            fingers = detector.fingersUp()
            # print(fingers)
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
            # Only index finger : Moving Mode
            if fingers[1] == 1 and fingers[2] == 0:
                # convert coordinates
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                # smoothing value
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # move mouse
                pyautogui.moveTo(wScr - x3, y3)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # if both Index and Middle finger are up : Click mode
            if fingers[0] == 1 and fingers[1] == 1:
                length, img, lineinfo = detector.findDistance(8, 12, img)
                print(length)
                if length < 40:
                    cv2.circle(img, (lineinfo[4], lineinfo[5]), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.click(button='left')

            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineinfo = detector.findDistance(8, 12, img)
                print(length)
                if length < 40:
                    cv2.circle(img, (lineinfo[0], lineinfo[1]), 15, (255, 255, 0), cv2.FILLED)
                pyautogui.click(button='right')

        # frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    virtual_mouse()
