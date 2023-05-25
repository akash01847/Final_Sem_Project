import cv2
import mediapipe as mp
import time
import math
import numpy as np

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):

            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

class FindHands():
    def __init__(self, detection_con=0.5, tracking_con=0.5):
        self.mpHands = mp.solutions.mediapipe.python.solutions.hands
        self.hands = self.mpHands.Hands(min_detection_confidence=detection_con, min_tracking_confidence=tracking_con)
        self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
    
    def getPosition(self, img, indexes, hand_no=0, draw=True):
        lst = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) >= hand_no+1:
                for id, lm in enumerate(results.multi_hand_landmarks[hand_no].landmark):
                    for index in indexes:
                        if id == index:
                            h, w, c = img.shape
                            x, y = int(lm.x*w), int(lm.y*h)
                            lst.append((x,y))
                if draw:
                    self.mpDraw.draw_landmarks(img, results.multi_hand_landmarks[hand_no], self.mpHands.HAND_CONNECTIONS)
        return lst

    def index_finger_up(self, img, hand_no=0):
        pos = self.getPosition(img, (6,8), draw=False)
        try:
            if pos[0][1] >= pos[1][1]:
                return True
            elif pos[0][1] < pos[1][1]:
                return False
        except:
            return "NO HAND FOUND"
        
    def middle_finger_up(self, img, hand_no=0):
        pos = self.getPosition(img, (10,12), draw=False)
        try:
            if pos[0][1] >= pos[1][1]:
                return True
            elif pos[0][1] < pos[1][1]:
                return False
        except:
            return "NO HAND FOUND"

    def ring_finger_up(self, img, hand_no=0):
        pos = self.getPosition(img, (14,16), draw=False)
        try:
            if pos[0][1] >= pos[1][1]:
                return True
            elif pos[0][1] < pos[1][1]:
                return False
        except:
            return "NO HAND FOUND"

    def little_finger_up(self, img, hand_no=0):
        pos = self.getPosition(img, (18,20), draw=False)
        try:
            if pos[0][1] >= pos[1][1]:
                return True
            elif pos[0][1] < pos[1][1]:
                return False
        except:
            return "NO HAND FOUND"


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    hands = FindHands()
    while True:
        succeed, img = cap.read()
        lst = hands.getPosition(img, 8)
        for pt in lst:
            cv2.circle(img, pt, 5, (0,255,0), cv2.FILLED)
        cv2.imshow("Image", img)
        if cv2.waitKey(10) == ord("q"):
            break
