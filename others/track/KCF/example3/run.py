import argparse
import cv2
from    kcf import Tracker

if __name__ == '__main__':
    vid_path = 'F:\Projects\\auto_Airplane\TS02/saved_video20191217_153659_02_3min.avi'
    cap = cv2.VideoCapture(vid_path)
    tracker = Tracker()
    ok, frame = cap.read()
    if not ok:
        print("error reading video")
        exit(-1)
    roi = cv2.selectROI("tracking", frame, False, False)
    # roi = (218, 302, 148, 108)
    tracker.init(frame, roi)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        (x, y, w, h), response = tracker.update(frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.imshow('tracking', frame)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
