import argparse
import time

import cv2
from kcf import Tracker


def main():
    save_reslut = 0
    # vid_path = 'car.avi'
    vid_path = '/media/luogeng/ssd_datasets/code/uav_code/datasets/track_video.mp4'
    cap = cv2.VideoCapture(vid_path)
    tracker = Tracker()
    ok, frame = cap.read()
    if not ok:
        print("error reading video")
        exit(-1)
    roi = cv2.selectROI("tracking", frame, False, False)
    # roi = (218, 302, 148, 108) (967, 409, 128, 70) (1047, 497, 120, 85)(886, 491, 118, 71)(1043, 495, 133, 92)
    tracker.init(frame, roi)
    i = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        t1 = time.time()
        (x, y, w, h), response = tracker.update(frame)
        t2 = time.time() - t1
        print('time:', t2, "frame:",1/t2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.putText(frame, '%.5f' % response, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('tracking', frame)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            break

        if save_reslut:
            imgpath = 'demo_result/images/%05d.jpg' % i
            labpath = 'demo_result/labels/%05d.txt' % i
            cv2.imwrite(imgpath, frame)
            f = open(labpath, 'w')
            txt = ','.join(list(map(str, [x, y, w, h]))) + '\n'
            f.write(txt)
            f.close()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
