from siamrpn import TrackerSiamRPN
import cv2
import os
import numpy as np
from glob import glob

'''
the performance on the test dataset is very good. will be useful !
'''


def load_bbox(ground_file, resize, dataformat=0):
    f = open(ground_file)
    lines = f.readlines()
    bbox = []
    for line in lines:
        if line:
            pt = line.strip().split(',')
            pt_int = [float(ii) for ii in pt]
            bbox.append(pt_int)
    bbox = np.array(bbox)
    if resize:
        bbox = (bbox.astype('float32') / 2).astype('int')
    else:
        bbox = bbox.astype('float32').astype('int')
    if dataformat:
        bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
        bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
    return bbox


def load_imglst(img_dir):
    file_lst = [pic for pic in os.listdir(img_dir) if '.jpg' in pic]
    img_list = [os.path.join(img_dir, filename) for filename in file_lst]
    return img_list


if __name__ == '__main__':
    OTB100 = 0
    LG = 1

    if OTB100:
        dataset = 'E:/datasets/TRACK/OTB100/BlurCar2/'
        save_directory = './'
        show_result = 1
        img_list = load_imglst(dataset + '/img/')
        bbox_lst = load_bbox(os.path.join(dataset + '/groundtruth_rect.txt'), 0, dataformat=0)
        x1, y1, w, h = bbox_lst[0]
        roi = (x1, y1, w, h)
    elif LG:
        # img_path = 'E:\datasets\TRACK\LG\images/img_'
        img_path = 'E:\\for_test\\fly1'
        img_list = []
        img_list = glob(img_path + '/*.png')
        # for i in range(0, 967):
        #     img_list.append(img_path + str(i) + '.jpg')

    net_path = '../../../saved/checkpoint/SimRPN.pth'
    tracker = TrackerSiamRPN(net_path)

    frames = len(img_list)
    frame1 = cv2.imread(img_list[0])
    roi = cv2.selectROI("tracking", frame1, False, False)  # x1, y1, w, h)
    # roi = (917, 547, 103, 106)
    tracker.init(frame1, roi)
    j = 1
    for i in range(1, frames):
        frame = cv2.imread(img_list[i])
        if frame is None:
            continue
        x1, y1, w, h = tracker.update(frame)
        x1, y1, w, h = int(x1), int(y1), int(w), int(h)
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
        cv2.imshow('tracking', frame)
        # cv2.imwrite(img_path + '/tracker/%06d.png' % i, frame)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            break
    cv2.destroyAllWindows()
