import numpy as np
import cv2
import os
import numpy as np

AngleModelPb = '/home/dell/lg/code/lg_pro_sets/saved/checkpoint/Angle-model.pb'
AngleModelPbtxt = '/home/dell/lg/code/lg_pro_sets/saved/checkpoint/Angle-model.pbtxt'
angle_net = cv2.dnn.readNetFromTensorflow(AngleModelPb, AngleModelPbtxt)  ##dnn 文字方向检测


def angle_detect(img, angle_net, adjust=False):
    cv2.imshow('img', img)
    cv2.waitKey()
    h, w = img.shape[:2]
    ROTATE = [0, 90, 180, 270]
    if adjust:
        thesh = 0.05
        xMin, yMin, xMax, yMax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
        img = img[yMin: yMax, xMin: xMax]  # cut the edge of image
    inputBlob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(224, 224), swapRB=True,
                                      mean=[103.939, 116.779, 123.68], crop=False)
    angle_net.setInput(inputBlob)
    pred = angle_net.forward()
    index = np.argmax(pred, axis=1)[0]
    angle = ROTATE[index]
    if angle == 90:
        img = cv2.transpose(img)
        img = cv2.flip(img, flipCode=0)  # counter clock wise
    elif angle == 180:
        img = cv2.flip(img, flipCode=-1)  # flip the image both horizontally and vertically
    elif angle == 270:
        img = cv2.transpose(img)
        img = cv2.flip(img, flipCode=1)  # clock wise rotation
    cv2.imshow('img', img)
    cv2.waitKey()
    return angle


import cv2
import sys
import time
import numpy as np

from skimage.transform import radon


def rotate_paper(img):
    # Load file, converting to grayscale
    t1 = time.time()
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = I.shape
    # If the resolution is high, resize the image to reduce processing time.
    # lower resolution, lower process time.
    adjust = 1
    if adjust:
        thesh = 0.3
        xMin, yMin, xMax, yMax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
        I = I[yMin: yMax, xMin: xMax]  # cut the edge of image

    size = 320
    if (w > size):
        I = cv2.resize(I, (size, int((h / w) * size)))

    I = I - np.mean(I)  # Demean; make the brightness extend above and below zero
    # Do the radon transform
    sinogram = radon(I)
    # Find the RMS value of each row and find "busiest" rotation,
    # where the transform is lined up perfectly with the alternating dark
    # text and white lines
    r = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
    rotation = np.argmax(r)
    print('Rotation: {:.2f} degrees'.format(90 - rotation))

    # Rotate and save with the original resolution
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 90 - rotation, 1)
    t2 = time.time()
    print(t2 - t1)
    dst = cv2.warpAffine(img, M, (w, h))
    return dst


if __name__ == '__main__':
    # imgdirs = '/media/dell/data/ocr/计费清单识别/仿真数据'
    imgdirs = '/media/dell/data/ocr/计费清单识别/仿真数据正向/images'
    for imgp in os.listdir(imgdirs):
        img = cv2.imread(os.path.join(imgdirs, imgp))
        # angle_detect(img, angle_net, adjust=False)
        ro_img = rotate_paper(img)
        ro_img = cv2.resize(ro_img, None, fx=0.2, fy=0.2)
        img = cv2.resize(img, None, fx=0.2, fy=0.2)
        imgshow = np.concatenate((img, ro_img), 1)
        # cv2.imshow('img',imgshow)
        # cv2.waitKey()
        cv2.imwrite(os.path.join(imgdirs, '..', 'ladon2',imgp), imgshow)

