import numpy as np
import cv2


def angle_detect(img, angle_net, adjust=False):
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
    # if angle == 90:
    #     img = cv2.transpose(img)
    #     img = cv2.flip(img, flipCode=0)  # counter clock wise
    # elif angle == 180:
    #     img = cv2.flip(img, flipCode=-1)  # flip the image both horizontally and vertically
    # elif angle == 270:
    #     img = cv2.transpose(img)
    #     img = cv2.flip(img, flipCode=1)  # clock wise rotation
    return angle
