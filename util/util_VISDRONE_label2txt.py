import cv2
import os

'''
VISDRONE:
 VID-dataset:
    sequence name.txt ->{index name of .jpg, target_index,  x, y, w, h, score, class, x,x,x }

'''

path = 'E:/datasets/VisDrone2019/VisDrone2019-VID-val/'

lables_base = os.path.join(path, 'annotations')
imgpath_base = os.path.join(path, 'sequences')
name_dict = {'0': 'ignored regions', '1': 'pedestrian', '2': 'people',
             '3': 'bicycle', '4': 'car', '5': 'van', '6': 'truck',
             '7': 'tricycle', '8': 'awning-tricycle', '9': 'bus',
             '10': 'motor', '11': 'others'}
for file in os.listdir(lables_base):
    for line in open(os.path.join(lables_base, file)).readlines():
        tmps = line.split(',')

        img_id = '%07d.jpg' % (int(tmps[0]))
        img_file = os.path.basename(file).split('.')[0]
        img_path = os.path.join(imgpath_base, img_file, img_id)

        xmin = int(tmps[2])
        ymin = int(tmps[3])
        xmax = xmin + int(tmps[4])
        ymax = ymin + int(tmps[5])

        name = name_dict[tmps[7]]

        images = cv2.imread(img_path)
        cv2.rectangle(images, (xmin, ymin), (xmax, ymax), (255, 0, 0))
        cv2.putText(images, name, (xmin, ymin), 1, 1, (0, 0, 255))
        cv2.imshow('img', images)
        cv2.waitKey()
