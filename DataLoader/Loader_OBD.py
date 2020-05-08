import os
import torch
import random
import cv2
from util.util_data_aug import Dataaug
import numpy as np
import xml.etree.ElementTree as ET
from util.util_get_cls_names import _get_class_names
from util.util_show_img import _show_img
from torch.utils.data import DataLoader


class Loader(DataLoader):
    def __init__(self, cfg):
        super(Loader, self).__init__(object)
        self.cfg = cfg
        self.dataaug = Dataaug(cfg)
        self.one_test = cfg.TEST.ONE_TEST
        self.one_name = cfg.TEST.ONE_NAME
        self.train_batch_num = 100
        self.test_batch_num = 1
        self.class_name = _get_class_names(cfg.PATH.CLASSES_PATH)
        self.print_path = self.cfg.TRAIN.SHOW_TRAIN_NAMES
        self.cls2idx = dict(zip(self.cfg.TRAIN.CLASSES, range(len(self.cfg.TRAIN.CLASSES))))
        self.a = 0

    def __len__(self):
        if self.one_test:
            if self.is_training:
                length = int(self.cfg.TEST.ONE_TEST_TRAIN_STEP)
            else:
                length = len(self.cfg.TEST.ONE_NAME)
        else:
            length = len(self.dataset_txt)
        return length

    def __getitem__(self, index):
        if self.one_test:
            data_info = self.dataset_txt[0]
        else:
            data_info = self.dataset_txt[index]

        img, label = self._read_datas(data_info)
        if self.cfg.TRAIN.DO_AUG and self.is_training:
            imgs, labels = self.dataaug.augmentation(([img], [label]))
            img = imgs[0]
            label = labels[0]
        if self.cfg.TRAIN.RESIZE:
            size = random.choice(self.cfg.TRAIN.MULTI_SIZE_RATIO) * self.cfg.TRAIN.IMG_SIZE
            img_i_size = img.shape
            img = cv2.resize(img, (size[1], size[0]))
            if self.cfg.TRAIN.RELATIVE_LABELS:
                label_after = [[lab[0],
                                lab[1] / img_i_size[1],
                                lab[2] / img_i_size[0],
                                lab[3] / img_i_size[1],
                                lab[4] / img_i_size[0]
                                ] for lab in label]
            else:
                label_after = [[lab[0],
                                lab[1] / img_i_size[1] * size[1],
                                lab[2] / img_i_size[0] * size[0],
                                lab[3] / img_i_size[1] * size[1],
                                lab[4] / img_i_size[0] * size[0]
                                ] for lab in label]
        else:
            label_after = label
        if self.cfg.TRAIN.SHOW_INPUT:
            _show_img(img, label_after, show_img=True, cfg=self.cfg, show_time=self.cfg.TRAIN.SHOW_INPUT)
        img = np.asarray(img, dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = img / 127.5 - 1.
        return img, label_after, data_info  # only need the labels

    def _load_dataset(self, dataset, is_training):
        self.dataset_txt = dataset
        self.is_training = is_training

    def _read_datas(self, id):
        '''
        Read images and labels depend on the idx.
        :param image: if there is a image ,the just return the image.
        :return: images, labels, image_size
        '''
        if self.print_path:
            print(id[1], '<--->', id[2])
        if not (os.path.isfile(id[1]) and os.path.isfile(id[2])):
            print('ERROR, NO SUCH A FILE.', id[1], '<--->', id[2])
            exit()
        # labels come first.
        label = self._read_line(id[2])
        # then add the images.
        img = cv2.imread(id[1])
        return img, label

    def _is_finedata(self, xyxy):
        x1, y1, x2, y2 = xyxy
        for point in xyxy:
            if point <= 0.: return False
        if x2 - x1 <= 0.: return False
        if y2 - y1 <= 0.: return False
        return True

    def _read_line(self, path, predicted_line=False, pass_obj=['DontCare', ]):
        """
        Parse the labels from file.

        :param pass_obj: pass the labels in the list.
                        e.g. pass_obj=['Others','Pedestrian', 'DontCare']
        :param path: the path of file that need to parse.
        :return:lists of the classes and the key points.
        """
        bbs = []
        if os.path.basename(path).split('.')[-1] == 'txt' and not predicted_line:
            file_open = open(path, 'r')
            for line in file_open.readlines():
                if 'UCAS_AOD' in self.cfg.TRAIN.TRAIN_DATA_FROM_FILE:
                    tmps = line.strip().split('\t')
                    box_x1 = float(tmps[9])
                    box_y1 = float(tmps[10])
                    box_x2 = box_x1 + float(tmps[11])
                    box_y2 = box_y1 + float(tmps[12])
                    if not self._is_finedata([box_x1, box_y1, box_x2, box_y2]): continue
                    bbs.append([1, box_x1, box_y1, box_x2, box_y2])
                elif 'VISDRONE' in self.cfg.TRAIN.TRAIN_DATA_FROM_FILE:
                    name_dict = {'0': 'ignored regions', '1': 'pedestrian', '2': 'people',
                                 '3': 'bicycle', '4': 'car', '5': 'van', '6': 'truck',
                                 '7': 'tricycle', '8': 'awning-tricycle', '9': 'bus',
                                 '10': 'motor', '11': 'others'}
                    tmps = line.strip().split(',')
                    realname = name_dict[tmps[5]]
                    if realname not in self.class_name:
                        continue
                    if realname in pass_obj:
                        continue
                    box_x1 = float(tmps[0])
                    box_y1 = float(tmps[1])
                    box_x2 = box_x1 +float(tmps[2])
                    box_y2 = box_y1 +float(tmps[3])

                    if not self._is_finedata([box_x1, box_y1, box_x2, box_y2]): continue
                    bbs.append([self.cls2idx[self.class_name[realname]], box_x1, box_y1, box_x2, box_y2])
                else:
                    tmps = line.strip().split(' ')
                    if tmps[0] not in self.class_name:
                        continue
                    if self.class_name[tmps[0]] in pass_obj:
                        continue
                    box_x1 = float(tmps[4])
                    box_y1 = float(tmps[5])
                    box_x2 = float(tmps[6])
                    box_y2 = float(tmps[7])
                    if not self._is_finedata([box_x1, box_y1, box_x2, box_y2]): continue
                    bbs.append([self.cls2idx[self.class_name[tmps[0]]], box_x1, box_y1, box_x2, box_y2])
        elif os.path.basename(path).split('.')[-1] == 'xml' and not predicted_line:
            tree = ET.parse(path)
            root = tree.getroot()
            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                if cls_name not in self.class_name:
                    continue
                if self.class_name[cls_name] in pass_obj:
                    continue
                bbox = obj.find('bndbox')
                box_x1 = float(bbox.find('xmin').text)
                box_y1 = float(bbox.find('ymin').text)
                box_x2 = float(bbox.find('xmax').text)
                box_y2 = float(bbox.find('ymax').text)
                if not self._is_finedata([box_x1, box_y1, box_x2, box_y2]): continue
                bbs.append([self.cls2idx[self.class_name[cls_name]], box_x1, box_y1, box_x2, box_y2])

        elif os.path.basename(path).split('.')[-1] == 'txt' and predicted_line:
            f_path = open(path, 'r')
            print(path)
            for line in f_path.readlines():
                tmp = line.split()
                cls_name = tmp[0]
                bbs.append([float(tmp[1]), self.cls2idx[self.class_name[cls_name]], [float(tmp[2]), float(tmp[3]), float(tmp[4]), float(tmp[5])]])
        return bbs

    def collate_fun(self, batch):
        '''
        collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
        其中default_collate会将labels分割合并转换成tensor。
        !!!***if not use my own collect_fun ,the labels will be wrong orders.***
        :param batch:
        :return:
        '''
        imgs, labels, infos = zip(*batch)
        imgs = torch.from_numpy(np.asarray(imgs))
        return imgs, list(labels), list(infos)
