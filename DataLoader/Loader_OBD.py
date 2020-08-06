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
        self.write_images = self.cfg.TRAIN.WRITE_IMAGES

    def __len__(self):
        if self.one_test:
            if self.is_training:
                length = int(self.cfg.TEST.ONE_TEST_TRAIN_STEP)
            else:
                length = 1
        else:
            length = len(self.dataset_txt)
        return length

    def __getitem__(self, index):
        img = None
        label = None

        while img is None or label is None:  # if there is no data in img or label
            if self.one_test:
                data_info = self.dataset_txt[0]
            else:
                data_info = self.dataset_txt[index]
            img, label = self._read_datas(data_info)  # labels: (x1, y1, x2, y2) & must be absolutely labels.
            if not self.is_training and not label:
                break
            index += 1

        # DOAUG:
        if self.cfg.TRAIN.DO_AUG and self.is_training:
            labels = 'None'
            try_tims = 0
            while labels is 'None':
                imgs, labels = self.dataaug.augmentation(aug_way_ids=([11, 20, 21, 22], [26]), datas=([img], [label]))  # [11,20, 21, 22]
                try_tims += 1
                if try_tims > 100:
                    print('trying', try_tims, ' times when data augmentation at file:', str(data_info[2]))
            img = imgs[0]
            label = labels[0]

        # TEST->PAD TO SIZE:
        if self.cfg.TEST.PADTOSIZE and not self.is_training:
            img, label = self.pad_to_size(img, label, self.cfg.TRAIN.IMG_SIZE, decode=False)

        img_i_size = img.shape
        size = img_i_size
        # RESIZE:
        if (self.cfg.TRAIN.RESIZE and self.is_training) or (self.cfg.TEST.RESIZE and not self.is_training):
            size = random.choice(self.cfg.TRAIN.MULTI_SIZE_RATIO) * self.cfg.TRAIN.IMG_SIZE
            img = cv2.resize(img, (size[1], size[0]))

        # RELATIVE LABELS:
        if not label:
            label_after = None
        elif self.cfg.TRAIN.RELATIVE_LABELS:  # x1y1x2y2
            label_after = [[0,
                            lab[0],
                            lab[1] / img_i_size[1],
                            lab[2] / img_i_size[0],
                            lab[3] / img_i_size[1],
                            lab[4] / img_i_size[0]
                            ] for lab in label]
        else:
            label_after = [[0,
                            lab[0],
                            lab[1] / img_i_size[1] * size[1],
                            lab[2] / img_i_size[0] * size[0],
                            lab[3] / img_i_size[1] * size[1],
                            lab[4] / img_i_size[0] * size[0]
                            ] for lab in label]

        if self.cfg.TRAIN.SHOW_INPUT:
            _show_img(img, label_after, cfg=self.cfg, show_time=self.cfg.TRAIN.SHOW_INPUT, pic_path=data_info[2])
        if self.write_images > 0:
            img_write = _show_img(img, label_after, cfg=self.cfg,show_time=0)
            self.cfg.TRAIN.WRITER.tbX_addImage('GT_'+data_info[0], img_write)
            self.write_images -= 1
        img = np.asarray(img, dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = img / 127.5 - 1.
        if label_after: label_after = torch.Tensor(label_after)
        return img, label_after, data_info  # only need the labels  label_after[x1y1x2y2]

    def _load_dataset(self, dataset, is_training):
        self.dataset_txt = dataset
        self.is_training = is_training

    def _read_datas(self, id):
        '''
        Read images and labels depend on the idx.
        :param image: if there is a image ,the just return the image.
        :return: images, labels, image_size
        '''

        x_path = os.path.join(self.cfg.PATH.INPUT_PATH, id[1])
        y_path = os.path.join(self.cfg.PATH.INPUT_PATH, id[2])
        if self.print_path:
            print(x_path, '<--->', y_path)
        if not (os.path.isfile(x_path) and os.path.isfile(y_path)):
            print('ERROR, NO SUCH A FILE.', x_path, '<--->', y_path)
            exit()
        # labels come first.
        label = self._read_line(y_path)
        if label == [[]] or label == []:
            print('none label at:', y_path)
            label = None
        # then add the images.
        img = cv2.imread(x_path)
        return img, label

    def _is_finedata(self, xyxy):
        x1, y1, x2, y2 = xyxy
        for point in xyxy:
            if point < 0.: return False
        if x2 - x1 <= 0.: return False
        if y2 - y1 <= 0.: return False
        return True

    def _read_line(self, path, predicted_line=False, pass_obj=['DontCare', ]):
        """
        Parse the labels from file.

        :param pass_obj: pass the labels in the list.
                        e.g. pass_obj=['Others','Pedestrian', 'DontCare']
        :param path: the path of file that need to parse.
        :return:lists of the classes and the key points============ [x1, y1, x2, y2].
        """
        bbs = []
        if os.path.basename(path).split('.')[-1] == 'txt' and not predicted_line:
            file_open = open(path, 'r')
            for line in file_open.readlines():
                if 'UCAS_AOD' in path:
                    tmps = line.strip().split('\t')
                    box_x1 = float(tmps[9])
                    box_y1 = float(tmps[10])
                    box_x2 = box_x1 + float(tmps[11])
                    box_y2 = box_y1 + float(tmps[12])
                    if not self._is_finedata([box_x1, box_y1, box_x2, box_y2]): continue
                    bbs.append([1, box_x1, box_y1, box_x2, box_y2])
                elif 'VisDrone2019' in path:
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
                    box_x2 = box_x1 + float(tmps[2])
                    box_y2 = box_y1 + float(tmps[3])

                    if not self._is_finedata([box_x1, box_y1, box_x2, box_y2]): continue
                    bbs.append([self.cls2idx[self.class_name[realname]], box_x1, box_y1, box_x2, box_y2])
                elif 'TS02' in path:
                    tmps = line.strip().split(' ')
                    realname = 'car'
                    if realname not in self.class_name:
                        continue
                    if realname in pass_obj:
                        continue
                    img_w = 1920
                    img_h = 1080
                    x = float(tmps[1]) * img_w
                    y = float(tmps[2]) * img_h
                    w = float(tmps[3]) * img_w
                    h = float(tmps[4]) * img_h

                    box_x1 = x - w / 2.
                    box_y1 = y - h / 2.
                    box_x2 = x + w / 2.
                    box_y2 = y + h / 2.

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
        if isinstance(labels[0], torch.Tensor):
            for i, label in enumerate(labels):
                label[:, 0] = i
            labels = torch.cat(labels, 0)
        return imgs, labels, list(infos)

    def pad_to_size(self, img, label, size, decode=False):
        h, w, c = img.shape
        ratio_w = size[1] / w
        ratio_h = size[0] / h
        # if ratio_w
        if ratio_w < ratio_h:
            ratio_min = ratio_w
        else:
            ratio_min = ratio_h

        # resize:

        img = cv2.resize(img, None, fx=ratio_min, fy=ratio_min)
        h, w, c = img.shape
        # Determine padding
        # pad =[left, right, top, bottom]
        if abs(w - size[1]) > abs(h - size[0]):
            dim_diff = abs(w - size[1])
            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            pad = (pad1, pad2, 0, 0)
        else:
            dim_diff = abs(h - size[0])
            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            pad = (0, 0, pad1, pad2)

        # Add padding
        img = cv2.copyMakeBorder(img, pad[2], pad[3], pad[0], pad[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if not decode:
            label_after = [[
                lab[0],
                lab[1] * ratio_min + pad[0],
                lab[2] * ratio_min + pad[2],
                lab[3] * ratio_min + pad[1],
                lab[4] * ratio_min + pad[3]
            ] for lab in label]
        else:
            label_after = [[
                lab[0],
                (lab[1] - pad[0]) / ratio_min,
                (lab[2] - pad[2]) / ratio_min,
                (lab[3] - pad[1]) / ratio_min,
                (lab[4] - pad[3]) / ratio_min,
            ] for lab in label]

        return img, label_after
