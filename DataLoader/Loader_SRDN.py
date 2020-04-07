from torch.utils.data import DataLoader
from util.util_read_label_xml import _read_label_voc
import os
import torch
import numpy as np
import cv2
import random
from util.util_CCPD import _crop_licience_plante
from util.util_data_aug import Dataaug
from util.util_JPEG_compression import Jpegcompress2


class Loader(DataLoader):
    """
    Load data with DataLoader.
    """

    def __init__(self, cfg, dataset_txt):
        super(Loader, self).__init__(object)
        self.cfg = cfg
        self.one_test = cfg.TEST.ONE_TEST
        self.one_name = cfg.TEST.ONE_NAME
        self.train_batch_num = 100
        self.test_batch_num = 1
        self.Data_aug = Dataaug(self.cfg)
        self.targets = []
        self.dataset_txt = dataset_txt

    def __len__(self):
        return len(self.dataset_txt)

    def __getitem__(self, index):
        if self.one_test:
            data_info = self.dataset_txt[0]
        else:
            data_info = self.dataset_txt[index]
        img, lab = self._prepare_data(data_info)

        img = np.asarray(img, dtype=np.float32)
        lab = np.asarray(lab, dtype=np.float32)
        img = torch.from_numpy(img)
        label = torch.from_numpy(lab)
        img = img.to(self.cfg.TRAIN.DEVICE)
        label = label.to(self.cfg.TRAIN.DEVICE)
        img = (img - self.cfg.TRAIN.PIXCELS_NORM[0]) / self.cfg.TRAIN.PIXCELS_NORM[1]
        label = (label - self.cfg.TRAIN.PIXCELS_NORM[0]) / self.cfg.TRAIN.PIXCELS_NORM[1]
        img = img.permute([2, 0, 1])
        return (img, label)  # only need the labels

    def _prepare_data(self, idx, is_training=False):
        target = self._target_prepare(filename=idx)
        input = self._input_prepare(target=target, filename=idx, is_training=is_training)

        if self.cfg.TRAIN.SHOW_INPUT:
            cv2.imshow('img', input)
            cv2.waitKey(self.cfg.TRAIN.SHOW_INPUT)
        return input, target

    def _target_prepare(self, **kwargs):

        id = kwargs['filename']
        target = cv2.imread(id[2])  # no norse image or HR image
        if target is None:
            print(id, 'image is wrong!!')
            exit()
        if self.cfg.TRAIN.TARGET_PREDEAL:
            # add the pre deal programs.
            # target = _crop_licience_plante(target, id[0])
            target, _ = self.Data_aug.augmentation(for_one_image=[target])
            target = target[0]
            pass

        target = cv2.resize(target, (self.cfg.TRAIN.IMG_SIZE[0], self.cfg.TRAIN.IMG_SIZE[1]))

        return target

    def _input_prepare(self, **kwargs):
        target = kwargs['target']
        id = kwargs['filename']
        is_training = kwargs['is_training']
        input_is_target = True  # id[1] in ['None', '', ' ', 'none']

        if input_is_target:
            input = target
        else:
            input = cv2.imread(id[1])

        if self.cfg.TRAIN.MODEL not in ['cbdnet', 'dbpn', 'dncnn']:  # 去噪网络就不用缩小尺寸
            input = cv2.resize(input, (self.cfg.TRAIN.IMG_SIZE[0] // self.cfg.TRAIN.UPSCALE_FACTOR,  # SR model 使用
                                       self.cfg.TRAIN.IMG_SIZE[1] // self.cfg.TRAIN.UPSCALE_FACTOR))

        if self.cfg.TRAIN.INPUT_AUG and is_training:
            # add the augmentation ...
            # img, _ = self.Data_aug.augmentation(for_one_image=[img])
            # img = img[0]
            # compress_level = random.randint(5, 20)
            input = Jpegcompress2(input, 10)

        if self.cfg.TRAIN.MODEL in ['cbdnet', 'srcnn', 'vdsr']:
            input = cv2.resize(input, (self.cfg.TRAIN.IMG_SIZE[0], self.cfg.TRAIN.IMG_SIZE[1]))

        return input
