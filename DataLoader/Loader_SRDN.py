from torch.utils.data import DataLoader
from util.util_read_label_xml import _read_label_voc
import os
import torch
import numpy as np
import cv2
from PIL import Image
import random
from util.util_CCPD import _crop_licience_plante
from util.util_JPEG_compression import Jpegcompress2
from torchvision import transforms as T
from torch.utils.data._utils.collate import default_collate
from prefetch_generator import BackgroundGenerator
from util.util_data_aug import Dataaug


class Loader(DataLoader):
    """
    Load data with DataLoader.
    """

    def __init__(self, cfg):
        super(Loader, self).__init__(object)
        self.cfg = cfg
        self.one_test = cfg.TEST.ONE_TEST
        self.one_name = cfg.TEST.ONE_NAME
        self.train_batch_num = 100
        self.test_batch_num = 1
        self.targets = []
        self.transform_toTensor = T.ToTensor()
        self.Data_aug = Dataaug(self.cfg)
        self.collate_fun = default_collate

    def __len__(self):
        if self.one_test:
            length = int(self.cfg.TEST.ONE_TEST_TRAIN_STEP)
        else:
            length = len(self.dataset_txt)
        return length

    def __getitem__(self, index):
        if self.one_test:
            data_info = self.dataset_txt[0]
        else:
            data_info = self.dataset_txt[index]

        labels = self._target_prepare(filename=data_info)
        imgs = self._input_prepare(target=labels, filename=data_info)



        labels = np.asarray(labels, dtype=np.float32)
        imgs = np.asarray(imgs, dtype=np.float32)
        imgs = (imgs - self.cfg.TRAIN.PIXCELS_NORM[0]) / self.cfg.TRAIN.PIXCELS_NORM[1]
        labels = (labels - self.cfg.TRAIN.PIXCELS_NORM[0]) / self.cfg.TRAIN.PIXCELS_NORM[1]
        imgs = np.transpose(imgs, (2, 0, 1))
        labels = np.transpose(labels, (2, 0, 1))
        return imgs, labels, data_info  # only need the labels

    def __iter__(self):
        '''
        原本Pytorch默认的DataLoader会创建一些worker线程来预读取新的数据，但是除非这些线程的数据全部都被清空，这些线程才会读下一批数据。

        使用prefetch_generator，我们可以保证线程不会等待，每个线程都总有至少一个数据在加载。
        :return:
        '''
        return BackgroundGenerator(super().__iter__())

    def _load_dataset(self, dataset, is_training):
        self.dataset_txt = dataset
        self.is_training = is_training

    def _target_prepare(self, **kwargs):
        id = kwargs['filename']
        if id[2] in ["", ' ', "none", "None"]:
            return 0
        target = cv2.imread(id[2])  # read faster than Image.open

        if target is None:
            print(id, 'image is None!!')
            return None
        if self.cfg.TRAIN.TARGET_TRANSFORM:  # and self.is_training
            # add the pre deal programs.
            target, _ = self.Data_aug.augmentation(aug_way_ids=([20, 22], [25]), datas=([target], None))
            target = target[0]
        return target

    def _input_prepare(self, **kwargs):
        target = kwargs['target']
        id = kwargs['filename']
        if self.cfg.TRAIN.INPUT_FROM_TARGET or self.cfg.TRAIN.TARGET_TRANSFORM:
            input = target
            if self.cfg.TRAIN.MODEL not in ['cbdnet', 'dncnn']:  # 去噪网络'cbdnet', 'dncnn'就不用缩小尺寸
                input = cv2.resize(input, (self.cfg.TRAIN.IMG_SIZE[1] // self.cfg.TRAIN.UPSCALE_FACTOR,  # SR model 使用
                                           self.cfg.TRAIN.IMG_SIZE[0] // self.cfg.TRAIN.UPSCALE_FACTOR))
        else:
            input = cv2.imread(id[1])

        if self.cfg.TRAIN.INPUT_TRANSFORM:  # and self.is_training and not self.cfg.TRAIN.TARGET_TRANSFORM:
            # add the augmentation ...
            input, _ = self.Data_aug.augmentation(aug_way_ids=([5, 6, 7, 11, 12, 13, 14,15, 16], []), datas=([input], None))
            input = input[0]
            compress_level = random.randint(5, 20)
            input = Jpegcompress2(input, compress_level)

        if self.cfg.TRAIN.MODEL in ['cbdnet', 'srcnn', 'vdsr'] and input.shape != target.shape:  # 输入与输出一样大小
            input = cv2.resize(input, target.shape)
        return np.asarray(input, dtype=np.float32)
