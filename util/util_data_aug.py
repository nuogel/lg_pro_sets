"""Collection of data augumentation functions."""
import os
import random
import logging

import cv2
import torch
import imgaug as ia
from imgaug import augmenters as iaa
import xml.etree.ElementTree as ET

LOG = logging.getLogger(__name__)


class Dataaug:
    '''
    Data augmentation.
    '''

    def __init__(self, cfg):
        self.cfg = cfg
        self.img_path = cfg.PATH.INPUT_PATH
        self.lab_path = cfg.PATH.LAB_PATH
        self.area_ratio = cfg.TRAIN.AREAR_RATIO
        self.min_area = cfg.TRAIN.MIN_AREAR

    def _parse_bbs(self, bbs_aug):
        """
        Parse the key points from augmentation.
    
        :param bbs_aug: key points from augmentation.
        :param relative: the relative location of bounding boxes
        :return:labs are in the shape of [[class, left, top, button, right],..]
        """
        labs = []
        height = bbs_aug.height
        width = bbs_aug.width
        for bb_aug in bbs_aug.bounding_boxes:
            _bb_x1, _bb_y1, _bb_x2, _bb_y2, cls = bb_aug.x1, bb_aug.y1, bb_aug.x2, bb_aug.y2, bb_aug.label
            bb_x1 = min(max(1, _bb_x1), width - 1)
            bb_y1 = min(max(1, _bb_y1), height - 1)
            bb_x2 = min(max(1, _bb_x2), width - 1)
            bb_y2 = min(max(1, _bb_y2), height - 1)
            # calculate the ratio of area
            area_before = (_bb_x2 - _bb_x1) * (_bb_y2 - _bb_y1)
            area_after = (bb_x2 - bb_x1) * (bb_y2 - bb_y1)
            if area_after / area_before < self.area_ratio or area_after < self.min_area:
                continue
            # get the lab
            lab = [cls, bb_x1, bb_y1, bb_x2, bb_y2]
            labs.append(lab)
        return labs

    def _augmenting(self, datas=None, image_for_aug=None):
        """Create augmentation images from kitti.

        :param idx:  the index of kitti images ,
         index is in the shape of [1, 3, 555, 1033...]
        :param do_aug: if do_aug is False , then do nothing about images,and labels.
        :param relative: the relative location of bounding boxes
        :param image_for_aug: if there is a single image,the augment it without labels
        :return: return a np array of images, and return a list about
         labs are in the shape of [[class, left, top, button, right],..]
        """
        # print(idx)
        # prepare the augmentation functions
        images, labels = datas
        labels = [[ia.BoundingBox(x1=labs[1], y1=labs[2], x2=labs[3], y2=labs[4], label=labs[0]) for labs in _labels] for _labels in labels]
        base_funs = [
            iaa.Fliplr(.5),  # 增加原图的概率。
            iaa.Fliplr(.5),  # 增加原图的概率。
            iaa.Fliplr(.5),  # 增加原图的概率。
            iaa.Fliplr(.5),  # 增加原图的概率。
            iaa.Fliplr(.5),  # 增加原图的概率。
            # iaa.Grayscale(alpha=(0, 1)),
            # iaa.ChangeColorspace('BGR'),
            # iaa.Add((-50, 50)),
            iaa.Dropout(0.02, per_channel=0.5),
            # iaa.GammaContrast(gamma=(0.5, 1.5), per_channel=True),

            ## 设置 加 雨 的噪声 类型
            iaa.Snowflakes(density=(0, 0.15), density_uniformity=0.08, flake_size=0.4,
                           flake_size_uniformity=0.5, speed=0.1),
            # 加 雪花 的噪声 类型
            iaa.Snowflakes(density=(0, 0.15), density_uniformity=0.5, flake_size=0.4, flake_size_uniformity=0.5,
                           speed=0.001),
            # iaa.Snowflakes(flake_size=(0.2, 0.7), speed=(0.001, 0.03)),
            # 运动模糊
            iaa.MotionBlur(k=(3, 20), angle=[-45, 45]),
            # 高斯模糊
            iaa.GaussianBlur((0, 7)),
            # 高斯噪音
            iaa.AdditiveGaussianNoise(scale=(0, 0.15 * 255)),
            # 拉普拉斯
            iaa.AdditiveLaplaceNoise(scale=(0, 0.15 * 255)),
            # 泊松
            iaa.AdditivePoissonNoise(lam=(0.0, 60.0)),
            # SaltAndPepper
            # iaa.Salt(0.05)
            # 椒盐
            iaa.SaltAndPepper((0.03, 0.1)),
            iaa.Affine(scale=(0.9, 1.1), translate_percent=(-.01, 0.01), rotate=(-3, 3))
        ]
        # choose one weather augmentation
        weather_aug = [
            # random.choice(
            #     [
            #         iaa.Snowflakes(flake_size=(0.4, 0.75), speed=(0.001, 0.03)),
            #         iaa.Fog(),
            #         iaa.Clouds(),
            #     ])
        ]
        base_funs += weather_aug
        random.shuffle(base_funs)
        base_funs = [random.choice(base_funs)]
        aug_funs = base_funs

        # do the augmentation
        seq_det = iaa.Sequential(aug_funs)
        seq_det = seq_det.to_deterministic()
        labels = [ia.BoundingBoxesOnImage(x, shape=images[i].shape) for i, x in enumerate(labels)]
        images = seq_det.augment_images(images)
        bbs_aug = seq_det.augment_bounding_boxes(labels)
        labels = [self._parse_bbs(x) for x in bbs_aug]
        return images, labels

    def augmentation(self, datas=None, for_one_image=None):
        images, labels = self._augmenting(datas, image_for_aug=for_one_image)

        if labels is not None:  # if labels is not none ,then check the label in labels,whether the label is none.
            for i, label in enumerate(labels):
                j = 1
                while not label:  # check every label, whether there is a label is empty.
                    print('no label at NO.', i)
                    label = labels[i - j]
                    labels[i] = labels[i - j]
                    images[i] = images[i - j]
                    j += 1
        return images, labels
