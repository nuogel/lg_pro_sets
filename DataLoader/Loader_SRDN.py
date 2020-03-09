import os
import torch
import numpy as np
import cv2
from util.util_CCPD import _crop_licience_plante
from util.util_data_aug import Dataaug


# import multiprocessing as mp


class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.one_test = cfg.TEST.ONE_TEST
        self.one_name = cfg.TEST.ONE_NAME
        self.train_batch_num = 100
        self.test_batch_num = 1
        self.resize_input2output = False
        if self.cfg.TRAIN.INPUT_AUG:
            self.Data_aug = Dataaug(self.cfg)
        if self.cfg.TRAIN.MODEL in ['cbdnet', 'srcnn', 'vdsr']:
            self.resize_input2output = True

    def get_data_by_idx(self, idx_store, index_from, index_to, is_training=True):
        '''
        :param idx_store:
        :param index_from:
        :param index_to:
        :return: imags: torch.Float32, relative labels:[[cls, x1, y1, x2, y2],[...],...]
        '''
        data = (None, None)
        if self.one_test:
            if self.one_name:
                idx = self.one_name
            else:
                idx = idx_store[1:2]
        else:
            idx = idx_store[index_from: index_to]
        if not idx:
            print('error, no IDX in loader_img.py')
            exit()
        if idx:
            # processes = mp.Pool(4)
            # imgs, labels = processes.apply_async(self._prepare_data, (idx))
            imgs, labels = self._prepare_data(idx, is_training)
            imgs = imgs.permute([0, 3, 1, 2])
            imgs = imgs.to(self.cfg.TRAIN.DEVICE)
            labels = labels.to(self.cfg.TRAIN.DEVICE)
            data = (imgs, labels)  #
        return data

    def _prepare_data(self, idx, is_training=False):
        input_imgs = []
        target_imgs = []
        for id in idx:
            target = cv2.imread(id[2])  # no norse image or HR image
            if self.cfg.TRAIN.TARGET_PREDEEL:
                target = self._target_predeal(img=target, filename=id[0])

            input = target if id[1] in ['None', '', ' ', 'none'] else cv2.imread(id[1])
            if self.cfg.TRAIN.INPUT_PREDEEL and is_training:
                input = self._input_predeal(img=input, filename=id[0])

            if self.cfg.TRAIN.SHOW_INPUT:
                cv2.imshow('img', input)
                cv2.waitKey(self.cfg.TRAIN.SHOW_INPUT)
            input = torch.from_numpy(np.asarray((input - self.cfg.TRAIN.PIXCELS_NORM[0]) * 1.0 / self.cfg.TRAIN.PIXCELS_NORM[1])).type(torch.FloatTensor)
            target = torch.from_numpy(np.asarray((target - self.cfg.TRAIN.PIXCELS_NORM[0]) * 1.0 / self.cfg.TRAIN.PIXCELS_NORM[1])).type(torch.FloatTensor)
            input_imgs.append(input)
            target_imgs.append(target)

        input_imgs = torch.stack(input_imgs)
        target_imgs = torch.stack(target_imgs)

        return input_imgs, target_imgs

    def _input_predeal(self, **kwargs):
        img = kwargs['img']
        filename = kwargs['filename']
        img = cv2.resize(img, (self.cfg.TRAIN.IMG_SIZE[0] // self.cfg.TRAIN.UPSCALE_FACTOR,
                               self.cfg.TRAIN.IMG_SIZE[1] // self.cfg.TRAIN.UPSCALE_FACTOR))
        if self.resize_input2output:
            img = cv2.resize(img, (self.cfg.TRAIN.IMG_SIZE[0], self.cfg.TRAIN.IMG_SIZE[1]))
        # add the augmentation ...
        if self.cfg.TRAIN.INPUT_AUG:
            img, _ = self.Data_aug.augmentation(for_one_image=[img])
            img = img[0]

        return img

    def _target_predeal(self, **kwargs):
        img = kwargs['img']
        filename = kwargs['filename']
        # img = _crop_licience_plante(img, filename)

        try:
            img = cv2.resize(img, (self.cfg.TRAIN.IMG_SIZE[0], self.cfg.TRAIN.IMG_SIZE[1]))
        except:
            print(filename)
        else:
            pass
        return img
