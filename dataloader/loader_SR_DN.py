import os
import torch
import numpy as np
import cv2


# import multiprocessing as mp


class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.one_test = cfg.TEST.ONE_TEST
        self.one_name = cfg.TEST.ONE_NAME
        self.train_batch_num = 100
        self.test_batch_num = 1
        self.resize_input2output = True

    def get_data_by_idx(self, idx_store, index_from, index_to, is_training):
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
            imgs, labels = self._prepare_data(idx)
            imgs = imgs.permute([0, 3, 1, 2])
            imgs = imgs.to(self.cfg.TRAIN.DEVICE)
            labels = labels.to(self.cfg.TRAIN.DEVICE)
            data = (imgs, labels)  #
        return data

    def _prepare_data(self, idx):
        input_imgs = []
        target_imgs = []
        for id in idx:
            raw_lab = cv2.imread(id[2])  # no norse image or HR image
            target = cv2.resize(raw_lab, (self.cfg.TRAIN.IMG_SIZE[0], self.cfg.TRAIN.IMG_SIZE[1]))

            if self.cfg.TRAIN.UPSCALE_FACTOR == 1:
                raw_img = cv2.imread(id[1])  # norse image or LR image
            else:
                raw_img = target
            input = cv2.resize(raw_img, (self.cfg.TRAIN.IMG_SIZE[0] // self.cfg.TRAIN.UPSCALE_FACTOR,
                                         self.cfg.TRAIN.IMG_SIZE[1] // self.cfg.TRAIN.UPSCALE_FACTOR))
            if self.resize_input2output:
                input = cv2.resize(raw_img, (self.cfg.TRAIN.IMG_SIZE[0], self.cfg.TRAIN.IMG_SIZE[1]))

            input = torch.from_numpy(np.asarray((input - self.cfg.TRAIN.PIXCELS_NORM[0]) * 1.0 / self.cfg.TRAIN.PIXCELS_NORM[1])).type(torch.FloatTensor)
            target = torch.from_numpy(np.asarray((target - self.cfg.TRAIN.PIXCELS_NORM[0]) * 1.0 / self.cfg.TRAIN.PIXCELS_NORM[1])).type(torch.FloatTensor)
            input_imgs.append(input)
            target_imgs.append(target)

        input_imgs = torch.stack(input_imgs)
        target_imgs = torch.stack(target_imgs)

        return input_imgs, target_imgs

    # def _prepare_data_denoise(self, idx):
