import os
import torch
import numpy as np
import cv2


class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.one_test = cfg.TEST.ONE_TEST
        self.one_name = cfg.TEST.ONE_NAME
        self.train_batch_num = 100
        self.test_batch_num = 1

    def get_data_by_idx(self, idx_store, index_from, index_to):
        '''
        :param idx_store:
        :param index_from:
        :param index_to:
        :return: imags: torch.Float32, relative labels:[[cls, x1, y1, x2, y2],[...],...]
        '''
        data = (None, None)
        idx = idx_store[index_from: index_to]
        if self.one_test:
            idx = self.one_name
        if idx:
            imgs, labels = self._prepare_data(idx)
            imgs = imgs.permute([0, 3, 1, 2, ])
            imgs = imgs.to(self.cfg.TRAIN.DEVICE)
            labels = labels.to(self.cfg.TRAIN.DEVICE)
            data = (imgs, labels)  #
        return data

    def _prepare_data(self, idx):

        input_imgs = []
        target_imgs = []
        for id in idx:
            raw_lab = cv2.imread(os.path.join(self.cfg.PATH.LAB_PATH, id + '.png'))  # no norse image or HR image
            target = cv2.resize(raw_lab, (self.cfg.TRAIN.IMG_SIZE[0], self.cfg.TRAIN.IMG_SIZE[1]))

            if self.cfg.TRAIN.UPSCALE_FACTOR == 1:
                raw_img = cv2.imread(os.path.join(self.cfg.PATH.IMG_PATH, id + '.png'))  # norse image or LR image
            else:
                raw_img = target
            input = cv2.resize(raw_img, (self.cfg.TRAIN.IMG_SIZE[0] // self.cfg.TRAIN.UPSCALE_FACTOR,
                                         self.cfg.TRAIN.IMG_SIZE[1] // self.cfg.TRAIN.UPSCALE_FACTOR))
            input = torch.from_numpy(np.asarray((input - self.cfg.TRAIN.PIXCELS_NORM[0]) * 1.0 / self.cfg.TRAIN.PIXCELS_NORM[1])).type(torch.FloatTensor)
            target = torch.from_numpy(np.asarray((target - self.cfg.TRAIN.PIXCELS_NORM[0]) * 1.0 / self.cfg.TRAIN.PIXCELS_NORM[1])).type(torch.FloatTensor)
            input_imgs.append(input)
            target_imgs.append(target)

        input_imgs = torch.stack(input_imgs)
        target_imgs = torch.stack(target_imgs)

        return input_imgs, target_imgs

    # def _prepare_data_denoise(self, idx):


