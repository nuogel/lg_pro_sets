import os
import torch
import numpy as np
from util.util_is_use_cuda import _is_use_cuda
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
        if idx:
            if self.one_test:
                idx = self.one_name
            imgs, labels = self._prepare_data(idx)
            if _is_use_cuda(self.cfg.TRAIN.GPU_NUM):
                imgs = imgs.cuda(self.cfg.TRAIN.GPU_NUM)
                labels = labels.cuda(self.cfg.TRAIN.GPU_NUM)
            data = (imgs, labels)  #
        return data

    def _prepare_data(self, idx):

        input_imgs = []
        target_imgs = []
        for id in idx:
            img_path = os.path.join(self.cfg.PATH.IMG_PATH, id + '.png')
            raw_img = cv2.imread(img_path, ) / 1.0
            target = cv2.resize(raw_img, (self.cfg.TRAIN.IMG_SIZE[0], self.cfg.TRAIN.IMG_SIZE[1]))
            input = cv2.resize(target, (self.cfg.TRAIN.IMG_SIZE[0] // self.cfg.TRAIN.UPSCALE_FACTOR,
                                        self.cfg.TRAIN.IMG_SIZE[1] // self.cfg.TRAIN.UPSCALE_FACTOR))
            input = torch.from_numpy(np.asarray(input)).type(torch.FloatTensor)
            target = torch.from_numpy(np.asarray(target)).type(torch.FloatTensor)
            input_imgs.append(input)
            target_imgs.append(target)

        input_imgs = torch.stack(input_imgs)
        target_imgs = torch.stack(target_imgs)

        return input_imgs, target_imgs
