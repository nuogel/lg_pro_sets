import torch
from util.util_data_aug import Dataaug
import numpy as np
from util.util_is_use_cuda import _is_use_cuda


class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataaug = Dataaug(cfg)
        self.one_test = cfg.TEST.ONE_TEST
        self.one_name = cfg.TEST.ONE_NAME
        self.train_batch_num = 100
        self.test_batch_num = 1
        self.show_augimg = cfg.TRAIN.SHOW_AUG

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
            do_aug = False
        else:
            do_aug = self.cfg.TRAIN.DO_AUG
        if idx:
            imgs, labels = self.dataaug.augmentation(idx, do_aug=do_aug, resize=self.cfg.TRAIN.RESIZE,
                                                     relative=self.cfg.TRAIN.RELATIVE_LABELS, show_img=self.show_augimg)
            imgs = torch.Tensor(np.array(imgs))

            imgs = imgs.permute([0, 3, 1, 2, ])
            imgs = imgs * 0.00392156885937 + 0.0

            if _is_use_cuda(self.cfg.TRAIN.GPU_NUM):
                imgs = imgs.cuda(self.cfg.TRAIN.GPU_NUM)
            data = (imgs, labels)  #
        return data
