import torch
import numpy as np
from DataLoader import Loader_ASR, Loader_OBD, Loader_SRDN, Loader_VID
from torch.utils.data import DataLoader
from util.util_ConfigFactory_Classes import get_loader_class


class dataloader_factory:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args


    def make_dataset(self, train_dataset=None, test_dataset=None):
        if self.cfg.BELONGS == 'VID':
            shuffle = False
        else:
            shuffle = True
        TrainDataset, TestDataset = None, None
        if train_dataset is not None:
            train_data = get_loader_class(self.cfg.BELONGS)(self.cfg)

            train_data._load_dataset(train_dataset, is_training=True)
            TrainDataset = DataLoader(dataset=train_data,
                                      batch_size=self.cfg.TRAIN.BATCH_SIZE,
                                      num_workers=self.args.number_works,
                                      collate_fn=train_data.collate_fun,
                                      shuffle=shuffle)

        if test_dataset is not None:
            test_data = get_loader_class(self.cfg.BELONGS)(self.cfg)
            test_data._load_dataset(test_dataset, is_training=False)
            TestDataset = DataLoader(dataset=test_data,
                                     batch_size=self.cfg.TRAIN.BATCH_SIZE,
                                     num_workers=self.args.number_works,
                                     collate_fn=test_data.collate_fun,
                                     shuffle=False)

        return TrainDataset, TestDataset

    def to_devce(self, data):
        datas = []
        for i, da in enumerate(data):
            try:
                datas.append(da.to(self.cfg.TRAIN.DEVICE, non_blocking=True))
            except:
                datas.append(da)
        return datas
