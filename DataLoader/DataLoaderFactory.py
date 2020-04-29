import torch
import numpy as np
from DataLoader import Loader_ASR, Loader_OBD, Loader_SRDN
from torch.utils.data import DataLoader


class dataloader_factory:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.DataLoaderDict = {"OBD": Loader_OBD.Loader,
                               "ASR": Loader_ASR.Loader,
                               "SRDN": Loader_SRDN.Loader,
                               }

    def make_dataset(self, datasets=[]):
        DataSets = []
        for i, dataset in enumerate(datasets):
            data = self.DataLoaderDict[self.cfg.BELONGS](self.cfg)
            shuffle = False if i == 1 or len(datasets) == 1 else True
            data._load_dataset(dataset, is_training=shuffle)
            Dataset = DataLoader(dataset=data,
                                 batch_size=self.cfg.TRAIN.BATCH_SIZE,
                                 num_workers=self.args.number_works,
                                 collate_fn=data.collate_fun,
                                 shuffle=shuffle)
            DataSets.append(Dataset)

        return DataSets

    def to_devce(self, data):
        datas = []
        for i, da in enumerate(data):
            try:
                datas.append(da.to(self.cfg.TRAIN.DEVICE, non_blocking=True))
            except:
                datas.append(da)
        return datas
