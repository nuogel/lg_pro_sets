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

    def make_dataset(self, traindataset, testdataset):
        traindata = self.DataLoaderDict[self.cfg.BELONGS](self.cfg, traindataset)
        train_Dataset = DataLoader(dataset=traindata,
                                   batch_size=self.cfg.TRAIN.BATCH_SIZE,
                                   num_workers=self.args.number_works,
                                   # collate_fn=self.collate_fun,
                                   shuffle=True)

        testdata = self.DataLoaderDict[self.cfg.BELONGS](self.cfg, testdataset)
        test_Dataset = DataLoader(dataset=testdata,
                                  batch_size=self.cfg.TRAIN.BATCH_SIZE,
                                  num_workers=self.args.number_works,
                                  shuffle=False)
        return train_Dataset, test_Dataset

    def iter_loader(self, dataset):
        dataloader = iter(dataset)
        return dataloader

    def load_next(self, dataset):
        data = next(dataset)
        return data

    def collate_fun(self, batch):
        imgs, labels = zip(*batch)
        return torch.from_numpy(np.asarray(imgs)), list(labels)
