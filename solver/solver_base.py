from util.util_ConfigFactory_Classes import get_loss_class, get_score_class
from util.util_prepare_device import load_device
from cfg.cfg import prepare_cfg
from lgdet.dataloader.DataLoaderFactory import DataLoaderFactory
from lgdet.registry import MODELS, LOSSES, SCORES, build_from_cfg
from util.util_load_state_dict import load_state_dict


class SolverBase:
    def __init__(self, cfg, args, train):
        self.cfg, self.args = prepare_cfg(cfg, args)
        self.cfg.TRAIN.DEVICE, self.device_ids = load_device(self.cfg)
        self.Model = build_from_cfg(MODELS, str(self.cfg.TRAIN.MODEL).upper())(self.cfg)
        self.DataFun = DataLoaderFactory(self.cfg, self.args)
        self.Score = get_score_class(self.cfg.BELONGS)(self.cfg)
        if train:
            self.LossFun = get_loss_class(self.cfg.BELONGS, self.cfg.TRAIN.MODEL)(self.cfg)
        else:
            self.Model, ep, lr = load_state_dict(self.Model, self.args.checkpoint, self.cfg.TRAIN.DEVICE)
            self.Model = self.Model.to(self.cfg.TRAIN.DEVICE)
            self.Model.eval()
