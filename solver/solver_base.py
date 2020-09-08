import os
import torch
from torch.optim import lr_scheduler
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
            self.Model, ep, lr, _ = load_state_dict(self.Model, self.args.checkpoint, self.cfg.TRAIN.DEVICE)
            self.Model = self.Model.to(self.cfg.TRAIN.DEVICE)
            self.Model.eval()

    def _get_optimizer(self, learning_rate):
        optimizer = self.cfg.TRAIN.OPTIMIZER
        model_parameters = filter(lambda p: p.requires_grad, self.Model.parameters())
        if optimizer == 'adam' or optimizer == 'Adam':
            optimizer = torch.optim.Adam(model_parameters, lr=learning_rate, betas=(self.cfg.TRAIN.BETAS_ADAM, 0.999),
                                         weight_decay=float(self.cfg.TRAIN.WEIGHT_DECAY))
        elif optimizer == 'sgd' or optimizer == 'SGD':
            optimizer = torch.optim.SGD(model_parameters, lr=learning_rate, momentum=0.9,
                                        weight_decay=float(self.cfg.TRAIN.WEIGHT_DECAY))
        else:
            self.cfg.logger.error('NO such a optimizer: ' + str(optimizer))
        # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg.LR_EXPONENTIAL_DECAY_RATE)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.cfg.TRAIN.STEP_LR, gamma=0.1)
        return optimizer, scheduler

    def _save_checkpoint(self, model, epoch, lr_now, global_step):
        checkpoint_path_0 = os.path.join(self.cfg.PATH.TMP_PATH, 'checkpoint', '{}.pkl'.format(epoch))
        checkpoint_path_1 = os.path.join(self.cfg.PATH.TMP_PATH, 'checkpoint', 'now.pkl'.format(epoch))
        checkpoint_path_2 = os.path.join(self.cfg.PATH.TMP_PATH + 'checkpoint/tbx_log_' + self.cfg.TRAIN.MODEL, 'now.pkl')
        if self.cfg.TEST.ONE_TEST:
            path_list = [checkpoint_path_1]
        else:
            path_list = [checkpoint_path_0, checkpoint_path_1, checkpoint_path_2]
        for path_i in path_list:
            os.makedirs(os.path.dirname(path_i), exist_ok=True)
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch, 'lr': lr_now,
                          'global_step': global_step}
            torch.save(saved_dict, path_i)
            self.cfg.logger.debug('Epoch: %s, checkpoint is saved to %s', epoch, path_i)
