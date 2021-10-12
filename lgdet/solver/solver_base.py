import os
import torch
from torch.optim import lr_scheduler
from lgdet.factory_classes import get_loss_class, get_score_class
from lgdet.util.util_prepare_device import load_device
from lgdet.config.cfg import prepare_cfg
from lgdet.dataloader.DataLoaderFactory import DataLoaderFactory
from lgdet.registry import MODELS, build_from_cfg
from lgdet.util.util_weights_init import weights_init
from lgdet.util.util_get_dataset_from_file import _read_train_test_dataset
from lgdet.util.util_load_save_checkpoint import _load_checkpoint, _save_checkpoint, _load_pretrained
from lgdet.metrics.ema import ModelEMA
import math
from torch.nn.parallel import DistributedDataParallel as DDP



class BaseSolver(object):
    def __init__(self, cfg, args, train):
        self.is_training = train
        self._get_configs(cfg, args)
        self._get_model()
        self._get_dataloader(train, cfg)
        self.epoch = 0
        self.metrics_ave = {}
        self.epoch_losses = 0

    def _get_configs(self, cfg, args):
        self.cfg, self.args = prepare_cfg(cfg, args, self.is_training)
        self.cfg.TRAIN.DEVICE, self.device_ids = load_device(self.cfg)

    def _get_model(self):
        self.model = build_from_cfg(MODELS, str(self.cfg.TRAIN.MODEL).upper())(self.cfg)
        self.optimizer_dict = None
        self.epoch_last = 0
        self.global_step = 0
        # init model:
        if self.args.checkpoint not in [0, '0', 'None', 'no', 'none', "''"]:
            if self.args.checkpoint in [1, '1']: self.args.checkpoint = os.path.join(self.cfg.PATH.TMP_PATH, 'checkpoints', self.cfg.TRAIN.MODEL, 'now.pkl')
            print('loading checkpoint:', self.args.checkpoint)
            self.model, self.epoch_last, self.optimizer_dict, self.optimizer_type, self.global_step = _load_checkpoint(self.model,
                                                                                                                       self.args.checkpoint,
                                                                                                                       self.cfg.TRAIN.DEVICE)
            if self.is_training:
                self.cfg.writer.tbX_reStart(self.epoch_last)
        elif self.args.pre_trained not in [0, '0', 'None', 'no', 'none', "''"]:
            if self.args.pre_trained in [1, '1']:
                self.args.pre_trained = os.path.join(self.cfg.PATH.TMP_PATH + 'checkpoints/' + self.cfg.TRAIN.MODEL, 'now.pkl')
                print('loading pre_trained:', self.args.pre_trained)
                self.model = _load_pretrained(self.model, self.args.pre_trained, self.cfg.TRAIN.DEVICE)
            elif self.args.pre_trained in [2, '2']:
                print('loading pre_trained from model itself')
            else:
                print('loading pre_trained:', self.args.pre_trained)
                self.model = _load_pretrained(self.model, self.args.pre_trained, self.cfg.TRAIN.DEVICE)
            if self.is_training:
                self.cfg.writer.clean_history_and_init_log()

        else:

            weights_init(self.model, self.cfg.manual_seed)
            if self.is_training:
                self.cfg.writer.clean_history_and_init_log()

        if len(self.device_ids) > 1:
            print('using device id:', self.device_ids)
            self.model = DDP(self.model, device_ids=self.device_ids)

        if self.cfg.TRAIN.EMA:
            self.ema = ModelEMA(self.model, device=self.cfg.TRAIN.DEVICE)

        if self.is_training:
            self.model.train()
        else:
            if self.cfg.TEST.ONE_TEST:
                self.model.train()
            else:
                self.model.eval()

        self.model = self.model.to(self.cfg.TRAIN.DEVICE)

    def _get_score(self):
        self.score = get_score_class(self.cfg.BELONGS)(self.cfg)

    def _get_lossfun(self):
        self.lossfun = get_loss_class(self.cfg.BELONGS, self.cfg.TRAIN.MODEL)(self.cfg)

    def _get_optimizer(self):
        opt_type = self.cfg.TRAIN.OPTIMIZER.lower()
        learning_rate = self.args.lr
        if opt_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=float(self.cfg.TRAIN.WEIGHT_DECAY))
        elif opt_type == 'adamw':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-2)  # weight_decay=1e-2
        elif opt_type == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.937, weight_decay=float(self.cfg.TRAIN.WEIGHT_DECAY))
        else:
            self.cfg.logger.error('NO such a optimizer: ' + str(opt_type))
        print('using: ', opt_type)
        if self.optimizer_dict and opt_type == self.optimizer_type:
            self.optimizer.state_dict = self.optimizer_dict
        self.optimizer.param_groups[0]['initial_lr'] = learning_rate
        if self.args.lr_continue:
            self.optimizer.param_groups[0]['lr'] = self.args.lr_continue
            self.optimizer.param_groups[0]['initial_lr'] = self.args.lr_continue
        self.learning_rate = self.optimizer.param_groups[0]['lr']

        if self.cfg.TRAIN.LR_SCHEDULE == 'cos':
            print('using cos LambdaLR lr_scheduler')
            finial_lr = 1e-5
            alpha = finial_lr / self.optimizer.param_groups[0]['initial_lr']
            lf = lambda x: (0.5 * (1 + math.cos(x * math.pi / self.cfg.TRAIN.EPOCH_SIZE))) * (1 - alpha) + alpha
            self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf, last_epoch=self.epoch_last - 1)
        elif self.cfg.TRAIN.LR_SCHEDULE == 'step':
            print('using StepLR lr_scheduler ', self.cfg.TRAIN.STEP_LR)
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.TRAIN.STEP_LR, gamma=0.1)
        elif self.cfg.TRAIN.LR_SCHEDULE == 'reduce':
            factor, patience, min_lr = 0.8, 3, 1e-6
            print('using ReduceLROnPlateau lr_scheduler: factor%.1f, patience:%d' % (factor, patience))
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=factor, patience=patience, min_lr=min_lr, verbose=True)
        # Plot lr schedule
        plot_lr = 0
        if plot_lr:
            y = []
            for _ in range(0, self.cfg.TRAIN.EPOCH_SIZE):
                if self.cfg.TRAIN.LR_SCHEDULE == 'reduce':
                    self.scheduler.step(1.)
                else:
                    self.scheduler.step()
                y.append(self.optimizer.param_groups[0]['lr'])
            print(y)
        self.optimizer.zero_grad()

    def _set_warmup_lr(self):
        self.optimizer.param_groups[0]['lr'] = self.learning_rate / self.cfg.TRAIN.WARM_UP_STEP * (self.global_step + 1)

    def _get_dataloader(self, is_training, cfg):
        """
        Get the self.model, learning_rate, epoch_last, train_set, test_set.
        :return: learning_rate, epoch_last, train_set, test_set.
        """
        self.DataFun = DataLoaderFactory(self.cfg, self.args)
        #  load the last data set
        if is_training != None:
            train_set, test_set = _read_train_test_dataset(self.cfg)

            print('train set:', train_set[0], '\ntest  set:', test_set[0])
            txt = 'train set:{};test  set:{}'.format(len(train_set), len(test_set))
            print(txt)
            self.cfg.logger.info(txt)
            self.trainDataloader, self.testDataloader = self.DataFun.make_dataset(train_set, test_set, cfg)

    def _calculate_loss(self, predict, dataset, **kwargs):
        metrics_info = ''
        epoch = kwargs['epoch']
        step = kwargs['step']
        len_batch = kwargs['len_batch']

        losses = self.lossfun.Loss_Call(predict, dataset, kwargs)
        total_loss = losses['total_loss']
        loss_metrics = losses['metrics']
        for k, v in loss_metrics.items():
            if step == 0:
                self.metrics_ave[k] = 0
                self.epoch_losses = 0
            self.metrics_ave[k] += v

            metrics_info += k + ':' + '%.3f' % (self.metrics_ave[k] / (step + 1)) + '|'
        self.epoch_losses += total_loss.item()

        info_base = (self.cfg.TRAIN.MODEL,
                     str(epoch) + '/' + str(self.cfg.TRAIN.EPOCH_SIZE),
                     self.global_step, '%0.6f' % self.optimizer.param_groups[0]['lr'],
                     '%0.3f' % total_loss.item(),
                     '%0.3f' % (self.epoch_losses / (step + 1)))

        train_info = ('%16s|%7s|%7s|%9s|' + '%6s|' * 2) % info_base + ' '
        train_info += metrics_info
        if self.global_step % 1000 == 0:
            self.cfg.logger.info(train_info)

        # add tensorboard writer.
        if self.global_step % len_batch == 0:
            w_dict = {'epoch': epoch, 'lr': self.optimizer.param_groups[0]['lr'], 'epoch_loss': self.epoch_losses / len(self.trainDataloader)}
            for k, v in self.metrics_ave.items():
                w_dict['metrics/' + k] = v / (step + 1)
            self.cfg.writer.tbX_write(w_dict=w_dict)

        if torch.isnan(total_loss) or total_loss.item() == float("inf") or total_loss.item() == -float("inf"):
            self.cfg.logger.error("received an nan/inf loss:", dataset[-1])

        return total_loss, train_info

    def _save_checkpoint(self):
        _save_checkpoint(self)
