import os
import torch
from torch.optim import lr_scheduler
from util.util_ConfigFactory_Classes import get_loss_class, get_score_class
from util.util_prepare_device import load_device
from cfg.cfg import prepare_cfg
from lgdet.dataloader.DataLoaderFactory import DataLoaderFactory
from lgdet.registry import MODELS, LOSSES, SCORES, build_from_cfg
from util.util_weights_init import weights_init
from util.util_get_dataset_from_file import _read_train_test_dataset
from collections import OrderedDict


class BaseSolver(object):
    def __init__(self, cfg, args, train):
        self.is_training = train
        self._get_configs(cfg, args)
        self._get_model()
        self._get_dataloader()

    def _get_configs(self, cfg, args):
        self.cfg, self.args = prepare_cfg(cfg, args)
        self.cfg.TRAIN.DEVICE, self.device_ids = load_device(self.cfg)

    def _get_model(self):
        self.model = build_from_cfg(MODELS, str(self.cfg.TRAIN.MODEL).upper())(self.cfg)
        # init model:
        if self.args.checkpoint not in [0, '0', 'None', 'no', 'none', "''"]:
            self.model, self.epoch_last, self.optimizer, self.global_step = self._load_checkpoint(self.model, self.args.checkpoint, self.cfg.TRAIN.DEVICE)
            self.cfg.writer.tbX_reStart(self.epoch_last)

        else:
            self.model = weights_init(self.model, self.cfg)
            self.optimizer = None
            self.epoch_last = 0
            self.global_step = 0
            self.cfg.writer.clean_history_and_init_log()

        self.model = self.model.to(self.cfg.TRAIN.DEVICE)

        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)

        if self.is_training:
            self.model.train()
            self.model.zero_grad()
        else:
            if not self.cfg.TEST.ONE_TEST:
                self.model.eval()

    def _get_score(self):
        self.score = get_score_class(self.cfg.BELONGS)(self.cfg)

    def _get_lossfun(self):
        self.lossfun = get_loss_class(self.cfg.BELONGS, self.cfg.TRAIN.MODEL)(self.cfg)

    def _get_optimizer(self):
        if not self.is_training:
            pass
        if self.optimizer is None:
            opt_type = self.cfg.TRAIN.OPTIMIZER
            learning_rate = self.args.lr
            model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            if opt_type == 'adam' or opt_type == 'Adam':
                self.optimizer = torch.optim.Adam(model_parameters,
                                                  lr=learning_rate,
                                                  betas=(self.cfg.TRAIN.BETAS_ADAM, 0.999),
                                                  weight_decay=float(self.cfg.TRAIN.WEIGHT_DECAY))
            elif opt_type == 'sgd' or opt_type == 'SGD':
                self.optimizer = torch.optim.SGD(model_parameters,
                                                 lr=learning_rate,
                                                 momentum=0.9,
                                                 weight_decay=0.0001)
            else:
                self.cfg.logger.error('NO such a optimizer: ' + str(opt_type))
        else:
            if self.args.lr_continue:
                self.optimizer.param_groups[0]['lr'] = self.args.lr_continue

        self.learning_rate = self.optimizer.param_groups[0]['lr']
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.TRAIN.STEP_LR, gamma=0.1)
        self.optimizer.zero_grad()

    def _set_warmup_lr(self, optimizer):

        optimizer.param_groups[0]['lr'] = self.learning_rate / self.cfg.TRAIN.WARM_UP_STEP * (self.global_step + 1)
        return optimizer

    def _get_dataloader(self):
        """
        Get the self.model, learning_rate, epoch_last, train_set, test_set.
        :return: learning_rate, epoch_last, train_set, test_set.
        """
        self.DataFun = DataLoaderFactory(self.cfg, self.args)
        #  load the last data set
        train_set, test_set = _read_train_test_dataset(self.cfg)
        print('train set:', train_set[0], '\n', 'test set:', test_set[0])
        self.cfg.logger.info('>' * 30 + 'train set:{}; test set:{}'.format(len(train_set), len(test_set)))
        self.trainDataloader, self.testDataloader = self.DataFun.make_dataset(train_set, test_set)

    def _calculate_loss(self, predict, dataset, **kwargs):
        total_loss = 0.
        loss_head_info = ''
        losses = self.lossfun.Loss_Call(predict, dataset, kwargs=kwargs)
        w_dict = {}
        for k, v in losses.items():
            total_loss += v

        # add tensorboard writer.
        if self.global_step % 200 == 0:
            for k, v in losses.items():
                loss_head_info += ' {}: {:6.4f}'.format(k, v.item())
                w_dict['item_losses/' + k] = v
            w_dict['epoch'] = self.global_step
            self.cfg.writer.tbX_write(w_dict=w_dict)

        self.cfg.logger.debug(loss_head_info)
        if torch.isnan(total_loss) or total_loss.item() == float("inf") or total_loss.item() == -float("inf"):
            self.cfg.logger.error("received an nan/inf loss:", dataset[-1])

            exit()
        return total_loss

    def _save_checkpoint(self, model, epoch, optimizer, global_step):
        checkpoint_path_0 = os.path.join(self.cfg.PATH.TMP_PATH, 'checkpoints/common_checkpoints', '{}.pkl'.format(epoch))
        checkpoint_path_1 = os.path.join(self.cfg.PATH.TMP_PATH, 'checkpoints/common_checkpoints', 'now.pkl'.format(epoch))
        checkpoint_path_2 = os.path.join(self.cfg.PATH.TMP_PATH + 'checkpoints/' + self.cfg.TRAIN.MODEL, 'now.pkl')
        if self.cfg.TEST.ONE_TEST:
            path_list = [checkpoint_path_1]
        else:
            path_list = [checkpoint_path_0, checkpoint_path_1, checkpoint_path_2]
        for path_i in path_list:
            os.makedirs(os.path.dirname(path_i), exist_ok=True)
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch, 'optimizer': optimizer,
                          'global_step': global_step}
            torch.save(saved_dict, path_i)
            self.cfg.logger.debug('Epoch: %s, checkpoint is saved to %s', epoch, path_i)

    def _load_checkpoint(self, model, checkpoint, device):
        new_dic = OrderedDict()
        checkpoint = torch.load(checkpoint, map_location=device)
        state_dict = checkpoint['state_dict']
        last_epoch = checkpoint['epoch']
        optimizer = checkpoint['optimizer']
        global_step = checkpoint['global_step']
        for k, v in state_dict.items():
            if 'module.' == k[:8]:
                k = k.replace('module.', '')
            new_dic[k] = v
        model.load_state_dict(new_dic)
        return model, last_epoch, optimizer, global_step
