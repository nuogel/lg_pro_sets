"""Train the model.
writer:luogeng 2018.09
Train.py is used for training the net to mark things with a box outside of
them, and with a label of what it is, and with it's score at the left top of
the box.

At the end, we will get the weight file of the net.

"""
import os
import logging

import numpy as np
import torch
import torch.nn
from torch.optim import lr_scheduler
from apex import amp
from util.util_show_save_parmeters import TrainParame
from NetWorks.NetworksConfigFactory import get_loss_class, get_model_class, get_score_class
from util.util_time_stamp import Time
from util.util_weights_init import weights_init
from util.util_get_train_test_dataset import _get_train_test_dataset, _read_train_test_dataset
from util.util_prepare_device import load_device
from util.util_load_state_dict import load_state_dict
from util.util_prepare_cfg import prepare_cfg
from util.util_logger import load_logger
from DataLoader.DataLoaderFactory import dataloader_factory
from util.util_quantization.util_quantization import util_quantize_model
# self.LOGGER = logging.getLogger(__name__)

class Solver:
    def __init__(self, cfg, args):
        self.cfg = prepare_cfg(cfg, args)
        self.args = args
        self.one_test = self.cfg.TEST.ONE_TEST
        self.cfg.TRAIN.DEVICE, self.device_ids = load_device(self.cfg)
        self.dataloader_factory = dataloader_factory(self.cfg, args)
        self.save_parameter = TrainParame(self.cfg)
        self.Model = get_model_class(self.cfg.BELONGS, self.cfg.TRAIN.MODEL)(self.cfg)
        self.LossFun = get_loss_class(self.cfg.BELONGS, self.cfg.TRAIN.MODEL)(self.cfg)
        self.Score = get_score_class(self.cfg.BELONGS)(self.cfg)
        self.test_batch_num = self.cfg.TEST.ONE_TEST_TEST_STEP
        self.LOGGER = load_logger(args)
        print('torch version: ', torch.__version__)
        print('torch.version.cuda: ', torch.version.cuda)

    def train(self):
        """Train the network.

        Args:
            self.cfg.TMP_PATH (str): Path to store files generated during training
            epochs (int): Total epochs to train
            checkpoint (str): Path to checkpoint

        """
        self.LOGGER.info('>' * 30 + '{} Start Training'.format(self.cfg.TRAIN.MODEL))
        os.makedirs(self.cfg.PATH.TMP_PATH, exist_ok=True)
        # Prepare network, data set idx
        learning_rate, epoch_last = self._prepare_parameters()
        # Prepare optimizer
        optimizer, scheduler = self._get_optimizer(learning_rate, optimizer=self.cfg.TRAIN.OPTIMIZER)
        if self.args.tensor_core in ['O1', 'O2', 'O3']:
            self.Model, optimizer = amp.initialize(self.Model, optimizer, opt_level=self.args.tensor_core, loss_scale="dynamic")
        for epoch in range(epoch_last, self.cfg.TRAIN.EPOCH_SIZE):
            if not self.cfg.TEST.TEST_ONLY and not self.args.test_only:
                self._train_an_epoch(epoch, optimizer, scheduler)
                self._save_checkpoint(epoch)
            if epoch > 2000 or self.one_test:
                self._test_an_epoch(epoch)

    def _prepare_parameters(self):
        """
        Get the self.Model, learning_rate, epoch_last, train_set, test_set.
        :return: learning_rate, epoch_last, train_set, test_set.
        """
        torch.backends.cudnn.benchmark = True
        # load the last train parameters
        if self.args.checkpoint in [0, '0', 'None', 'no', 'none', "''"]:
            weights_init(self.Model, self.cfg)
            # start a new train, delete the exist parameters
            self.save_parameter.clean_history_and_init_log()
            epoch = 0
            learning_rate = self.args.lr  # if self.args.lr else self.cfg.TRAIN.LR_CONTINUE
            # generate a new data set
            if self.cfg.TRAIN.TRAIN_DATA_FROM_FILE:
                train_set, test_set = _read_train_test_dataset(self.cfg)
            else:
                train_set, test_set = _get_train_test_dataset(self.cfg)
        else:
            self.Model = load_state_dict(self.Model, self.args.checkpoint, self.cfg.TRAIN.DEVICE)
            epoch_last, learning_rate_last = self.save_parameter.tbX_read()
            epoch = self.args.epoch_continue if self.args.epoch_continue else epoch_last + 1
            learning_rate = self.args.lr_continue if self.args.lr_continue else learning_rate_last
            self.LOGGER.info('>' * 30 + 'Loading Last Checkpoint: %s, Last Learning Rate:%s, Last Epoch:%s',
                             self.args.checkpoint, learning_rate, epoch)
            #  load the last data set
            train_set, test_set = _read_train_test_dataset(self.cfg)

        print('TRAIN SET:', train_set[:4], '\n', 'TEST SET:', test_set[:4])
        self.LOGGER.info('>' * 30 + 'The Train Set is :{}, and The Test Set is :{}'.format(len(train_set), len(test_set)))
        self.trainDataloader, self.testDataloader = self.dataloader_factory.make_dataset([train_set, test_set])

        self.Model = self.Model.to(self.cfg.TRAIN.DEVICE)
        if len(self.device_ids) > 1:
            self.Model = torch.nn.DataParallel(self.Model, device_ids=self.device_ids)
        # _print_model_parm_nums(self.Model.to(self.cfg.TRAIN.DEVICE), self.cfg.TRAIN.IMG_SIZE[0], self.cfg.TRAIN.IMG_SIZE[1])
        quantization = 0
        if quantization:
            self.Model = util_quantize_model(self.Model)

        return learning_rate, epoch

    def _get_optimizer(self, learning_rate, optimizer='adam'):
        if optimizer == 'adam' or optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.Model.parameters(), lr=learning_rate, betas=(self.cfg.TRAIN.BETAS_ADAM, 0.999), weight_decay=float(self.cfg.TRAIN.WEIGHT_DECAY))
        elif optimizer == 'sgd' or optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.Model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        else:
            self.LOGGER.error('NO such a optimizer: ' + str(optimizer))
        # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg.LR_EXPONENTIAL_DECAY_RATE)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.cfg.TRAIN.STEP_LR, gamma=0.1)
        return optimizer, scheduler

    def _calculate_loss(self, predict, dataset, losstype=None):
        total_loss = 0.
        losses = self.LossFun.Loss_Call(predict, dataset, losstype=losstype)
        if self.cfg.BELONGS == 'OBD':
            loss_names = ['[obj_loss]', '[noobj_loss]', '[cls_loss]', '[loc_loss]']  # obj_loss, noobj_loss, cls_loss, loc_loss
            for i in range(len(losses)):
                total_loss += losses[i]
            loss_head_info = ''
            for loss_name, head_loss in zip(loss_names[:len(losses)], losses):
                loss_head_info += ' {}: {:6.4f}'.format(loss_name, head_loss.item())
            self.LOGGER.debug('Loss per head: %s', loss_head_info)
        else:
            total_loss = losses[0]
        if torch.isnan(total_loss) or total_loss.item() == float("inf") or total_loss.item() == -float("inf"):
            self.LOGGER.error("received an nan/inf loss")
            exit()
        return total_loss

    def _save_checkpoint(self, epoch):
        checkpoint_path_0 = os.path.join(self.cfg.PATH.TMP_PATH, 'checkpoint', '{}.pkl'.format(epoch))
        checkpoint_path_1 = os.path.join(self.cfg.PATH.TMP_PATH, 'checkpoint', 'now.pkl'.format(epoch))
        checkpoint_path_2 = os.path.join(self.cfg.PATH.TMP_PATH + '/tbx_log_' + self.cfg.TRAIN.MODEL, 'now.pkl')
        if self.one_test:
            path_list = [checkpoint_path_1]
        else:
            path_list = [checkpoint_path_0, checkpoint_path_1, checkpoint_path_2]
        for path_i in path_list:
            os.makedirs(os.path.dirname(path_i), exist_ok=True)
            torch.save(self.Model.state_dict(), path_i)
            self.LOGGER.info('Epoch: %s, checkpoint is saved to %s', epoch, path_i)

    def _train_an_epoch(self, epoch, optimizer, scheduler):
        # pylint: disable=too-many-arguments
        self.Model.train()
        scheduler.step()
        self.LOGGER.info('>' * 30 + '[TRAIN] Model:%s,   Epoch: %s,   Learning Rate: %s', self.cfg.TRAIN.MODEL, epoch, optimizer.param_groups[0]['lr'])
        losses = 0
        # count the step time, total time...
        optimizer.zero_grad()
        _timer = Time()
        # for step in range(batch_num):
        for step, train_data in enumerate(self.trainDataloader):
            if step >= len(self.trainDataloader):
                break
            train_data = self.dataloader_factory.to_devce(train_data)
            # forward process
            predict = self.Model.forward(input_x=train_data[0], input_y=train_data[1], input_data=train_data, is_training=True)
            # calculate the total loss
            total_loss = self._calculate_loss(predict, train_data, losstype=self.cfg.TRAIN.LOSSTYPE)
            losses += total_loss.item()
            # backward process
            if self.args.tensor_core in ['O1', 'O2', 'O3']:
                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()

            if step % self.cfg.TRAIN.BATCH_BACKWARD_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            if (step + 1) % self.cfg.TRAIN.SAVE_STEP == 0:
                self._save_checkpoint(epoch)
            _timer.time_end()
            self.LOGGER.info('[TRAIN] Model: %s Epoch-Step:%3d-%4d/%4d, Step_LOSS: %8.4f, Batch_Average_LOSS: %8.4f, Time Step/Total-%s/%s', self.cfg.TRAIN.MODEL,
                             epoch, step, len(self.trainDataloader), total_loss.item(), losses / (step + 1), _timer.diff, _timer.from_begin)
            _timer.time_start()
        self.save_parameter.tbX_write(epoch=epoch, learning_rate=optimizer.param_groups[0]['lr'], batch_average_loss=losses / len(self.trainDataloader), )
        self.LOGGER.info('[TRAIN] Summary: Epoch: %s, average total loss: %s', epoch, losses / len(self.trainDataloader))

    def _test_an_epoch(self, epoch):
        if not self.one_test: self.Model.eval()
        self.LOGGER.info('[EVALUATE] Model:%s, Evaluating ...', self.cfg.TRAIN.MODEL)
        _timer = Time()
        self.Score.init_parameters()
        for step, train_data in enumerate(self.testDataloader):
            _timer.time_start()
            if step >= len(self.trainDataloader):
                break
            test_data = self.dataloader_factory.to_devce(train_data)
            # test_data = self.DataLoader.get_data_by_idx(test_set, step * batch_size, (step + 1) * batch_size, is_training=False)
            if test_data[0] is None: continue
            predict = self.Model.forward(input_x=test_data[0], input_y=test_data[1], input_data=test_data, is_training=False)
            if self.cfg.BELONGS in ['OBD']: test_data = test_data[1]
            self.Score.cal_score(predict, test_data)
            _timer.time_end()
            self.LOGGER.info('[EVALUATE] Epoch-Step:%3d-%4d/%4d, Time Step/Total-%s/%s', epoch, step, len(self.testDataloader), _timer.diff, _timer.from_begin)
        score_out, precision, recall = self.Score.score_out()
        self.save_parameter.tbX_write(epoch=epoch, score_out=score_out, precision=precision, recall=recall, )
        self.LOGGER.info('[EVALUATE] Summary: Epoch: %s, Score0: %s, Score1: %s, Score2: %s', epoch, score_out, precision, recall)
