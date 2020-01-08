"""Train the model.

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
from evasys.Score_Dict import Score
from util.util_show_save_parmeters import TrainParame
from net_works.Model_Loss_Dict import ModelDict, LossDict
from util.util_time_stamp import Time
from util.util_weights_init import weights_init
from util.util_get_train_test_dataset import _get_train_test_dataset, _read_train_test_dataset
from util.util_prepare_device import load_device
from util.util_load_state_dict import load_state_dict
from util.util_prepare_cfg import prepare_cfg
from dataloader.DataLoaderDict import DataLoaderDict

LOGGER = logging.getLogger(__name__)


class Solver:
    def __init__(self, cfg, args):

        self.cfg = prepare_cfg(cfg)
        self.args = args
        self.one_test = self.cfg.TEST.ONE_TEST
        self.cfg.TRAIN.DEVICE, self.device_ids = load_device(self.cfg)
        self.DataLoader = DataLoaderDict[self.cfg.BELONGS](self.cfg)
        self.save_parameter = TrainParame(self.cfg)
        self.Model = ModelDict[self.cfg.TRAIN.MODEL](self.cfg)
        self.LossFun = LossDict[self.cfg.TRAIN.MODEL](self.cfg)
        self.score = Score[self.cfg.BELONGS](self.cfg)
        self.train_batch_num = self.cfg.TEST.ONE_TEST_TRAIN_STEP
        self.test_batch_num = self.cfg.TEST.ONE_TEST_TEST_STEP

    def train(self):
        """Train the network.

        Args:
            self.cfg.TMP_PATH (str): Path to store files generated during training
            epochs (int): Total epochs to train
            checkpoint (str): Path to checkpoint

        """
        LOGGER.info('{} Start Training...'.format(self.cfg.TRAIN.MODEL))
        os.makedirs(self.cfg.PATH.TMP_PATH, exist_ok=True)
        # Prepare network, data set idx
        learning_rate, epoch_last, train_set, test_set = self._prepare_parameters()
        # Prepare optimizer
        optimizer, scheduler = self._get_optimizer(learning_rate, optimizer=self.cfg.TRAIN.OPTIMIZER)
        for epoch in range(epoch_last, self.cfg.TRAIN.EPOCH_SIZE):
            self._train_an_epoch(epoch, train_set, optimizer, scheduler)
            # saving the weights to checkpoint
            self._save_checkpoint(epoch)
            # evaluating from test data set
            self._test_an_epoch(epoch, test_set)

    def _prepare_parameters(self):
        """
        Get the self.Model, learning_rate, epoch_last, train_set, test_set.
        :return: learning_rate, epoch_last, train_set, test_set.
        """
        idx_stores_dir = os.path.join(self.cfg.PATH.TMP_PATH, 'idx_stores')
        # load the last train parameters
        if self.args.checkpoint:
            self.Model = load_state_dict(self.Model, self.args.checkpoint, self.cfg.TRAIN.DEVICE)
            epoch_last, learning_rate_last = self.save_parameter.tbX_read()
            epoch = self.args.epoch_continue if self.args.epoch_continue else epoch_last + 1
            learning_rate = self.args.lr_continue if self.args.lr_continue else learning_rate_last
            LOGGER.info('Loading Last Checkpoint: %s, Last Learning Rate:%s, Last Epoch:%s',
                        self.args.checkpoint, learning_rate, epoch)
            #  load the last data set
            train_set, test_set = _read_train_test_dataset(idx_stores_dir)
        else:
            weights_init(self.Model)
            # start a new train, delete the exist parameters
            self.save_parameter.clean_history_and_init_log()
            epoch = 0
            learning_rate = self.args.lr  # if self.args.lr else self.cfg.TRAIN.LR_CONTINUE
            # generate a new data set
            if self.cfg.TRAIN.TRAIN_DATA_FROM_FILE:
                train_set, test_set = _read_train_test_dataset(idx_stores_dir)
            else:
                train_set, test_set = _get_train_test_dataset(x_dir=self.cfg.PATH.IMG_PATH, y_dir=self.cfg.PATH.LAB_PATH, idx_stores_dir=idx_stores_dir,
                                                              test_train_ratio=self.cfg.TEST.TEST_SET_RATIO, cfg=self.cfg, )
            print(train_set[:4], '\n', test_set[:4])

        self.Model = self.Model.to(self.cfg.TRAIN.DEVICE)
        if len(self.device_ids) > 1:
            self.Model = torch.nn.DataParallel(self.Model, device_ids=self.device_ids)

        LOGGER.info('the train set is :{}, ant the test set is :{}'.format(len(train_set), len(test_set)))
        # _print_model_parm_nums(self.Model.cuda(), self.cfg.TRAIN.IMG_SIZE[0], self.cfg.TRAIN.IMG_SIZE[1])
        return learning_rate, epoch, train_set, test_set

    def _get_optimizer(self, learning_rate, optimizer='adam'):
        if optimizer == 'adam' or optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.Model.parameters(),
                                         lr=learning_rate, betas=(self.cfg.TRAIN.BETAS_ADAM, 0.999), weight_decay=5e-4)
        elif optimizer == 'sgd' or optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.Model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        else:
            LOGGER.error('NO such a optimizer: ' + str(optimizer))
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
            LOGGER.debug('Loss per head: %s', loss_head_info)
        else:
            total_loss = losses[0]
            LOGGER.debug('Train Acc is: %s', losses[1])
        if torch.isnan(total_loss) or total_loss.item() == float("inf") or total_loss.item() == -float("inf"):
            LOGGER.error("received an nan/inf loss")
            exit()
        return total_loss

    def _save_checkpoint(self, epoch):
        checkpoint_path_0 = os.path.join(self.cfg.PATH.TMP_PATH, 'checkpoint', '{}.pkl'.format(epoch))
        checkpoint_path_1 = os.path.join(self.cfg.PATH.TMP_PATH, 'checkpoint', 'now.pkl'.format(epoch))
        checkpoint_path_2 = os.path.join(self.cfg.PATH.TMP_PATH + '/tbx_log_' + self.cfg.TRAIN.MODEL, 'checkpoint.pkl')
        for path_i in [checkpoint_path_0, checkpoint_path_1, checkpoint_path_2]:
            os.makedirs(os.path.dirname(path_i), exist_ok=True)
            torch.save(self.Model.state_dict(), path_i)
            torch.save(self.Model.state_dict(), path_i)
            LOGGER.info('Epoch: %s, checkpoint is saved to %s', epoch, path_i)

    def _train_an_epoch(self, epoch, train_set, optimizer, scheduler):
        # pylint: disable=too-many-arguments
        self.Model.train()
        scheduler.step()
        LOGGER.info('[TRAIN] Epoch: %s, learing_rate: %s', epoch, optimizer.param_groups[0]['lr'])
        np.random.shuffle(train_set)
        batch_size = self.cfg.TRAIN.BATCH_SIZE
        batch_num = self.train_batch_num if self.one_test else len(train_set) // batch_size
        losses = 0
        # count the step time, total time...
        _timer = Time()
        optimizer.zero_grad()
        for step in range(batch_num):
            _timer.time_start()
            train_data = self.DataLoader.get_data_by_idx(train_set, step * batch_size, (step + 1) * batch_size, is_training=True)
            if train_data[1] is None:
                LOGGER.warning('[TRAIN] NO gt_labels IN THIS BATCH. Epoch: %3d, step: %4d/%4d ', epoch, step, batch_num)
                continue
            # forward process
            predict = self.Model.forward(input_x=train_data[0], input_y=train_data[1], input_data=train_data, is_training=True)
            # calculate the total loss
            total_loss = self._calculate_loss(predict, train_data, losstype=self.cfg.TRAIN.LOSSTYPE)
            losses += total_loss.item()
            # backward process
            total_loss.backward()
            if step % self.cfg.TRAIN.BATCH_BACKWARD_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            if (step + 1) % self.cfg.TRAIN.SAVE_STEP == 0:
                self._save_checkpoint(epoch)
            _timer.time_end()
            LOGGER.info('[TRAIN] Epoch-Step:%3d-%4d/%4d, Step_LOSS: %10.2f, Batch_Average_LOSS: %10.2f, Time Step/Total-%s/%s',
                        epoch, step, batch_num, total_loss.item(), losses / (step + 1), _timer.diff, _timer.from_begin)
        self.save_parameter.tbX_write(epoch=epoch, learning_rate=optimizer.param_groups[0]['lr'], batch_average_loss=losses / batch_num, )
        LOGGER.info('[TRAIN] Summary: Epoch: %s, average total loss: %s', epoch, losses / batch_num)

    def _test_an_epoch(self, epoch, test_set):
        # if epoch < 5: pass
        self.Model.eval()
        LOGGER.info('[EVALUATE] Evaluating from test data set ...')
        _timer = Time()
        batch_size = self.cfg.TRAIN.BATCH_SIZE
        batch_num = self.test_batch_num if self.one_test else len(test_set) // batch_size
        self.score.init_parameters()
        for step in range(batch_num):
            _timer.time_start()
            test_data = self.DataLoader.get_data_by_idx(test_set, step * batch_size, (step + 1) * batch_size, is_training=False)
            if test_data[0] is None: continue
            predict = self.Model.forward(input_x=test_data[0], input_y=test_data[1], input_data=test_data, is_training=False)
            if self.cfg.BELONGS in ['OBD']: test_data = test_data[1]
            self.score.cal_score(predict, test_data)
            _timer.time_end()
            LOGGER.info('[EVALUATE] Epoch-Step:%3d-%4d/%4d, Time Step/Total-%s/%s', epoch, step, batch_num, _timer.diff, _timer.from_begin)
        score_out, precision, recall = self.score.score_out()
        self.save_parameter.tbX_write(epoch=epoch, score_out=score_out, precision=precision, recall=recall)
        LOGGER.info('[EVALUATE] Summary: Epoch: %s, Score: %s, Precision: %s, Recall: %s', epoch, score_out, precision, recall)
