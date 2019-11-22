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
from torch.optim import lr_scheduler
from evasys.score import Score
from util.util_show_save_parmeters import TrainParame
from net_works.model.Model_Loss_Dict import ModelDict, LossDict
from util.util_time_stamp import Time
from util.util_weights_init import weights_init
from util.util_get_data_idx_stores import _get_data_idx_stores
from util.util_is_use_cuda import _is_use_cuda
from util.util_print_model_parm_nums import _print_model_parm_nums
from dataloader.DataLoaderDict import DataLoaderDict

LOGGER = logging.getLogger(__name__)


class Solver:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.one_test = cfg.TEST.ONE_TEST
        self.one_name = cfg.TEST.ONE_NAME
        if self.one_test:
            self.cfg.TRAIN.BATCH_SIZE = len(self.one_name)
        self.DataLoader = DataLoaderDict[cfg.BELONGS](cfg)
        self.save_parameter = TrainParame(cfg)
        self.Model = ModelDict[cfg.TRAIN.MODEL](cfg)
        self.LossFun = LossDict[cfg.TRAIN.MODEL](cfg)
        self.score = Score[cfg.BELONGS](cfg)
        self.train_batch_num = 50
        self.test_batch_num = 1

    def train(self):
        """Train the network.

        Args:
            self.cfg.TMP_PATH (str): Path to store files generated during training
            epochs (int): Total epochs to train
            checkpoint (str): Path to checkpoint

        """
        LOGGER.info('Apollo Yolo Start Training...')
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
        if _is_use_cuda(self.cfg.TRAIN.GPU_NUM):
            self.Model = self.Model.cuda(self.cfg.TRAIN.GPU_NUM)
        # load the last train parameters
        checkpoint = self.args.checkpoint
        if checkpoint:
            self.Model.load_state_dict(torch.load(checkpoint))
            dict_loaded = torch.load(self.cfg.PATH.PARAMETER_PATH)
            epoch_last = self.args.epoch_continue if self.args.epoch_continue else dict_loaded['epoch'] + 1
            learning_rate = self.args.lr if self.args.lr else dict_loaded['learning_rate'][epoch_last - 1]
            LOGGER.info('Loading last checkpoint: %s, last learning rate:%s, last epoch:%s',
                        os.path.split(checkpoint)[1], learning_rate, epoch_last)
            #  load the last data set
            train_set = torch.load(os.path.join(idx_stores_dir, 'train_set'))
            test_set = torch.load(os.path.join(idx_stores_dir, 'test_set'))

        else:
            weights_init(self.Model)
            # start a new train, delete the exist parameters
            self.save_parameter.clear_parameters()
            epoch_last = 0
            learning_rate = self.args.lr if self.args.lr else self.cfg.TRAIN.LR_START
            # generate a new data set
            train_set, test_set = _get_data_idx_stores(lab_dir=self.cfg.PATH.LAB_PATH, idx_stores_dir=idx_stores_dir,
                                                       test_train_ratio=self.cfg.TEST.TEST_SET_RATIO, cfg=self.cfg, )
        LOGGER.info('the train set is :{}, ant the test set is :{}'.format(len(train_set), len(test_set)))
        print(train_set[:10], test_set[:10])
        # _print_model_parm_nums(self.Model.cuda(), self.cfg.TRAIN.IMG_SIZE[0], self.cfg.TRAIN.IMG_SIZE[1])
        return learning_rate, epoch_last, train_set, test_set

    def _get_optimizer(self, learning_rate, optimizer='adam'):
        if optimizer == 'adam' or optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.Model.parameters(),
                                         lr=learning_rate, betas=(self.cfg.TRAIN.BETAS_ADAM, 0.99), weight_decay=5e-4)
        elif optimizer == 'sgd' or optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.Model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        else:
            LOGGER.error('no optimizer...')
        # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg.LR_EXPONENTIAL_DECAY_RATE)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return optimizer, scheduler

    def _calculate_loss(self, predict, dataset, losstype=None):
        total_loss = 0.
        losses = self.LossFun.Loss_Call(predict, dataset, losstype=losstype)
        if self.cfg.BELONGS == 'img':
            loss_names = ['[obj_loss]', '[noobj_loss]', '[cls_loss]', '[loc_loss]']  #obj_loss, noobj_loss, cls_loss, loc_loss
            loss_tmp = range(len(losses))
            for i in loss_tmp:
                total_loss += losses[i]
            loss_head_info = ''
            for loss_name, head_loss in zip(loss_names[:len(loss_tmp)], losses):
                loss_head_info += ' {}: {:6.4f}'.format(loss_name, head_loss.item())
            LOGGER.debug('Loss per head: %s', loss_head_info)
        if self.cfg.BELONGS == 'ASR':
            total_loss = losses[0]
            LOGGER.debug('Train Acc is: %s', losses[1])
        if torch.isnan(total_loss) or total_loss.item() == float("inf") or total_loss.item() == -float("inf"):
            LOGGER.error("received an nan/inf loss")
            exit()
        return total_loss

    def _save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.cfg.PATH.TMP_PATH, 'checkpoint', '{}.pkl'.format(epoch))
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.Model.state_dict(), checkpoint_path)
        LOGGER.info('Epoch: %s, checkpoint is saved to %s', epoch, checkpoint_path)

    def _train_an_epoch(self, epoch, train_set, optimizer, scheduler):
        # pylint: disable=too-many-arguments
        self.Model.train()
        scheduler.step()
        LOGGER.info('[TRAIN] Epoch: %s, learing_rate: %s', epoch, optimizer.param_groups[0]['lr'])
        np.random.shuffle(train_set)
        batch_size = self.cfg.TRAIN.BATCH_SIZE
        if self.one_test:
            batch_num = self.train_batch_num
            batch_size = 1
        else:
            batch_num = len(train_set) // batch_size
        losses = 0
        # count the step time, total time...
        t1_timer = Time()
        optimizer.zero_grad()
        for step in range(batch_num):
            t1_timer.time_start()
            train_data = self.DataLoader.get_data_by_idx(train_set, step * batch_size, (step + 1) * batch_size)
            if train_data[1] is None:
                LOGGER.warning('[TRAIN] NO gt_labels IN THIS BATCH. Epoch: %3d, step: %4d/%4d ', epoch, step, batch_num)
                t1_timer.time_end()
                LOGGER.info('Step Time is: %.4f & Total Time is: %.4f', t1_timer.diff, t1_timer.from_begin)
                continue
            # forward process
            predict = self.Model.forward(train_data)
            # calculate the total loss
            total_loss = self._calculate_loss(predict, train_data, losstype=self.cfg.TRAIN.LOSSTYPE)
            losses += total_loss.item()
            LOGGER.info('[TRAIN] Epoch: %3d, step: %4d/%4d, Step_LOSS: %10.2f, '
                        'Batch_Average_LOSS: %10.2f',
                        epoch, step, batch_num, total_loss.item(), losses / (step + 1))
            # backward process
            LOGGER.info('backwarding...')
            total_loss.backward()
            if step % self.cfg.TRAIN.BATCH_BACKWARD_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            if (step + 1) % 100 == 0:
                self._save_checkpoint(epoch)
            t1_timer.time_end()
            LOGGER.info('Step Time is: %.4f & Total Time is: %.4F', t1_timer.diff, t1_timer.from_begin)
        # save the main parameters
        self.save_parameter.save_parameters(epoch=epoch, batch_average_loss=losses / batch_num,
                                            learning_rate=optimizer.param_groups[0]['lr'])
        LOGGER.info('[TRAIN] Summary: Epoch: %s, average total loss: %s', epoch, losses / batch_num)

    def _test_an_epoch(self, epoch, test_set):
        # if epoch < 5: pass
        self.Model.eval()
        LOGGER.info('[EVALUATE] Evaluating from test data set ...')
        batch_size = self.cfg.TRAIN.BATCH_SIZE
        if self.one_test:
            batch_num = self.test_batch_num
            batch_size = 1
        else:
            batch_num = len(test_set[:800]) // batch_size
        losses = 0.
        # print(score.true_positive, score.false_positive, score.obj_num)
        self.score.init_parameters()
        for step in range(batch_num):
            test_data = self.DataLoader.get_data_by_idx(test_set, step * batch_size, (step + 1) * batch_size)
            if test_data[0] is None:
                continue
            # forward process
            predict = self.Model.forward(test_data, eval=True)
            '''
                        calculate the total loss
            total_loss = self._calculate_loss(predict, test_data, losstype=self.cfg.TRAIN.LOSSTYPE)
            losses += total_loss.item()
            losses += total_loss.item()
            LOGGER.info('[EVALUATE] Epoch: %3d, step: %4d/%4d, step_test_loss:  %10.2f, '
                        'test_average_loss:  %10.2f',
                        epoch, step, batch_num, total_loss, losses / (step + 1))
            calculate the score
            '''
            if self.cfg.BELONGS == 'img':
                test_data = test_data[1]
            self.score.cal_score(predict, test_data)
        score_out, precision, recall = self.score.score_out()
        self.save_parameter.save_parameters(epoch=epoch, f1_score=score_out, precision=precision, recall=recall)
        LOGGER.info('[EVALUATE] Summary: Epoch: %s, average test loss: %s', epoch, losses / batch_num)
        LOGGER.info('[EVALUATE] Summary: Epoch: %s, Score: %s, Precision: %s, Recall: %s',
                    epoch, score_out, precision, recall)
