"""Train the model.
writer:luogeng 2017.02
Train.py is used for training the net to mark things with a box outside of
them, and with a label of what it is, and with it's score at the left top of
the box.

At the end, we will get the weight file of the net.

"""
import os
import torch
import torch.nn
import tqdm
from .solver_base import SolverBase
from util.util_time_stamp import Time
from util.util_weights_init import weights_init
from util.util_get_dataset_from_file import _get_train_test_dataset, _read_train_test_dataset
from util.util_load_state_dict import load_state_dict
from others.Quantization.pq_quantization.util_quantization import util_quantize_model


class Solver(SolverBase):

    def train(self):
        """Train the network.

        Args:
            self.cfg.TMP_PATH (str): Path to store files generated during training
            epochs (int): Total epochs to train
            checkpoint (str): Path to checkpoint

        """
        self.cfg.logger.info('>' * 30 + '{} Start Training'.format(self.cfg.TRAIN.MODEL))
        os.makedirs(self.cfg.PATH.TMP_PATH, exist_ok=True)
        # Prepare network, data set idx
        learning_rate, epoch_last = self._prepare_parameters()
        # Prepare optimizer
        optimizer, scheduler = self._get_optimizer(learning_rate)
        if self.args.tensor_core in ['O1', 'O2', 'O3']:
            from apex import amp
            self.Model, optimizer = amp.initialize(self.Model, optimizer, opt_level=self.args.tensor_core,
                                                   loss_scale="dynamic")
        for epoch in range(epoch_last, self.cfg.TRAIN.EPOCH_SIZE):
            if not self.cfg.TEST.TEST_ONLY and not self.args.test_only:
                self._train_an_epoch(epoch, optimizer, scheduler)
                self._save_checkpoint(self.Model, epoch, optimizer.param_groups[0]['lr'], self.global_step)
            if epoch > 0 or self.cfg.TEST.ONE_TEST:
                self._test_an_epoch(epoch)

    def _prepare_parameters(self):
        """
        Get the self.Model, learning_rate, epoch_last, train_set, test_set.
        :return: learning_rate, epoch_last, train_set, test_set.
        """
        # load the last train parameters
        if self.args.checkpoint in [0, '0', 'None', 'no', 'none', "''"]:
            weights_init(self.Model, self.cfg)
            # start a new train, delete the exist parameters
            self.cfg.writer.clean_history_and_init_log()
            epoch = 0
            self.global_step = 0
            learning_rate = self.args.lr  # if self.args.lr else self.cfg.TRAIN.LR_CONTINUE
            # generate a new data set
            if self.cfg.TRAIN.TRAIN_DATA_FROM_FILE:
                train_set, test_set = _read_train_test_dataset(self.cfg)
            else:
                train_set, test_set = _get_train_test_dataset(self.cfg)
        else:
            self.Model, epoch_last, learning_rate_last, self.global_step = load_state_dict(self.Model, self.args.checkpoint, self.cfg.TRAIN.DEVICE)
            epoch = self.args.epoch_continue if self.args.epoch_continue else epoch_last
            self.cfg.writer.tbX_reStart(epoch)
            learning_rate = self.args.lr_continue if self.args.lr_continue else learning_rate_last
            self.cfg.logger.info('>' * 30 + 'Loading Last Checkpoint: %s, Last Learning Rate:%s, Last Epoch:%s',
                                 self.args.checkpoint, learning_rate, epoch)
            #  load the last data set
            train_set, test_set = _read_train_test_dataset(self.cfg)

        print('train set:', train_set[:4], '\n', 'test set:', test_set[:4])
        self.cfg.logger.info(
            '>' * 30 + 'The train set is :{}, and The test set is :{}'.format(len(train_set), len(test_set)))
        self.trainDataloader, self.testDataloader = self.DataFun.make_dataset(train_set, test_set)

        self.Model = self.Model.to(self.cfg.TRAIN.DEVICE)
        if len(self.device_ids) > 1:
            self.Model = torch.nn.DataParallel(self.Model, device_ids=self.device_ids)
        quantization = 0
        if quantization:
            self.Model = util_quantize_model(self.Model)

        return learning_rate, epoch

    def _calculate_loss(self, predict, dataset, **kwargs):
        total_loss = 0.
        loss_head_info = ''
        losses = self.LossFun.Loss_Call(predict, dataset, kwargs=kwargs)
        w_dict = {}
        for k, v in losses.items():
            total_loss += v
            loss_head_info += ' {}: {:6.4f}'.format(k, v.item())
            w_dict['item_losses/' + k] = v
        # add tensorboard writer.
        if self.global_step % 200 == 0:
            w_dict['epoch'] = self.global_step
            self.cfg.writer.tbX_write(w_dict=w_dict)
        self.cfg.logger.debug(loss_head_info)
        if torch.isnan(total_loss) or total_loss.item() == float("inf") or total_loss.item() == -float("inf"):
            self.cfg.logger.error("received an nan/inf loss")
            exit()
        return total_loss

    def _train_an_epoch(self, epoch, optimizer, scheduler):
        # pylint: disable=too-many-arguments
        self.Model.train()
        self.cfg.logger.debug('>' * 30 + '[TRAIN] Model:%s,   Epoch: %s,   Learning Rate: %s',
                              self.cfg.TRAIN.MODEL, epoch, optimizer.param_groups[0]['lr'])
        losses = 0
        # count the step time, total time...
        optimizer.zero_grad()
        Pbar = tqdm.tqdm(self.trainDataloader)
        for step, train_data in enumerate(Pbar):
            train_data = self.DataFun.to_devce(train_data)
            # forward process
            predict = self.Model.forward(input_x=train_data[0], input_y=train_data[1], input_data=train_data,
                                         is_training=True)
            # calculate the total loss
            total_loss = self._calculate_loss(predict, train_data, losstype=self.cfg.TRAIN.LOSSTYPE)
            losses += total_loss.item()
            # backward process
            if self.args.tensor_core in ['O1', 'O2', 'O3']:
                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()

            self.global_step += 1
            if self.global_step % self.cfg.TRAIN.SAVE_STEP == 0:
                self._save_checkpoint(self.Model, epoch, optimizer.param_groups[0]['lr'], self.global_step)
            if self.global_step % self.cfg.TRAIN.BATCH_BACKWARD_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            info = '[train] model:%s;  step loss: %0.4f;  batch average loss: %0.4f;  epoch:%0d;  global step:%d, ' \
                   % (self.cfg.TRAIN.MODEL, total_loss.item(), losses / (step + 1), epoch, self.global_step)
            self.cfg.logger.debug(info)
            Pbar.set_description(info)
        scheduler.step()
        w_dict = {'epoch': epoch,
                  'learning_rate': optimizer.param_groups[0]['lr'],
                  'batch_average_loss': losses / len(self.trainDataloader)}
        self.cfg.writer.tbX_write(w_dict)
        self.cfg.logger.debug('[train] summary: epoch: %s, average total loss: %s', epoch, losses / len(self.trainDataloader))

    def _test_an_epoch(self, epoch):
        if not self.cfg.TEST.ONE_TEST: self.Model.eval()
        self.cfg.logger.debug('[EVALUATE] Model:%s, Evaluating ...', self.cfg.TRAIN.MODEL)
        _timer = Time()
        self.Score.init_parameters()
        Pbar = tqdm.tqdm(self.testDataloader)
        for step, train_data in enumerate(Pbar):
            _timer.time_start()
            if step >= len(self.trainDataloader):
                break
            test_data = self.DataFun.to_devce(train_data)
            if test_data[0] is None: continue
            predict = self.Model.forward(input_x=test_data[0], input_y=test_data[1], input_data=test_data,
                                         is_training=False)
            if self.cfg.BELONGS in ['OBD']: test_data = test_data[1]
            self.Score.cal_score(predict, test_data)
            _timer.time_end()
            info = '[EVALUATE] Epoch:%3d, Time Step/Total-%s/%s' % (epoch, _timer.diff, _timer.from_begin)
            self.cfg.logger.debug(info)
            Pbar.set_description(info)

        main_score, item_score = self.Score.score_out()
        w_dict = {'epoch': epoch,
                  'main_score': main_score,
                  'item_score': item_score}
        self.cfg.writer.tbX_write(w_dict)
        self.cfg.logger.debug('[EVALUATE] Summary: Epoch: %s, total_score: %s, other_score: %s', epoch, str(main_score),
                              str(item_score))
        if self.cfg.TEST.TEST_ONLY: exit()
