"""Train the model.
writer:luogeng 2017.02
Train.py is used for training the net to mark things with a box outside of
them, and with a label of what it is, and with it's score at the left top of
the box.

At the end, we will get the weight file of the net.

"""
import tqdm
from .solver_base import BaseSolver
from util.util_time_stamp import Time


class Solver(BaseSolver):
    def __init__(self, cfg, args, train=True):
        super().__init__(cfg, args, train)
        self._get_optimizer()
        self._get_score()
        self._get_lossfun()
        self._get_dataloader()

    def train(self):
        """Train the network.
        """
        for epoch in range(self.epoch_last, self.cfg.TRAIN.EPOCH_SIZE):
            if not self.cfg.TEST.TEST_ONLY and not self.args.test_only:
                self._train_an_epoch(epoch)
            if epoch > 0 or self.cfg.TEST.ONE_TEST:
                self._test_an_epoch(epoch)

    def _train_an_epoch(self, epoch):
        self.model.train()
        self.cfg.logger.debug('>' * 30 + '[TRAIN] model:%s,   epoch: %s' % (self.cfg.TRAIN.MODEL, epoch))
        losses = 0
        # count the step time, total time...
        Pbar = tqdm.tqdm(self.trainDataloader)
        for step, train_data in enumerate(Pbar):
            if self.global_step < self.cfg.TRAIN.WARM_UP_STEP:
                self.optimizer = self._set_warmup_lr(self.optimizer)
            train_data = self.DataFun.to_devce(train_data)
            # forward process
            predict = self.model.forward(input_x=train_data[0], input_y=train_data[1], input_data=train_data,
                                         is_training=True)
            # calculate the total loss
            total_loss = self._calculate_loss(predict, train_data, losstype=self.cfg.TRAIN.LOSSTYPE)
            losses += total_loss.item()
            # backward process

            total_loss.backward()

            self.global_step += 1
            if self.global_step % self.cfg.TRAIN.SAVE_STEP == 0:
                self._save_checkpoint(self.model, epoch, self.optimizer, self.global_step)
            if self.global_step % self.cfg.TRAIN.BATCH_BACKWARD_SIZE == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.model.zero_grad()
            info = '[train] model:%s;lr:%0.6f; step loss: %0.4f; batch average loss:%0.4f; epoch:%0d; global step:%d' \
                   % (self.cfg.TRAIN.MODEL, self.optimizer.param_groups[0]['lr'], total_loss.item(), losses / (step + 1), epoch, self.global_step)
            self.cfg.logger.debug(info)
            Pbar.set_description(info)
        self.scheduler.step()
        w_dict = {'epoch': epoch,
                  'learning_rate': self.optimizer.param_groups[0]['lr'],
                  'batch_average_loss': losses / len(self.trainDataloader)}
        self.cfg.writer.tbX_write(w_dict)
        self.cfg.logger.debug('[train] summary: epoch: %s, average total loss: %s', epoch, losses / len(self.trainDataloader))
        self._save_checkpoint(self.model, epoch, self.optimizer, self.global_step)

    def _test_an_epoch(self, epoch):
        if not self.cfg.TEST.ONE_TEST: self.model.eval()
        self.cfg.logger.debug('[EVALUATE] Model:%s, Evaluating ...', self.cfg.TRAIN.MODEL)
        _timer = Time()
        self.score.init_parameters()
        Pbar = tqdm.tqdm(self.testDataloader)
        for step, train_data in enumerate(Pbar):
            _timer.time_start()
            if step >= len(self.trainDataloader):
                break
            test_data = self.DataFun.to_devce(train_data)
            if test_data[0] is None: continue
            predict = self.model.forward(input_x=test_data[0], input_y=test_data[1], input_data=test_data,
                                         is_training=False)
            if self.cfg.BELONGS in ['OBD']: test_data = test_data[1]
            self.score.cal_score(predict, test_data)
            _timer.time_end()
            info = '[EVALUATE] Epoch:%3d, Time Step/Total-%s/%s' % (epoch, _timer.diff, _timer.from_begin)
            self.cfg.logger.debug(info)
            Pbar.set_description(info)

        main_score, item_score = self.score.score_out()
        w_dict = {'epoch': epoch,
                  'main_score': main_score,
                  'item_score': item_score}
        self.cfg.writer.tbX_write(w_dict)
        self.cfg.logger.debug('[EVALUATE] Summary: Epoch: %s, total_score: %s, other_score: %s', epoch, str(main_score),
                              str(item_score))
        if self.cfg.TEST.TEST_ONLY: exit()
