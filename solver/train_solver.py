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

    def train(self):
        """Train the network.
        """
        self.cfg.logger.info('>' * 30 + 'Loading Checkpoint: %s, Last Learning Rate:%s, Last Epoch:%s',
                             self.args.checkpoint, self.learning_rate, self.epoch_last)

        for epoch in range(self.epoch_last, self.cfg.TRAIN.EPOCH_SIZE):
            self.epoch = epoch
            if not self.cfg.TEST.TEST_ONLY and not self.args.test_only:
                self._train_an_epoch(epoch)
            if epoch > -1 or self.cfg.TEST.ONE_TEST:
                self._test_an_epoch(epoch)

    def _train_an_epoch(self, epoch):
        self.model.train()
        self.cfg.logger.info('>' * 30 + '[TRAIN] model:%s,   epoch: %s' % (self.cfg.TRAIN.MODEL, epoch))
        epoch_losses = 0
        # count the step time, total time...
        print(('\n[train] ' + '%8s|' + '%8s|' * 6) % ('model_n', 'epoch', 'g_step', 'l_rate', 'step_l', 'aver_l', 'others'))
        Pbar = tqdm.tqdm(self.trainDataloader)
        for step, train_data in enumerate(Pbar):
            if self.global_step < self.cfg.TRAIN.WARM_UP_STEP:
                self.optimizer = self._set_warmup_lr(self.optimizer)
            train_data = self.DataFun.to_devce(train_data)
            # forward process
            predict = self.model.forward(input_x=train_data[0], input_y=train_data[1], input_data=train_data,
                                         is_training=True)
            # calculate the total loss
            total_loss, loss_metrics = self._calculate_loss(predict, train_data, losstype=self.cfg.TRAIN.LOSSTYPE)
            epoch_losses += total_loss.item()
            # backward process

            total_loss.backward()

            self.global_step += 1
            if self.global_step % self.cfg.TRAIN.SAVE_STEP == 0:
                self._save_checkpoint()
            if self.global_step % self.cfg.TRAIN.BATCH_BACKWARD_SIZE == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.ema: self.ema.update(self.model)

            info_base = (self.cfg.TRAIN.MODEL, epoch, self.global_step, '%0.5f' % self.optimizer.param_groups[0]['lr'],
                         '%0.3f' % total_loss.item(), '%0.3f' % (epoch_losses / (10 + 1)))

            info = ('%16s|' + '%8s|' * 5) % info_base + ' ' * 2
            for k, v in loss_metrics.items():
                info += k + ':' + '%.3f' % v + '|'

            if self.global_step % 50 == 0:
                self.cfg.logger.info(info)
            Pbar.set_description(info)
        self.scheduler.step()
        if self.ema: self.ema.update_attr(self.model)
        w_dict = {'epoch': epoch,
                  'lr': self.optimizer.param_groups[0]['lr'],
                  'epoch_loss': epoch_losses / len(self.trainDataloader)}
        self.cfg.writer.tbX_write(w_dict)
        self._save_checkpoint()

    def _test_an_epoch(self, epoch):
        if not self.cfg.TEST.ONE_TEST: self.model.eval()
        self.cfg.logger.debug('[evaluate] model:%s, evaluating ...', self.cfg.TRAIN.MODEL)
        self.score.init_parameters()
        Pbar = tqdm.tqdm(self.testDataloader)
        for step, train_data in enumerate(Pbar):
            if step >= len(self.trainDataloader):
                break
            test_data = self.DataFun.to_devce(train_data)
            if test_data[0] is None: continue
            if self.ema:
                predict = self.ema.ema(input_x=test_data[0], input_y=test_data[1], input_data=test_data, is_training=False)
            else:
                predict = self.model(input_x=test_data[0], input_y=test_data[1], input_data=test_data, is_training=False)
            if self.cfg.BELONGS in ['OBD']: test_data = test_data[1]
            self.score.cal_score(predict, test_data)
            Pbar.set_description('[valid]')

        main_score, item_score = self.score.score_out()
        w_dict = {'epoch': epoch,
                  'main_score': main_score,
                  'item_score': item_score}
        self.cfg.writer.tbX_write(w_dict)
        self.cfg.logger.debug('[EVALUATE] Summary: Epoch: %s, total_score: %s, other_score: %s', epoch, str(main_score), str(item_score))
        if self.cfg.TEST.TEST_ONLY: exit()
