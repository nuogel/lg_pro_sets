import torch
from dataloader.loader_asr import DataLoader


class ASR_SCORE:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rate_all = 0.
        self.rate_batch = 0.
        self.batches = 0
        self.dataloader = DataLoader(cfg)

    def init_parameters(self):
        self.rate_all = 0.
        self.rate_batch = 0.
        self.batches = 0

    def cal_score(self, pre, gt_data):
        self.rate_batch = 0.
        _, target, input_lengths, target_lengths = gt_data
        print('batch NO:', self.batches)
        for i in range(self.cfg.TRAIN.BATCH_SIZE):
            pre_i = pre[i][:target_lengths[i]]
            gt_i = target[i, :target_lengths[i]].cpu()
            print('pr_:', i, self.dataloader._number2pinying(pre_i))
            print('gt_:', i, self.dataloader._number2pinying(gt_i.numpy().tolist()))
            if len(pre_i) != target_lengths[i]: continue
            pre_i = torch.Tensor(pre_i)
            yes = torch.eq(pre_i[:target_lengths[i]], gt_i.type(torch.FloatTensor))
            self.rate_batch += float(yes.sum() / target_lengths[i])

        self.rate_all += self.rate_batch / self.cfg.TRAIN.BATCH_SIZE
        self.batches += 1

    def score_out(self):
        return self.rate_all / self.batches, None, None
