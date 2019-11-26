import torch
import cv2
import math
import numpy as np
from util.util_parse_SR_img import parse_Tensor_img


class SR_SCORE:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rate_all = 0.
        self.rate_batch = 0.
        self.batches = 1

    def init_parameters(self):
        self.rate_all = 0.
        self.rate_batch = 0.
        self.batches = 1

    def cal_score(self, predict, dataset):
        input, target = dataset
        self.rate_batch = 0.
        print('batch NO:', self.batches)
        self.batches += 1

        for batch_i in range(predict.shape[0]):
            self.rate_batch += self.PSNR(predict[batch_i], target[batch_i])
        self.rate_all += self.rate_batch / predict.shape[0]
        # parse_Tensor_img(predict, save_path=None, show_time=10000)

    def score_out(self):
        score = self.rate_all / self.batches
        return score, None, None

    def denormalize(self, tensors):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        """ Denormalizes image tensors using mean and std """
        for c in range(3):
            tensors[:, c].mul_(std[c]).add_(mean[c])
        return torch.clamp(tensors, 0, 255)

    def PSNR(self, pred, gt, shave_border=0):
        height, width = pred.shape[:2]
        pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
        gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
        imdff = pred - gt
        rmse = torch.sqrt(torch.mean(imdff ** 2)).item()
        if rmse == 0:
            return 100
        return 20 * math.log10(255.0 / rmse)
