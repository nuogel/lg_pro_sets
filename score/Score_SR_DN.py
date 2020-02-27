import torch
import cv2
import math
import numpy as np
from util.util_parse_SR_img import parse_Tensor_img


class SR_DN_SCORE:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rate_all = 0.
        self.rate_batch = 0.
        self.batches = 0.
        self.mseloss = torch.nn.MSELoss()  # size_average=False

    def init_parameters(self):
        self.rate_all = 0.
        self.rate_batch = 0.
        self.batches = 0.

    def cal_score(self, predict, dataset):
        input, target = dataset
        self.rate_batch = 0.
        print('batch NO:', self.batches)
        self.batches += 1

        for batch_i in range(predict.shape[0]):
            self.rate_batch += self.PSNR(predict[batch_i], target[batch_i])
        self.rate_all += self.rate_batch / predict.shape[0]
        if self.cfg.TEST.SHOW_EVAL_TIME:
            input = torch.nn.functional.interpolate(input, size=(target.shape[1], target.shape[2]))
            input = input.permute(0, 2, 3, 1)
            img_cat = torch.cat([input, predict, target], dim=1)
            save_path = 'saved/denoise/' + str(self.cfg.TRAIN.MODEL) + '_' + str(self.cfg.TRAIN.IMG_SIZE[0]) + 'x' + str(self.cfg.TRAIN.IMG_SIZE[1])\
                        + 'x' + str(self.cfg.TRAIN.UPSCALE_FACTOR) + '.png'
            parse_Tensor_img(img_cat, pixcels_norm=self.cfg.TRAIN.PIXCELS_NORM, save_path=save_path, show_time=1)

    def score_out(self):
        score = self.rate_all / self.batches
        return score, None, None

    def PSNR(self, pred, gt):
        mse = torch.mean((pred - gt) ** 2).item()
        if mse == 0:
            return 100
        return 10 * math.log10(((255. - self.cfg.TRAIN.PIXCELS_NORM[0]) / self.cfg.TRAIN.PIXCELS_NORM[1]) ** 2 / mse)
