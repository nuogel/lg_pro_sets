import torch
import torch.nn as nn
from math import log10
from util.util_parse_SR_img import parse_Tensor_img

class SR_DN_Loss:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loss_l1loss = nn.L1Loss()  # reduction='sum'
        self.mseloss = nn.MSELoss()  # size_average=False

    def Loss_Call(self, prediction, train_data, losstype='MSE'):
        low_img, high_img = train_data
        parse_Tensor_img(prediction, pixcels_norm=self.cfg.TRAIN.PIXCELS_NORM, show_time=1)
        if losstype == 'MSE' or losstype == 'mse':
            loss = self.mseloss(prediction, high_img)
            mse = loss
        elif losstype == 'L1' or losstype == 'l1':
            loss = self.loss_l1loss(prediction, high_img)
            mse = self.mseloss(prediction, high_img)
        else:
            loss = None

        PSNR = 10 * log10(1 / mse.item())
        print('PSNR: ', PSNR)
        return loss, None
