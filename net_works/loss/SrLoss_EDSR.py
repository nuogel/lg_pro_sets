import torch
from torch.nn import L1Loss


class EDSRLoss:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loss_l1loss = L1Loss()  # reduction='sum'

    def Loss_Call(self, prediction, train_data, losstype='mse'):
        low_img, high_img = train_data
        loss = self.loss_l1loss(prediction, high_img)
        return loss, None
