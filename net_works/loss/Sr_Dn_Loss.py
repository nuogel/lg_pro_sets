import torch
from torch.nn import L1Loss


class SR_DN_Loss:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loss_l1loss = L1Loss()  # reduction='sum'
        self.mseloss = torch.nn.MSELoss()  # size_average=False

    def Loss_Call(self, prediction, train_data, losstype='MSE'):
        low_img, high_img = train_data
        if losstype == 'MSE' or losstype == 'mse':
            loss = self.mseloss(prediction, high_img)
        elif losstype == 'L1' or losstype == 'l1':
            loss = self.loss_l1loss(prediction, high_img)
        else:
            loss = None
        return loss, None
