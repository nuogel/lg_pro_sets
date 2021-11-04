"""Loss calculation based on yolo."""
import torch
import numpy as np
import torch.nn.functional as F
'''
with the new yolo loss, in 56 images, loss is 0.18 and map is 0.2.and the test show wrong bboxes.
with the new yolo loss, in 8 images, loss is 0.015 and map is 0.99.and the test show terrible bboxes.

'''


class IMCLoss:
    # pylint: disable=too-few-public-methods
    """Calculate loss."""

    def __init__(self, cfg):
        """Init."""
        #
        self.cfg = cfg
        self.device = self.cfg.TRAIN.DEVICE
        self.cls_num = cfg.TRAIN.CLASSES_NUM
        self.one_test = cfg.TEST.ONE_TEST

        self.reduction = 'mean'

    def Loss_Call(self, f_maps, dataset, kwargs):
        images, labels, datainfos = dataset
        total_loss = F.cross_entropy(f_maps, labels)
        cls = torch.argmax(torch.softmax(f_maps, -1), -1)
        score = torch.sum(cls== labels) / f_maps.size(0)
        score = score.item()
        metrics = {'total_loss': total_loss, 'train score': score}
        return {'total_loss': total_loss, 'metrics': metrics}
