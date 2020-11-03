import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from lgdet.util.util_iou import iou_xyxy, xywh2xyxy, xyxy2xywh, iou_xywh
import numpy as np
from lgdet.loss.loss_base.focal_loss import FocalLoss_lg


class RETINANETLOSS():
    def __init__(self, cfg):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        self.cfg = cfg
        self.num_cls = self.cfg.TRAIN.CLASSES_NUM
        self.neg_iou_threshold = 0.5
        self.neg_pos_ratio = 3  # 3:1
        self.reduction = 'mean'
        self.focalloss = FocalLoss_lg(reduction=self.reduction)

    def Loss_Call(self, predictions, targets, **kwargs):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        pre_cls, pre_loc, anc_xywh = predictions
        batchsize = pre_loc.shape[0]
        _, gt_labels, infos = targets

        gt_loc, gt_cls = [], []
        for i in range(batchsize):
            gt_i = gt_labels[gt_labels[..., 0] == i]
            lab = gt_i[..., 1].long()
            box = gt_i[..., 2:]
            box_xywh = xyxy2xywh(box)
            _gt_loc_xywh, _labels = self._assign_priors(box_xywh, lab, anc_xywh, self.neg_iou_threshold)
            _gt_loc_xywh = self._encode_bbox(_gt_loc_xywh, anc_xywh)
            gt_loc.append(_gt_loc_xywh)
            gt_cls.append(_labels)

        gt_loc = torch.stack(gt_loc, dim=0)
        gt_cls = torch.stack(gt_cls, dim=0)
        num_classes = pre_cls.size(2)

        pos_mask = gt_cls < self.num_cls
        cls_loss = self.focalloss(pre_cls, gt_cls, pos_mask)

        pre_loc = pre_loc[pos_mask, :].reshape(-1, 4)
        gt_loc = gt_loc[pos_mask, :].reshape(-1, 4)

        loc_loss = F.smooth_l1_loss(pre_loc, gt_loc)

        # num_pos = gt_loc.size(0)
        loc_loss = loc_loss
        class_loss = loc_loss
        total_loss = loc_loss + class_loss

        metrics = {'loc_loss': loc_loss.item(),
                   'cls_loss': class_loss.item()}
        return {'total_loss': total_loss, 'metrics': metrics}

    def _assign_priors(self, gt_boxes, gt_labels, corner_form_priors,
                       iou_threshold):
        """Assign ground truth boxes and targets to priors.

        Args:
            gt_boxes (num_targets, 4): ground truth boxes.
            gt_labels (num_targets): labels of targets.
            priors (num_priors, 4): corner form priors
        Returns:
            boxes (num_priors, 4): real values for priors.
            labels (num_priros): labels for priors.
        """
        # size: num_priors x num_targets
        ious = iou_xywh(corner_form_priors, gt_boxes, type='N2N')
        # size: num_priors
        try:
            best_target_per_prior, best_target_per_prior_index = ious.max(1)
        except:
            print('xxxx')
        # size: num_targets
        best_prior_per_target, best_prior_per_target_index = ious.max(0)

        for target_index, prior_index in enumerate(best_prior_per_target_index):
            best_target_per_prior_index[prior_index] = target_index
        # 2.0 is used to make sure every target has a prior assigned
        best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)  # fill 2>1.
        # size: num_priors
        labels = gt_labels[best_target_per_prior_index]
        labels[best_target_per_prior < iou_threshold] = self.num_cls  # the backgournd id
        boxes = gt_boxes[best_target_per_prior_index]
        return boxes, labels

    def _encode_bbox(self, xywh_boxes, xywh_priors):
        encode_target = torch.cat([(xywh_boxes[..., :2] - xywh_priors[..., :2]) / xywh_priors[..., 2:] / 0.1,
                                   torch.log(xywh_boxes[..., 2:] / xywh_priors[..., 2:]) / 0.2
                                   ], dim=- 1)
        return encode_target
