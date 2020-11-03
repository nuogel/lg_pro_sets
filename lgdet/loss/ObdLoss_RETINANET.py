import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from lgdet.util.util_iou import iou_xyxy, xywh2xyxy, xyxy2xywh, iou_xywh
import numpy as np
from lgdet.loss.loss_base.focal_loss import FocalLoss_lg, FocalLoss


class RETINANETLOSS():
    def __init__(self, cfg):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """

        self.cfg = cfg
        self.device = cfg.TRAIN.DEVICE
        self.num_cls = self.cfg.TRAIN.CLASSES_NUM
        self.neg_iou_threshold = 0.5
        self.focalloss = FocalLoss(reduction='none')

    def Loss_Call(self, predictions, targets, kwargs):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        pre_cls, pre_loc, anc_xywh = predictions
        pre_cls = pre_cls.sigmoid()
        batchsize = pre_loc.shape[0]
        _, gt_labels, infos = targets

        gt_loc, gt_cls, pos_idx = [], [], []
        for i in range(batchsize):
            gt_i = gt_labels[gt_labels[..., 0] == i]
            lab = gt_i[..., 1].long()
            box = gt_i[..., 2:]
            box_xywh = xyxy2xywh(box)
            _gt_cls, _gt_loc_xywh, _pos_idx = self._assign_priors(pre_cls[i], box_xywh, lab, anc_xywh)
            _gt_loc_xywh = self._encode_bbox(_gt_loc_xywh, anc_xywh)
            gt_loc.append(_gt_loc_xywh)
            gt_cls.append(_gt_cls)
            pos_idx.append(_pos_idx)

        gt_loc = torch.stack(gt_loc, dim=0)
        gt_cls = torch.stack(gt_cls, dim=0)
        pos_idx = torch.stack(pos_idx, dim=0)

        pos_num = pos_idx.sum()
        ignore_mask = gt_cls == -1

        # cls_loss = self.focalloss(pre_cls, gt_cls, logist=True)
        alpha_factor = torch.ones(gt_cls.shape).to(self.device) * 0.25
        alpha_factor = torch.where(torch.eq(gt_cls, 1.), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(gt_cls, 1.), 1. - pre_cls, pre_cls)
        focal_weight = alpha_factor * torch.pow(focal_weight, 2)
        bce = F.binary_cross_entropy(pre_cls, gt_cls, reduction='none')
        cls_loss = focal_weight * bce
        cls_loss = cls_loss[~ignore_mask].sum()/pos_num

        pre_loc = pre_loc[pos_idx, :].reshape(-1, 4)
        gt_loc = gt_loc[pos_idx, :].reshape(-1, 4)

        loc_loss = F.smooth_l1_loss(pre_loc, gt_loc)

        # num_pos = gt_loc.size(0)
        total_loss = loc_loss + cls_loss

        # metrics:
        cls_p = (pre_cls[pos_idx].argmax(-1) == gt_cls[pos_idx].argmax(-1)).float().mean().item()
        obj_sc = pre_cls[pos_idx].max(-1)[0].mean().item()
        obj_t = (pre_cls[pos_idx].max(-1)[0] > self.cfg.TEST.SCORE_THRESH).float().mean().item()
        nob_sc = pre_cls[~pos_idx].max(-1)[0].mean().item()
        nob_t = (pre_cls[~pos_idx].max(-1)[0] > self.cfg.TEST.SCORE_THRESH).float().mean().item()

        metrics = {
            'cls_p': cls_p,
            'obj_sc': obj_sc,
            'obj>t': obj_t,
            'nob_sc': nob_sc,
            'nob>t': nob_t,

            'cls_loss': cls_loss.item(),
            'loc_loss': loc_loss.item(),
        }
        return {'total_loss': total_loss, 'metrics': metrics}

    def _assign_priors(self, pre_cls, gt_boxes, gt_labels, corner_form_priors):
        ious = iou_xywh(corner_form_priors, gt_boxes, type='N2N')
        best_target_per_prior, best_target_per_prior_index = ious.max(1)

        best_prior_per_target, best_prior_per_target_index = ious.max(0)
        # #
        for target_index, prior_index in enumerate(best_prior_per_target_index):
            best_target_per_prior_index[prior_index] = target_index
        # 2.0 is used to make sure every target has a prior assigned
        best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)  # fill 2>1.
        # size: num_priors

        gt_cls = torch.ones_like(pre_cls) * -1
        gt_cls[torch.lt(best_target_per_prior, 0.4), :] = 0
        pos_idx = torch.ge(best_target_per_prior, 0.5)
        assigned_annotations = gt_labels[best_target_per_prior_index]

        gt_cls[pos_idx, :] = 0
        gt_cls[pos_idx, assigned_annotations[pos_idx].long()] = 1
        gt_loc = gt_boxes[best_target_per_prior_index]
        return gt_cls, gt_loc, pos_idx

    def _encode_bbox(self, xywh_boxes, xywh_priors):
        encode_target = torch.cat([(xywh_boxes[..., :2] - xywh_priors[..., :2]) / xywh_priors[..., 2:] / 0.1,
                                   torch.log(xywh_boxes[..., 2:] / xywh_priors[..., 2:]) / 0.2
                                   ], dim=- 1)
        return encode_target
