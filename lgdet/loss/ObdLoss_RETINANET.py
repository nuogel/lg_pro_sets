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
        self.neg_pos_ratio = 3
        self.neg_iou_threshold = 0.5
        self.focalloss = FocalLoss()
        self.becloss = torch.nn.BCELoss(reduction='sum')
        self.use_hard_mining = 0 # use BCEloss

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
        if self.use_hard_mining:
            with torch.no_grad():
                loss = -F.log_softmax(pre_cls, dim=-1)[..., -1]  # -1 reprsent the background
                mask = self._hard_negative_mining(loss, pos_idx, self.neg_pos_ratio)
            cls_loss = self.becloss(pre_cls[mask], gt_cls[mask]) / pos_num

        else:
            cls_loss = self.focalloss(pre_cls, gt_cls, logist=False, reduction='sum') / pos_num

        pre_loc = pre_loc[pos_idx, :].reshape(-1, 4)
        gt_loc = gt_loc[pos_idx, :].reshape(-1, 4)
        loc_loss = F.smooth_l1_loss(pre_loc, gt_loc, reduction='sum') / pos_num

        total_loss = (loc_loss + cls_loss)

        # metrics:
        cls_p = (pre_cls[pos_idx][..., :-1].argmax(-1) == gt_cls[pos_idx][..., :-1].argmax(-1)).float().mean().item()
        obj_sc = pre_cls[pos_idx][..., :-1].max(-1)[0].mean().item()
        obj_t = (pre_cls[pos_idx][..., :-1].max(-1)[0] > self.cfg.TEST.SCORE_THRESH).float().mean().item()
        nob_sc = pre_cls[~pos_idx][..., :-1].max(-1)[0].mean().item()
        nob_t = (pre_cls[~pos_idx][..., :-1].max(-1)[0] > self.cfg.TEST.SCORE_THRESH).float().sum().item()

        metrics = {
            'obj_sc': obj_sc,
            'nob_sc': nob_sc,
            'obj>t': obj_t,
            'nob>t': nob_t,
            'cls_p': cls_p,

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

        gt_cls = torch.zeros_like(pre_cls)
        pos_idx = torch.ge(best_target_per_prior, 0.5)
        neg_idx = ~pos_idx
        gt_cls[neg_idx, -1] = 1  # background set to one hot [0, ..., 1]
        assigned_annotations = gt_labels[best_target_per_prior_index]

        # gt_cls[pos_idx, :] = 0
        gt_cls[pos_idx, assigned_annotations[pos_idx].long()] = 1
        gt_loc = gt_boxes[best_target_per_prior_index]
        return gt_cls, gt_loc, pos_idx

    def _encode_bbox(self, xywh_boxes, xywh_priors):
        encode_target = torch.cat([(xywh_boxes[..., :2] - xywh_priors[..., :2]) / xywh_priors[..., 2:] / 0.1,
                                   torch.log(xywh_boxes[..., 2:] / xywh_priors[..., 2:]) / 0.2
                                   ], dim=- 1)
        return encode_target

    def _hard_negative_mining(self, loss, pos_idx, neg_pos_ratio):
        num_pos = pos_idx.long().sum(dim=1, keepdim=True)
        num_neg = num_pos * neg_pos_ratio
        loss[pos_idx] = 0.
        _, indexes = loss.sort(dim=1, descending=True)
        _, orders = indexes.sort(dim=1)
        neg_mask = orders < num_neg
        return pos_idx | neg_mask
