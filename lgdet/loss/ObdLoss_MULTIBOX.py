import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from util.util_iou import iou_xyxy, xywh2xyxy, xyxy2xywh, iou_xywh
import numpy as np


class MULTIBOXLOSS():
    def __init__(self, cfg):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        self.cfg = cfg
        self.num_cls = self.cfg.TRAIN.CLASSES_NUM
        self.neg_iou_threshold = 0.5
        self.neg_pos_ratio = 3  # 3:1

    def Loss_Call(self, predictions, targets, kwargs):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        confidence, predicted_locations, anchors_xywh = predictions
        batchsize = predicted_locations.shape[0]
        gt_images, gt_labels, infos = targets

        encode_target, labels = [], []
        for i in range(batchsize):
            gt_i = gt_labels[gt_labels[..., 0] == i]
            lab = gt_i[..., 1].long()
            box = gt_i[..., 2:]
            box_xywh = xyxy2xywh(box)
            _gt_loc_xywh, _labels = self._assign_priors(box_xywh, lab, anchors_xywh, self.neg_iou_threshold)
            _gt_loc_xywh = self._encode_bbox(_gt_loc_xywh, anchors_xywh)
            encode_target.append(_gt_loc_xywh)
            labels.append(_labels)
        encode_target = torch.stack(encode_target, dim=0)
        labels = torch.stack(labels, dim=0)
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=-1)[:, :, -1]  # -1 reprsent the background
            mask = self._hard_negative_mining(loss, labels, self.neg_pos_ratio)

        input_c = confidence[mask].reshape(-1, num_classes)
        target_c = labels[mask]
        classification_loss = F.cross_entropy(input_c, target_c)
        pos_mask = labels < self.num_cls
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        encode_target = encode_target[pos_mask, :].reshape(-1, 4)

        loc_loss = F.smooth_l1_loss(predicted_locations, encode_target)

        # num_pos = encode_target.size(0)
        loc_loss = loc_loss
        class_loss = classification_loss
        total_loss = loc_loss + class_loss

        metrics = {'loc_loss': loc_loss.item(),
                   'cls_loss': class_loss.item()}
        return {'total_loss': total_loss, 'metrics': metrics}

    def _hard_negative_mining(self, loss, labels, neg_pos_ratio):
        """
        It used to suppress the presence of a large number of negative prediction.
        It works on image level not batch level.
        For any example/image, it keeps all the positive predictions and
         cut the number of negative predictions to make sure the ratio
         between the negative examples and positive examples is no more
         the given ratio for an image.

        Args:
            loss (N, num_priors): the loss for each example.
            labels (N, num_priors): the labels.
            neg_pos_ratio:  the ratio between the negative examples and positive examples.
        """
        pos_mask = labels < self.num_cls
        num_pos = pos_mask.long().sum(dim=1, keepdim=True)
        num_neg = num_pos * neg_pos_ratio

        loss[pos_mask] = 0.
        _, indexes = loss.sort(dim=1, descending=True)
        _, orders = indexes.sort(dim=1)
        neg_mask = orders < num_neg
        return pos_mask | neg_mask

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
