import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from util.util_iou import iou_xyxy, xywh2xyxy, xyxy2xywh
from util.util_anchor_maker import Anchors


class MultiboxLoss():
    def __init__(self, cfg):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        self.iou_threshold = cfg.TEST.IOU_THRESH
        self.neg_pos_ratio = 3
        # self.center_variance = center_variance
        # self.size_variance = size_variance
        # self.priors = priors
        # self.priors.to(device)

    def Loss_Call(self, predictions, targets, losstype=None):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        predicted_locations, confidence = predictions
        batchsize = predicted_locations.shape[0]
        gt_images, gt_labels = targets
        anchors = Anchors()(gt_images)[0]
        anchors = xywh2xyxy(anchors)
        gt_locations, labels = [], []
        for i in range(batchsize):
            lab = [box[0] for box in gt_labels[i]]
            box = [box[1:] for box in gt_labels[i]]
            lab = torch.LongTensor(lab).to(anchors.device)
            box = torch.Tensor(box).to(anchors.device)
            _gt_locations, _labels = self._assign_priors(box, lab, anchors, self.iou_threshold)
            _gt_locations = self._encode_bbox(xyxy2xywh(_gt_locations), xyxy2xywh(anchors))
            gt_locations.append(_gt_locations)
            labels.append(_labels)
        gt_locations = torch.stack(gt_locations, dim=0)
        labels = torch.stack(labels, dim=0)
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = self._hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)

        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos

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
        pos_mask = labels > 0
        num_pos = pos_mask.long().sum(dim=1, keepdim=True)
        num_neg = num_pos * neg_pos_ratio

        loss[pos_mask] = -math.inf
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
        ious = iou_xyxy(corner_form_priors, gt_boxes, type='N2N')
        # size: num_priors
        best_target_per_prior, best_target_per_prior_index = ious.max(1)
        # size: num_targets
        best_prior_per_target, best_prior_per_target_index = ious.max(0)

        for target_index, prior_index in enumerate(best_prior_per_target_index):
            best_target_per_prior_index[prior_index] = target_index
        # 2.0 is used to make sure every target has a prior assigned
        best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
        # size: num_priors
        labels = gt_labels[best_target_per_prior_index]
        labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
        boxes = gt_boxes[best_target_per_prior_index]
        return boxes, labels

    def _encode_bbox(self, center_form_boxes, center_form_priors):
        encode_target = torch.cat([
            (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / 0.1,
            torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / 0.2
        ], dim=- 1)
        return encode_target


class DecodeBBox(nn.Module):

    def __init__(self, mean=None, std=None):
        super(DecodeBBox, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, predicts, anchors):
        a_width = anchors[:, :, 2] - anchors[:, :, 0]
        a_height = anchors[:, :, 3] - anchors[:, :, 1]
        ctr_x = anchors[:, :, 0] + 0.5 * a_width
        ctr_y = anchors[:, :, 1] + 0.5 * a_height

        dx = predicts[:, :, 0] * self.std[0] + self.mean[0]
        dy = predicts[:, :, 1] * self.std[1] + self.mean[1]
        dw = predicts[:, :, 2] * self.std[2] + self.mean[2]
        dh = predicts[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * a_width
        pred_ctr_y = ctr_y + dy * a_height
        pred_w = torch.exp(dw) * a_width
        pred_h = torch.exp(dh) * a_height

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes
