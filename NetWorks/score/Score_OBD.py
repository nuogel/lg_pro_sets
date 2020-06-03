"""Calculate the F1 score."""
import os

import numpy as np
import torch

from util.util_get_cls_names import _get_class_names
from util.util_iou import iou_xyxy


class Score:
    """Calculate the F1 score."""

    def __init__(self, cfg):
        """Init the parameters."""
        self.cfg = cfg
        self.cls_name = _get_class_names(cfg.PATH.CLASSES_PATH)
        self.cls_num = len(self.cfg.TRAIN.CLASSES)
        self.true_positive = {}
        self.false_positive = {}
        self.pre_scores = {}
        try:
            from util.util_parse_prediction import ParsePredict
        except:
            print("no ParsePredict")
        else:
            self.parsepredict = ParsePredict(self.cfg)

    def init_parameters(self):
        for i in range(self.cls_num):
            self.true_positive[i] = []
            self.false_positive[i] = []
            self.pre_scores[i] = []
        self.gt_obj_num = np.zeros(self.cls_num)

    def cal_score(self, pre_labels, gt_labels, from_net=True):  # did not merge the self.get_labels_txt, because from net,don't need that function.
        """
        Calculate the TP & FP.

        :param pre_labels: prediction labels
        :param gt_labels: ground truth labels
        :param from_net : pre_labels from net work
        :return: TP FP object numbers
        """
        if from_net:
            pre_labels = self.parsepredict._parse_predict(pre_labels)
        if pre_labels != [[]]:
            for i, pre_label in enumerate(pre_labels):  # calculate every image
                self._cal_per_image(pre_label, gt_labels[gt_labels[..., 0] == i], self.cfg.TEST.IOU_THRESH)
        else:
            print('predicted labels is None.')

    def _cal_per_image(self, pre_label, gt_label, iou_thresh=0.5):
        """
        Calculate TP FP for one image.

        :param pre_label: prediction labels of one image
        :param gt_label: ground truth labels of one image
        :param iou_thresh: the threshold of iou
        :return: P FP object numbers of one image
        """
        label_used = np.zeros(len(gt_label))

        for cls in range(self.cls_num):
            for one_pre_box in pre_label:
                if one_pre_box[1] != cls:
                    continue
                max_iou = -1
                _id = None
                for i, lab in enumerate(gt_label):
                    lab = lab[1:]
                    cls_gt = int(lab[0].cpu())
                    if cls_gt != cls:
                        continue
                    pre_box = torch.Tensor(one_pre_box[2]).unsqueeze(0).to(self.cfg.TRAIN.DEVICE)
                    gt_box = lab[1:5].unsqueeze(0)
                    iou = iou_xyxy(pre_box, gt_box)[0][0]  # to(anchors.device))
                    if iou > max_iou:
                        max_iou = iou
                        _id = i
                if max_iou > iou_thresh and label_used[_id] == 0:
                    self.true_positive[cls].append(1)
                    self.false_positive[cls].append(0)
                    label_used[_id] = 1
                else:
                    self.true_positive[cls].append(0)
                    self.false_positive[cls].append(1)
                self.pre_scores[cls].append(one_pre_box[0])
        for lab in gt_label:
            self.gt_obj_num[int(lab[1])] += 1

    def score_out(self):
        if self.cfg.TEST.SCORE_TYPE.lower() == 'map':
            score, prec, rec = self.mAP_SCORE()
            a, b, c = self.F1_SCORE()
            print('F1SCORE: ', a)
        else:
            score, prec, rec = self.F1_SCORE()

        score_dict = dict(zip(self.cfg.TRAIN.CLASSES, score))
        prec_dict = dict(zip(self.cfg.TRAIN.CLASSES, prec))
        rec_dict = dict(zip(self.cfg.TRAIN.CLASSES, rec))
        print('{}: {}\nprec: {}\nrec: {}'.format(self.cfg.TEST.SCORE_TYPE.lower(), score, prec, rec))
        return score_dict, prec_dict, rec_dict

    def F1_SCORE(self, beta=1):
        """
        Calculate the F1 score.

        :param beta: beta
        :return: F1 score
        """
        f1_sore = -1 * np.ones(self.cls_num, dtype=np.float32)
        prec = -1 * np.ones(self.cls_num, dtype=np.float32)
        rec = -1 * np.ones(self.cls_num, dtype=np.float32)
        for i in range(self.cls_num):
            if self.gt_obj_num[i] == 0:
                # TODO: BUG, when there is a FP about this class. how ?, But ,gt_obj_num is hardly tobe Zero.
                continue
            tp = np.sum(np.asarray(self.true_positive[i]))
            fp = np.sum(np.asarray(self.false_positive[i]))

            if tp == 0 and fp == 0:
                prec[i] = 0
                rec[i] = 0
                f1_sore[i] = 0
                continue
            prec[i] = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            rec[i] = tp / self.gt_obj_num[i]
            f1_sore[i] = (1 + beta ** 2) * (prec[i] * rec[i]) / (beta ** 2 * prec[i] + rec[i])
        return f1_sore, prec, rec

    def mAP_SCORE(self):
        """Compute VOC AP given precision and recall. If use_07_metric is true, uses
        the VOC 07 11-point method (default:False).
        """

        def count_ap(rec, prec, use_07_metric=False):
            if use_07_metric:
                # 11 point metric
                ap = 0.
                for t in np.arange(0., 1.1, 0.1):
                    if np.sum(rec >= t) == 0:
                        p = 0
                    else:
                        p = np.max(prec[rec >= t])
                    ap = ap + p / 11.
            else:
                # correct AP calculation
                # first append sentinel values at the end
                mrec = np.concatenate(([0.], rec, [1.]))
                mpre = np.concatenate(([0.], prec, [0.]))

                # compute the precision envelope
                for i in range(mpre.size - 1, 0, -1):
                    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

                # to calculate area under PR curve, look for points
                # where X axis (recall) changes value
                i = np.where(mrec[1:] != mrec[:-1])[0]

                # and sum (\Delta recall) * prec
                ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            return ap

        ap = -1 * np.ones(self.cls_num)
        precision = -1 * np.ones(self.cls_num, dtype=np.float32)
        recall = -1 * np.ones(self.cls_num, dtype=np.float32)
        for cls_i in range(self.cls_num):
            if self.pre_scores[cls_i] == []: continue
            pre_scores = np.asarray(self.pre_scores[cls_i])
            fp = np.asarray(self.false_positive[cls_i])
            tp = np.asarray(self.true_positive[cls_i])

            sorted_ind = np.argsort(-pre_scores)
            pre_scores = pre_scores[sorted_ind]
            fp = fp[sorted_ind]
            tp = tp[sorted_ind]

            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(self.gt_obj_num[cls_i])
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap[cls_i] = count_ap(rec, prec)
            precision[cls_i] = prec[-1]
            recall[cls_i] = rec[-1]
        return ap, precision, recall
