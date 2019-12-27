"""Calculate the F1 score."""
import os
import torch
import numpy as np

from util.util_iou import iou_xyxy
from util.util_get_cls_names import _get_class_names


class F1Score:
    """Calculate the F1 score."""

    def __init__(self, cfg):
        """Init the parameters."""
        self.cfg = cfg
        self.cls_name = _get_class_names(cfg.PATH.CLASSES_PATH)
        self.cls_num = len(self.cfg.TRAIN.CLASSES)
        self.true_positive = np.zeros(self.cls_num)
        self.false_positive = np.zeros(self.cls_num)
        self.obj_num = np.zeros(self.cls_num)
        self.apolloclass2num = dict(zip(self.cfg.TRAIN.CLASSES, range(len(self.cfg.TRAIN.CLASSES))))
        try:
            from util.util_parse_prediction import ParsePredict
        except:
            print("no ParsePredict")
        else:
            self.parsepredict = ParsePredict(self.cfg)

    def init_parameters(self):
        self.true_positive = np.zeros(self.cls_num)
        self.false_positive = np.zeros(self.cls_num)
        self.obj_num = np.zeros(self.cls_num)

    def get_labels_txt(self, pre_path, gt_path):
        def read_line(path, gt=False):
            # TODO: read line of file.xml.
            f_path = open(path, 'r')
            label = []
            for line in f_path.readlines():
                tmp = line.split()
                if tmp[0] not in self.cls_name:
                    continue
                if self.cls_name[tmp[0]] == 'DontCare':
                    continue
                if not gt:
                    label.append([float(tmp[15]), self.apolloclass2num[self.cls_name[tmp[0]]],
                                  [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]])
                else:
                    label.append(
                        [self.apolloclass2num[self.cls_name[tmp[0]]], float(tmp[4]), float(tmp[5]), float(tmp[6]),
                         float(tmp[7])])
            return label

        pre_files = os.listdir(pre_path)
        pre_labels = []
        gt_labels = []
        for pre_file in pre_files:
            file_id = pre_file.split('.')[0]
            if os.path.isfile(os.path.join(gt_path, file_id + '.txt')):
                file_GT_path = gt_path + file_id + '.txt'
            elif os.path.isfile(os.path.join(gt_path, file_id + '.xml')):
                file_GT_path = gt_path + file_id + '.xml'
            else:
                print('not exist the GT file from <== prediction file : ', pre_file)
                continue
            pre_label = read_line(pre_path + pre_file, gt=False)
            gt_label = read_line(file_GT_path, gt=True)
            pre_labels.append(pre_label)
            gt_labels.append(gt_label)

        return pre_labels, gt_labels

    def cal_score(self, pre_labels, gt_labels,
                  from_net=True):  # did not merge the self.get_labels_txt, because from net,don't need that function.
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
                self._cal_per_image(pre_label, gt_labels[i])
        else:
            print('predicted labels is None.')

    def _cal_per_image(self, pre_label, label, iou_thresh=0.5):
        """
        Calculate TP FP for one image.

        :param pre_label: prediction labels of one image
        :param label: ground truth labels of one image
        :param iou_thresh: the threshold of iou
        :return: P FP object numbers of one image
        """
        label_used = np.zeros(len(label))

        for cls in range(self.cls_num):
            for one_box in pre_label:
                if one_box[1] is not cls:
                    continue
                max_iou = -1
                _id = None
                for i, lab in enumerate(label):
                    if lab[0] is not cls:
                        continue
                    a = torch.Tensor(one_box[2]).cuda().unsqueeze(0)
                    b = torch.Tensor(lab[1:5]).to(a.device).unsqueeze(0)
                    iou = iou_xyxy(a, b)[0][0]  # to(anchors.device))
                    if iou > max_iou:
                        max_iou = iou
                        _id = i
                if max_iou > iou_thresh and label_used[_id] == 0:
                    self.true_positive[cls] += 1
                    label_used[_id] = 1
                else:
                    self.false_positive[cls] += 1
        for lab in label:
            self.obj_num[int(lab[0])] += 1

    def score_out(self, beta=1):
        """
        Calculate the F1 score.

        :param beta: beta
        :return: F1 score
        """
        print('Calculating The F1 Score...')
        f1_sore = -1 * np.ones(self.cls_num, dtype=np.float32)
        prec = -1 * np.ones(self.cls_num, dtype=np.float32)
        rec = -1 * np.ones(self.cls_num, dtype=np.float32)
        for i in range(self.cls_num):
            if self.obj_num[i] == 0:
                # TODO: BUG, when there is a FP about this class. how ?, But ,obj_num is hardly tobe Zero.
                continue
            if self.true_positive[i] == 0 and self.false_positive[i] == 0:
                prec[i] = 0
                rec[i] = 0
                f1_sore[i] = 0
                continue
            prec[i] = self.true_positive[i] / (self.true_positive[i] + self.false_positive[i])
            rec[i] = self.true_positive[i] / self.obj_num[i]
            f1_sore[i] = (1 + beta ** 2) * (prec[i] * rec[i]) / (beta ** 2 * prec[i] + rec[i])

        # matrix = np.stack(f1_sore, prec, rec)
        score_dict = dict(zip(self.cfg.TRAIN.CLASSES, f1_sore))
        prec_dict = dict(zip(self.cfg.TRAIN.CLASSES, prec))
        rec_dict = dict(zip(self.cfg.TRAIN.CLASSES, rec))

        print('f1_sore: {}\nprec: {}\nrec: {}'.format(f1_sore, prec, rec))

        return score_dict, prec_dict, rec_dict


if __name__ == "__main__":
    f1sore = F1Score()
    pre_path = '../../tmp/generated_labels_lg/'
    gt_path = '../../datasets/goldenridge_testset/test_images_inlane/labels/'

    pre_labels, gt_labels = f1sore.get_labels_txt(pre_path, gt_path)
    f1sore.cal_score(pre_labels, gt_labels, from_net=False)
    f1sore.score_out()
