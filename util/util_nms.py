import torch
import numpy as np
from util.util_iou import iou_xywh


def nms2labels(keep, pre_score, pre_loc, pre_class):
    labels_out = []
    for keep_idx in keep:
        box = pre_loc[keep_idx]
        box = box.squeeze()
        boxx1 = max(0, box[0] - box[2]/2)
        boxy1 = max(0, box[1] - box[3]/2)
        boxx2 = box[0] + box[2]/2
        boxy2 = box[1] + box[3]/2
        box_out = [boxx1, boxy1, boxx2, boxy2]
        if min(box_out) < 0:
            print('error!')
            continue
        pre_score_out = pre_score[keep_idx].item()
        class_out = pre_class[keep_idx].item()
        labels_out.append([pre_score_out, class_out, box_out])
    return labels_out


def NMS(pre_score_raw, pre_loc, score_thresh, iou_thresh=0.5):
    """
    Nms.

    :param pre_score: in the shape of [box_number, class_number]
    :param pre_loc: int the shape of [box_number, 4] 4 means [x1, y1, x2, y2]
    :param score_thresh:score_thresh
    :param iou_thresh:iou_thresh
    :return: labels_out
    """
    # print('using Greedy NMS')

    class_num = pre_score_raw.shape[1]     # get the numbers of classes.
    pre_score_raw = pre_score_raw.max(-1)  # get the max score of the scores of classes.
    pre_score = pre_score_raw[0]           # score out
    pre_class = pre_score_raw[1]           # idx of score: is class

    score_sort = pre_score.sort(descending=True)   # sort the scores.

    score_idx = score_sort[1][score_sort[0] > score_thresh]  # find the scores>0.7(thresh)

    keep = []
    for i in range(class_num):             # with different classess.   
        a = pre_class[score_idx] == i      # each class for NMS.
        order_index = score_idx[a]         # get the index of orders
        while order_index.shape[0] > 0:    # deal with all the boxes.
            max_one = order_index[0].item()      # get index of the max score box.
            box_head = pre_loc[max_one]           # get the score of it
            box_others = pre_loc[order_index[1:]]      # the rest boxes.
            ious = iou_xywh(box_head, box_others)      # count the ious between the max one and the others
            rest = torch.lt(ious, iou_thresh).squeeze()  # find the boxes of iou<0.5(thresh), discard the iou>0.5.
            order_index = order_index[1:][rest]    # get the new index of the rest boxes, except the max one.
            keep.append(max_one)

    # ###########  NMS UP  ####################

    labels_out = nms2labels(keep, pre_score, pre_loc, pre_class)

    return labels_out


def Soft_NMS(pre_score_raw, pre_loc, score_thresh, theta=0.5):
    """
       Nms.

       :param pre_score: in the shape of [box_number, class_number]
       :param pre_loc: int the shape of [box_number, 4] 4 means [x1, y1, x2, y2]
       :param score_thresh:score_thresh
       :param iou_thresh:iou_thresh
       :return: labels_out
       """
    # print('using Soft NMS')

    class_num = pre_score_raw.shape[1]
    pre_score_raw = pre_score_raw.max(-1)
    pre_score = pre_score_raw[0]
    pre_class = pre_score_raw[1]

    score_sort = pre_score.sort(descending=True)
    score_idx = score_sort[1][score_sort[0] > score_thresh]

    keep = []
    for i in range(class_num):  # with different classess.
        a = pre_class[score_idx] == i
        order_index = score_idx[a]
        while order_index.shape[0] > 0:
            max_one = order_index[0].item()
            keep.append(max_one)
            box_head = pre_loc[max_one]
            box_others = pre_loc[order_index[1:]]
            score_others = pre_score[order_index[1:]]
            # print(score_others)
            ious = iou_xywh(box_head, box_others).reshape(1, -1)
            # print('iou', ious)
            soft_score = score_others*torch.exp(-pow(ious, 2) / theta).squeeze()  # s = s*e^(-iou^2 / theta)
            # print('s',soft_score)
            rest = torch.gt(soft_score, score_thresh)
            order_index = order_index[1:][rest]
            new_index = pre_score[order_index].sort(descending=True)
            order_index = order_index[new_index[1]]

    # ###########  NMS UP  ####################

    labels_out = nms2labels(keep, pre_score, pre_loc, pre_class)

    return labels_out


def NMS_from_FAST_RCNN(pre_score_raw, pre_loc, score_thresh, iou_thresh):  # NMS changed from FAST RCNN
    """
    Nms.

    :param pre_score: in the shape of [box_number, class_number]
    :param pre_loc: int the shape of [box_number, 4] 4 means [x1, y1, x2, y2]
    :param score_thresh:score_thresh
    :param iou_thresh:iou_thresh
    :return: labels_out
    """
    # print('using NMS_from_FAST_RCNN')
    pre_score_raw = pre_score_raw.max(-1)
    scores = pre_score_raw[0]
    pre_class = pre_score_raw[1]

    x1 = pre_loc[:, 0]
    y1 = pre_loc[:, 1]
    x2 = pre_loc[:, 2]
    y2 = pre_loc[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.sort(descending=True)[1]
    keep = []
    while order.shape[0] > 0:
        i = order[0]
        keep.append(i)
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        w = torch.max(0.0, xx2 - xx1 + 1)
        h = torch.max(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def NMS_old_edition(pre_score_raw, pre_loc, score_thresh, iou_thresh):
    """
    Nms.

    :param pre_score: in the shape of [box_number, class_number]
    :param pre_loc: int the shape of [box_number, 4] 4 means [x1, y1, x2, y2]
    :param score_thresh:score_thresh
    :param iou_thresh:iou_thresh
    :return: labels_out
    """
    pre_score_raw = pre_score_raw.max(-1)
    pre_score = pre_score_raw[0]
    pre_class = pre_score_raw[1]
    idx = pre_score > score_thresh  # Return the index of the max pre_score
    print('max score:', torch.max(pre_score).item())
    pre_score_max = pre_score[idx]
    # print(pre_score[torch.gt(pre_score, pre_score_thre)])
    print('Num of box:', len(pre_score_max))
    if len(pre_score_max) > 1:
        sor = pre_score_max.sort(descending=True)[1]
        _idx = idx.nonzero()[sor]
        _leng = len(_idx)

        for i in range(_leng):
            for j in range(i + 1, _leng):
                if pre_class[_idx[i]] != pre_class[_idx[j]]:  # diffent classes
                    continue
                if pre_score[_idx[i]] < score_thresh or\
                        pre_score[_idx[j]] < score_thresh:
                    continue  # get out of the likely anchors which have been counted
                box1 = pre_loc[_idx[i]].unsqueeze(0)
                box2 = pre_loc[_idx[j]].unsqueeze(0)
                iou_ = iou_xywh(box1, box2)
                if iou_ > iou_thresh:
                    pre_score[_idx[j]] = 0.0
    _idx = (pre_score > score_thresh).nonzero()  # Return the index of the max pre_score
    labels_out = []
    for keep_idx in _idx:
        box = pre_loc[keep_idx]
        box = box.squeeze()
        boxx1 = box[0] - box[2]/2
        boxy1 = box[1] - box[3]/2
        boxx2 = box[0] + box[2]/2
        boxy2 = box[1] + box[3]/2
        box_out = [boxx1, boxy1, boxx2, boxy2]
        if min(box_out) <= 0:
            print('error!')
            continue
        pre_score_out = pre_score[keep_idx].item()
        class_out = pre_class[keep_idx].item()
        labels_out.append([pre_score_out, class_out, box_out])
    return labels_out

