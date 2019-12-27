"""Parse the predictions."""
import torch
import numpy as np
from util.util_nms import NMS
import logging
from net_works.loss.ObdLoss_RefineDet import Detect_RefineDet
from util.util_iou import xywh2xyxy, xyxy2xywh

LOGGER = logging.getLogger(__name__)


class ParsePredict:

    def __init__(self, cfg):
        self.cfg = cfg
        self.anchors = torch.Tensor(cfg.TRAIN.ANCHORS)
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = len(cfg.TRAIN.CLASSES)
        self.NMS = NMS(cfg)

    def _parse_predict(self, f_maps):
        PARSEDICT = {
            'yolov2': self._parse_yolo_predict,
            'yolov3': self._parse_yolo_predict,
            'yolov3_tiny': self._parse_yolo_predict,
            'yolov3_tiny_mobilenet': self._parse_yolo_predict,
            'yolov3_tiny_squeezenet': self._parse_yolo_predict,
            'yolov3_tiny_shufflenet': self._parse_yolo_predict,
            'fcos': self._parse_fcos_predict,
            'refinedet': self._parse_refinedet_predict,
            'efficientdet': self._parse_efficientdet_predict,
        }
        labels_predict = PARSEDICT[self.cfg.TRAIN.MODEL](f_maps)
        return labels_predict

    def _predict2nms(self, pre_cls_score, pre_loc, xywh2x1y1x2y2=True):
        labels_predict = []
        for batch_n in range(pre_cls_score.shape[0]):
            # TODO: make a matrix instead of for...
            LOGGER.info('[NMS]')
            score = pre_cls_score[batch_n]
            loc = pre_loc[batch_n]
            labels = self.NMS.forward(score, loc, xywh2x1y1x2y2)
            labels_predict.append(labels)
        return labels_predict

    def _parse_yolo_predict_fmap(self, f_map, f_id, tolabel=False):
        # pylint: disable=no-self-use
        """
        Reshape the predict, or reshape it to label shape.

        :param predict: out of net
        :param tolabel: True or False to label shape
        :return:
        """
        # make the feature map anchors
        mask = np.arange(self.anc_num) + self.anc_num * f_id
        anchors = self.anchors[mask]

        ###1: pre deal feature mps
        obj_pred, cls_perd, loc_pred = torch.split(f_map, [self.anc_num, self.anc_num * self.cls_num, self.anc_num * 4], 3)
        pre_obj = obj_pred.sigmoid()
        cls_reshape = torch.reshape(cls_perd, (-1, self.cls_num))
        cls_pred_prob = torch.softmax(cls_reshape, -1)
        pre_cls = torch.reshape(cls_pred_prob, cls_perd.shape)
        pre_loc = loc_pred

        batch_size = pre_obj.shape[0]
        shape = pre_obj.shape[1:3]

        pre_obj = pre_obj.unsqueeze(-1)
        pre_cls = pre_cls.reshape([batch_size, shape[0], shape[1], self.anc_num, self.cls_num])

        # reshape the pre_loc
        pre_loc = pre_loc.reshape([batch_size, shape[0], shape[1], self.anc_num, 4])
        pre_loc_xy = pre_loc[..., 0:2].sigmoid()
        pre_loc_wh = pre_loc[..., 2:4]

        grid_x = torch.arange(0, shape[1]).view(-1, 1).repeat(1, shape[0]).unsqueeze(2).permute(1, 0, 2)
        grid_y = torch.arange(0, shape[0]).view(-1, 1).repeat(1, shape[1]).unsqueeze(2)
        grid_xy = torch.cat([grid_x, grid_y], 2).unsqueeze(2).unsqueeze(0). \
            expand(1, shape[0], shape[1], self.anc_num, 2).expand_as(pre_loc_xy).type(torch.cuda.FloatTensor)

        # prepare gird xy
        box_ch = torch.Tensor([shape[1], shape[0]]).cuda()
        pre_realtive_xy = (pre_loc_xy + grid_xy) / box_ch
        anchor_ch = anchors.view(1, 1, 1, self.anc_num, 2).expand(1, shape[0], shape[1], self.anc_num, 2).cuda()
        pre_realtive_wh = pre_loc_wh.exp() * anchor_ch

        pre_relative_box = torch.cat([pre_realtive_xy, pre_realtive_wh], -1)

        if tolabel:
            return pre_obj, pre_cls, pre_relative_box
        return pre_obj, pre_cls, pre_relative_box, pre_loc_xy, pre_loc_wh, grid_xy, shape

    def _parse_yolo_predict(self, f_maps):
        """
        Parse the predict. with all feature maps to labels.
    
        :param f_maps: predictions out of net.
        :return:parsed predictions, to labels.
        """
        pre_obj = []
        pre_cls = []
        pre_loc = []
        for f_id, f_map in enumerate(f_maps):
            # Parse one feature map.
            _pre_obj, _pre_cls, _pre_relative_box = self._parse_yolo_predict_fmap(f_map, f_id=f_id, tolabel=True)
            BN = _pre_obj.shape[0]
            _pre_obj = _pre_obj.reshape(BN, -1, _pre_obj.shape[-1])
            _pre_cls = _pre_cls.reshape(BN, -1, _pre_cls.shape[-1])
            _pre_loc = _pre_relative_box.reshape(BN, -1, _pre_relative_box.shape[-1])

            pre_obj.append(_pre_obj)
            pre_cls.append(_pre_cls)
            pre_loc.append(_pre_loc)

        pre_obj = torch.cat(pre_obj, -2)
        pre_cls = torch.cat(pre_cls, -2)
        pre_loc = torch.cat(pre_loc, -2)

        pre_score = pre_cls * (pre_obj.expand_as(pre_cls))  # score of obj * score of class.
        labels_predict = self._predict2nms(pre_score, pre_loc)
        return labels_predict

    def _parse_fcos_predict(self, predicts):

        def parse_one_feature(predict, feature_idx):
            stride = self.cfg.TRAIN.STRIDES[feature_idx]
            pre_cls, pre_ctness, pre_loc = predict
            pre_cls, pre_ctness, pre_loc = pre_cls.permute(0, 2, 3, 1), \
                                           pre_ctness.permute(0, 2, 3, 1), \
                                           pre_loc.permute(0, 2, 3, 1)
            # softmax the classes
            clsshape = pre_cls.shape
            N = clsshape[0]
            feature_size = [clsshape[1], clsshape[2]]

            pre_cls = pre_cls.sigmoid()
            # pre_cls = torch.reshape(pre_cls, (-1, 4)).softmax(-1)
            # pre_cls = torch.reshape(pre_cls, clsshape)

            pre_ctness = pre_ctness.sigmoid()
            if feature_idx == 2:
                i_0, i_1, i_2 = 0, 13, 13
                print('pre_loc[i_0, i_1, i_2]:', pre_loc[i_0, i_1, i_2])
                print('pre_cls[i_0, i_1, i_2]', pre_cls[i_0, i_1, i_2])
                print('pre_ctness[i_0, i_1, i_2]', pre_ctness[i_0, i_1, i_2])

            # print('pre_loc:', pre_loc[0, 13, 19])
            pre_loc = torch.exp(pre_loc)
            pre_cls_conf = pre_cls * pre_ctness

            # TODO:change the for ...to martrix
            # make a grid_xy
            pre_loc_xy = pre_loc[..., :2]
            grid_x = torch.arange(0, feature_size[1]).view(-1, 1).repeat(1, feature_size[0]) \
                .unsqueeze(2).permute(1, 0, 2)
            grid_y = torch.arange(0, feature_size[0]).view(-1, 1).repeat(1, feature_size[1]).unsqueeze(2)
            grid_xy = torch.cat([grid_x, grid_y], 2).unsqueeze(0).expand_as(pre_loc_xy).type(torch.cuda.FloatTensor)
            # print('grid_xy:', grid_xy[0, 1, 19])
            grid_xy = grid_xy * stride + stride / 2
            # print('grid_xy:', grid_xy[0, 1, 19])
            # print('pre_loc:', pre_loc_xy[0, 1, 19])
            x1 = grid_xy[..., 0] - pre_loc[..., 0]
            y1 = grid_xy[..., 1] - pre_loc[..., 1]
            w = pre_loc[..., 0] + pre_loc[..., 2]
            h = pre_loc[..., 1] + pre_loc[..., 3]
            x = x1 + w / 2
            y = y1 + h / 2
            if self.cfg.TRAIN.RELATIVE_LABELS:
                x /= self.cfg.TRAIN.IMG_SIZE[1]
                y /= self.cfg.TRAIN.IMG_SIZE[0]
                w /= self.cfg.TRAIN.IMG_SIZE[1]
                h /= self.cfg.TRAIN.IMG_SIZE[0]
            predicted_boxes = torch.stack([x, y, w, h], -1)

            return pre_cls_conf, predicted_boxes

        scores = []
        locs = []
        for feature_idx, predict in enumerate(predicts):
            pre_cls_conf, predicted_boxes = parse_one_feature(predict, feature_idx)
            if feature_idx == 2:
                i_0, i_1, i_2 = 0, 13, 13
                print('predicted_boxes[i_0, i_1, i_2]:', predicted_boxes[i_0, i_1, i_2])
                print('pre_cls_conf[i_0, i_1, i_2]', pre_cls_conf[i_0, i_1, i_2])

            batch_n = pre_cls_conf.shape[0]
            score = torch.reshape(pre_cls_conf, (batch_n, -1, pre_cls_conf.shape[-1]))
            loc = torch.reshape(predicted_boxes, (batch_n, -1, 4))
            scores.append(score)
            locs.append(loc)

        scores = torch.cat(scores, 1)
        locs = torch.cat(locs, 1)

        labels_predict = self._predict2nms(scores, locs)

        return labels_predict

    def _parse_refinedet_predict(self, predicts):
        detecte = Detect_RefineDet(self.cfg)
        pre_score, pre_loc = detecte.forward(predicts)

        wh = pre_loc[..., 2:] - pre_loc[..., :2]
        xy = pre_loc[..., :2] + wh / 2
        _pre_loc = torch.cat([xy, wh], 2)

        labels_predict = self._predict2nms(pre_score, _pre_loc)

        return labels_predict

    def _parse_efficientdet_predict(self, predicts):
        pre_score, pre_loc_xyxy = predicts
        pre_loc_xywh = xyxy2xywh(pre_loc_xyxy)
        labels_predict = self._predict2nms(pre_score, pre_loc_xywh)
        return labels_predict
