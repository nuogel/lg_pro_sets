"""Loss calculation based on yolo."""
import torch
import numpy as np
from util.util_iou import iou_xywh
from util.util_get_cls_names import _get_class_names
from util.util_parse_prediction import ParsePredict


class YoloLoss:
    # pylint: disable=too-few-public-methods
    """Calculate loss."""

    def __init__(self, cfg):
        """Init."""
        #
        self.cfg = cfg
        self.anchors = torch.Tensor(cfg.TRAIN.ANCHORS)
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = len(cfg.TRAIN.CLASSES)

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.apollo_cls2idx = dict(zip(cfg.TRAIN.CLASSES, range(len(cfg.TRAIN.CLASSES))))

        self.mseloss = torch.nn.MSELoss()
        self.bcelogistloss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.bceloss = torch.nn.BCELoss(reduction='none')
        self.class_name = _get_class_names(cfg.PATH.CLASSES_PATH)

        self.parsepredict = ParsePredict(cfg)
        self.multiply_area_scale = False  # whether multiply loss to area_scale.

    def _reshape_labels(self, pre_obj, labels, f_id):
        """
        Reshape the labels.

        :param labels: labels from training data
        :param grid_xy: the matrix of the grid numbers
        :return: labels_obj, labels_cls, lab_loc_xy, lab_loc_wh, labels_boxes, area_scal
        """
        B, H, W = pre_obj.shape[0:3]
        mask = np.arange(self.anc_num) + self.anc_num * f_id
        anchors = self.anchors[mask] / torch.Tensor([self.cfg.TRAIN.IMG_SIZE[1], self.cfg.TRAIN.IMG_SIZE[0]])  # * torch.Tensor([W, H])
        anchors = anchors.to(self.cfg.TRAIN.DEVICE)
        obj_mask = torch.ByteTensor(B, H, W, self.anc_num).fill_(0).to(self.cfg.TRAIN.DEVICE)
        noobj_mask = torch.ByteTensor(B, H, W, self.anc_num).fill_(1).to(self.cfg.TRAIN.DEVICE)
        labels_loc_xy = torch.zeros([B, H, W, self.anc_num, 2]).to(self.cfg.TRAIN.DEVICE)
        labels_loc_wh = torch.zeros([B, H, W, self.anc_num, 2]).to(self.cfg.TRAIN.DEVICE)
        labels_cls = torch.zeros([B, H, W, self.anc_num, self.cls_num]).to(self.cfg.TRAIN.DEVICE)

        target_boxes = labels[:, 2:6]
        # x1y1x2y2->xywh
        gx1y1 = target_boxes[:, :2]
        gx2y2 = target_boxes[:, 2:]
        gxy = (gx1y1 + gx2y2) / 2.0 * torch.Tensor([W, H]).to(self.cfg.TRAIN.DEVICE)
        gwh = gx2y2 - gx1y1

        box_iou = torch.cat([torch.zeros_like(gwh), gwh], 1)
        anc = torch.cat([torch.zeros_like(anchors), anchors], 1)

        ious = iou_xywh(anc.to(self.cfg.TRAIN.DEVICE), box_iou.to(self.cfg.TRAIN.DEVICE), type='N2N')
        best_ious, best_n = ious.max(0)
        # Separate target values
        b, target_labels = labels[:, :2].long().t()
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        gi, gj = gxy.long().t()
        # Set masks
        obj_mask[b, gj, gi, best_n] = 1
        noobj_mask[b, gj, gi, best_n] = 0
        ignore_thresh = 0.5
        for i, anchor_ious in enumerate(ious.t()):
            noobj_mask[b[i], gj[i], gi[i], anchor_ious > ignore_thresh] = 0

        # Coordinates
        labels_loc_xy[b, gj, gi, best_n, 0] = gx - gx.floor()
        labels_loc_xy[b, gj, gi, best_n, 1] = gy - gy.floor()
        # Width and height
        labels_loc_wh[b, gj, gi, best_n, 0] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
        labels_loc_wh[b, gj, gi, best_n, 1] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
        # One-hot encoding of label
        labels_cls[b, gj, gi, best_n, target_labels] = 1
        # # Compute label correctness and iou at best anchor
        # class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
        # iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

        if self.multiply_area_scale:
            area_scale = (1 - torch.sqrt(labels_loc_wh[..., 0] * labels_loc_wh[..., 1])).unsqueeze(-1).expand_as(labels_loc_wh)
        else:
            area_scale = 1.0
        return obj_mask, noobj_mask, labels_cls, labels_loc_xy, labels_loc_wh, area_scale

    # def _reshape_labels_raw(self, labels, grid_xy, shape, f_id):
    #     """
    #     Reshape the labels.
    #
    #     :param labels: labels from training data
    #     :param grid_xy: the matrix of the grid numbers
    #     :return: labels_obj, labels_cls, lab_loc_xy, lab_loc_wh, labels_boxes, area_scal
    #     """
    #     mask = np.arange(self.anc_num) + self.anc_num * f_id
    #     anchors = self.anchors[mask] / torch.Tensor([self.cfg.TRAIN.IMG_SIZE[1], self.cfg.TRAIN.IMG_SIZE[0]]) * torch.Tensor([shape[1], shape[0]])
    #
    #     labels_obj = torch.zeros([self.batch_size, shape[0], shape[1],
    #                               self.anc_num, 1]).to(self.cfg.TRAIN.DEVICE)
    #     labels_loc = torch.zeros([self.batch_size, shape[0], shape[1],
    #                               self.anc_num, 4]).to(self.cfg.TRAIN.DEVICE)
    #     labels_cls = torch.zeros([self.batch_size, shape[0], shape[1],
    #                               self.anc_num, self.cls_num]).to(self.cfg.TRAIN.DEVICE)
    #     labels_boxes = []
    #     for batch_idx, labs in enumerate(labels):
    #         lab_boxes = []
    #         for lab in labs:
    #             lab = torch.Tensor(lab).to(self.cfg.TRAIN.DEVICE)
    #             box_xy = (lab[1:3] + lab[3:5]) / 2
    #             box_wh = lab[3:5] - lab[1:3]
    #             boxes = torch.cat([box_xy, box_wh])
    #             box_center = (box_xy * torch.Tensor([shape[1], shape[0]]).to(self.cfg.TRAIN.DEVICE)).long()
    #             anc = torch.cat([torch.zeros_like(anchors), anchors], 1)
    #             box_iou = torch.cat([torch.Tensor([0, 0]).to(self.cfg.TRAIN.DEVICE),
    #                                  box_wh * (torch.Tensor([shape[1], shape[0]]).to(self.cfg.TRAIN.DEVICE))])
    #             # TODO: there might be WRONG, when choice a anchor which is not fit for this box in the certain feature map.
    #             iou = iou_xywh(anc.to(self.cfg.TRAIN.DEVICE), box_iou.to(self.cfg.TRAIN.DEVICE), type='N21')
    #             iou_max = torch.max(iou, 0)
    #             anc_idx = iou_max[1].item()
    #             # print('the anchor idx is:', anc_idx)
    #             lab_boxes.append(boxes)
    #             # print(box_center, boxes)
    #
    #             labels_obj[batch_idx, box_center[1], box_center[0], anc_idx, 0] = 1
    #             labels_loc[batch_idx, box_center[1], box_center[0], anc_idx] = boxes
    #             labels_cls[batch_idx, box_center[1], box_center[0], anc_idx, lab[0].long()] = 1
    #         labels_boxes.append(lab_boxes)
    #
    #     lab_loc_xy = labels_loc[..., 0:2]
    #     lab_loc_wh = labels_loc[..., 2:4]
    #     # count the scale of  w*h, in order to count area_scal*wh
    #     if self.multiply_area_scale:
    #         area_scale = (1 - torch.sqrt(lab_loc_wh[..., 0] * lab_loc_wh[..., 1])).unsqueeze(-1).expand_as(lab_loc_wh)
    #     else:
    #         area_scale = 1.0
    #     # print(area_scal.shape, area_scal[0, 14, 39, 12:])
    #     lab_loc_xy = lab_loc_xy * torch.Tensor([shape[1], shape[0]]).to(self.cfg.TRAIN.DEVICE) - grid_xy
    #     anchor_ch = anchors.view(1, 1, 1, self.anc_num, 2).expand(1, shape[0], shape[1], self.anc_num, 2).to(self.cfg.TRAIN.DEVICE)
    #     lab_loc_wh = lab_loc_wh / anchor_ch
    #     lab_loc_wh = torch.log(torch.clamp(lab_loc_wh, 1e-9, 1e9))
    #
    #     return labels_obj, labels_cls, lab_loc_xy, lab_loc_wh, labels_boxes, area_scale

    # def _ignore(self, pre_loc, lab_boxes):
    #     # pylint: disable=no-self-use
    #     """
    #     Count the ignore mask.
    #
    #     :param pre_loc:
    #     :param lab_boxes:
    #     :return:
    #     """
    #     batch = pre_loc.shape[0]
    #     _ignore_mask = []
    #     for i in range(batch):
    #         b_box = lab_boxes[i]
    #         b_box = torch.stack(b_box, 0)
    #         iou = iou_xywh(pre_loc[i], b_box, type='N2N_yolo')
    #         iou_max = torch.max(iou, 0)
    #         # print(iou_max[0][15, 31:34])
    #         ignore_iou = torch.lt(iou_max[0], 0.6)
    #         # print(ignore_iou[15, 31:34])
    #         _ignore_mask.append(ignore_iou)
    #
    #     ignore_mask = torch.stack(_ignore_mask, 0)
    #     # print(ignore_mask[0, 15, 31:34])
    #
    #     return ignore_mask

    def _loss_cal_one_Fmap(self, f_map, f_id, labels, losstype=None):
        """Calculate the loss."""

        pre_obj, pre_cls, pre_loc_xy, pre_loc_wh = self.parsepredict._parse_yolo_predict_fmap(f_map, f_id)
        obj_mask, noobj_mask, labels_cls, labels_loc_xy, labels_loc_wh, area_scale = self._reshape_labels(pre_obj, labels, f_id)
        labels_obj = obj_mask.float()

        '''
        Debug code.
                
        if pre_obj.shape[1] > 15:
            i_0, i_1, i_2 = 0, 14, 37
            print('pre_obj[i_0, i_1, i_2]:\n', pre_obj[i_0, i_1, i_2].t())
            print('GT_obj[i_0, i_1, i_2]:\n', labels_obj[i_0, i_1, i_2].t())
            print('pre_cls[i_0, i_1, i_2]', pre_cls[i_0, i_1, i_2])
            print('GT_cls[i_0, i_1, i_2]', labels_cls[i_0, i_1, i_2])
            print('pre_loc_xy', pre_loc_xy[i_0, i_1, i_2])
            print('lab_loc_xy', labels_loc_xy[i_0, i_1, i_2])
            print('obj_mask', obj_mask[i_0, i_1, i_2].t())
            print('sum of obj_mask', obj_mask.sum())
            print('NOobj_mask', noobj_mask[i_0, i_1, i_2].t())
        '''
        if losstype == 'focalloss':
            # FOCAL loss
            alpha = 0.25
            obj_loss = alpha * pow((torch.ones_like(pre_obj) - pre_obj)[obj_mask], 2) * self.bceloss(pre_obj[obj_mask], labels_obj[obj_mask])
            noobj_loss = (1 - alpha) * pow(pre_obj[noobj_mask], 2) * self.bceloss(pre_obj[noobj_mask], labels_obj[noobj_mask])
            obj_loss = torch.sum(obj_loss) / self.batch_size
            noobj_loss = torch.sum(noobj_loss) / self.batch_size
        elif losstype == 'mse' or losstype is None:
            # nomal loss
            obj_loss = self.mseloss(pre_obj[obj_mask], labels_obj[obj_mask]) / self.batch_size
            noobj_loss = self.mseloss(pre_obj[noobj_mask], labels_obj[noobj_mask]) / self.batch_size
            noobj_loss = noobj_loss * 100.
        else:
            print(losstype, 'is no define.')
            obj_loss = 0.
            noobj_loss = 0.
        lxy_loss = self.mseloss(pre_loc_xy[obj_mask], labels_loc_xy[obj_mask]) / self.batch_size
        lwh_loss = self.mseloss(pre_loc_wh[obj_mask] * area_scale, labels_loc_wh[obj_mask] * area_scale) / self.batch_size
        cls_loss = self.mseloss(pre_cls[obj_mask], labels_cls[obj_mask]) / self.batch_size
        loc_loss = lxy_loss + lwh_loss
        return noobj_loss, obj_loss, cls_loss, loc_loss

    def Loss_Call(self, f_maps, dataset, losstype=None):
        images, labels, datainfos = dataset
        noobj_loss, obj_loss, cls_loss, loc_loss = 0.0, 0.0, 0.0, 0.0
        for f_id, f_map in enumerate(f_maps):
            _noobj_loss, _obj_loss, _cls_loss, _loc_loss = self._loss_cal_one_Fmap(f_map, f_id, labels, losstype)
            noobj_loss += _noobj_loss
            obj_loss += _obj_loss
            cls_loss += _cls_loss
            loc_loss += _loc_loss
        return obj_loss, noobj_loss, cls_loss, loc_loss
