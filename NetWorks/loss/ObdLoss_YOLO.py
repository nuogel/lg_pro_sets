"""Loss calculation based on yolo."""
import torch
import numpy as np
from util.util_iou import iou_xywh
from util.util_get_cls_names import _get_class_names
from util.util_parse_prediction import ParsePredict
from torch.autograd import Variable

'''
with the new yolo loss, in 56 images, loss is 0.18 and map is 0.2.and the test show wrong bboxes.
with the new yolo loss, in 8 images, loss is 0.015 and map is 0.99.and the test show terrible bboxes.

'''


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

        self.mseloss = torch.nn.MSELoss(reduction='sum')
        self.bceloss = torch.nn.BCELoss()
        self.class_name = _get_class_names(cfg.PATH.CLASSES_PATH)

        self.parsepredict = ParsePredict(cfg)
        self.multiply_area_scale = 0  # whether multiply loss to area_scale.

        self.alpha = 0.25
        self.gamma = 2

    def _reshape_labels(self, pre_obj, pre_cls, pre_loc_xy, pre_loc_wh, labels, f_id):
        """
        Reshape the labels.

        :param labels: labels from training data
        :param grid_xy: the matrix of the grid numbers
        :return: labels_obj, labels_cls, lab_loc_xy, lab_loc_wh, labels_boxes, area_scal
        """
        B, H, W = pre_obj.shape[0:3]
        mask = np.arange(self.anc_num) + self.anc_num * f_id  # be care of the relationship between anchor size and the feature map size.
        anchors = self.anchors[mask] / torch.Tensor([self.cfg.TRAIN.IMG_SIZE[1], self.cfg.TRAIN.IMG_SIZE[0]])  # * torch.Tensor([W, H])
        anchors = anchors.to(self.cfg.TRAIN.DEVICE)
        obj_mask = torch.BoolTensor(B, H, W, self.anc_num).fill_(0).to(self.cfg.TRAIN.DEVICE)
        noobj_mask = torch.BoolTensor(B, H, W, self.anc_num).fill_(1).to(self.cfg.TRAIN.DEVICE)
        labels_loc_xy = torch.zeros([B, H, W, self.anc_num, 2]).to(self.cfg.TRAIN.DEVICE)
        labels_loc_wh = torch.zeros([B, H, W, self.anc_num, 2]).to(self.cfg.TRAIN.DEVICE)
        labels_cls = torch.zeros([B, H, W, self.anc_num, self.cls_num]).to(self.cfg.TRAIN.DEVICE)

        target_boxes = labels[..., 2:6]
        # x1y1x2y2->xywh
        gx1y1 = target_boxes[..., :2]
        gx2y2 = target_boxes[..., 2:]
        gxy = (gx1y1 + gx2y2) / 2.0 * torch.Tensor([W, H]).to(self.cfg.TRAIN.DEVICE)
        gwh = gx2y2 - gx1y1

        box_iou = torch.cat([torch.zeros_like(gwh), gwh], 1)
        anc = torch.cat([torch.zeros_like(anchors), anchors], 1)

        ious = iou_xywh(anc.to(self.cfg.TRAIN.DEVICE), box_iou.to(self.cfg.TRAIN.DEVICE), type='N2N')
        best_ious, best_n = ious.max(0)
        # Separate target values
        b, target_labels = labels[..., :2].long().t()
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

        watch = 1
        if watch:
            watcher = pre_obj[obj_mask].mean()
            watcher2 = pre_obj[noobj_mask].mean()
            # print(labels_cls[obj_mask].sum())
            # print(obj_mask.sum())
            # a = pre_cls[b, best_n, gj, gi].argmax(-1)== target_labels
            watcher3 = (pre_cls[b, gj, gi, best_n].argmax(-1) == target_labels).float().mean()
            # # Compute label correctness and iou at best anchor
            pre_loc = torch.cat((pre_loc_xy, pre_loc_wh), -1)
            # iou_scores = iou_xywh(pre_loc[b, gj, gi, best_n].to(self.cfg.TRAIN.DEVICE),
            #                                          target_boxes.to(self.cfg.TRAIN.DEVICE), type='N2N')
            # iou50 = (iou_scores > 0.5).float()

            print("layer %d: pre_obj.mean():%0.4f pre_noobj.mean():%0.4f pre_cls.acc():%0.4f" % (f_id, watcher, watcher2, watcher3))

        if self.multiply_area_scale:
            # TODO: labels_loc_wh is no fit for area_scale
            area_scale = (1 - torch.sqrt(labels_loc_wh[..., 0] * labels_loc_wh[..., 1])).unsqueeze(-1).expand_as(labels_loc_wh)
        else:
            area_scale = 1.0
        return obj_mask, noobj_mask, labels_cls, labels_loc_xy, labels_loc_wh

    def _loss_cal_one_Fmap(self, f_map, f_id, labels, losstype=None):
        """Calculate the loss."""

        pre_obj, pre_cls, pre_loc_xy, pre_loc_wh = self.parsepredict._parse_yolo_predict_fmap(f_map, f_id)
        obj_mask, noobj_mask, labels_cls, labels_loc_xy, labels_loc_wh = self._reshape_labels(pre_obj, pre_cls, pre_loc_xy, pre_loc_wh, labels, f_id)

        labels_obj = obj_mask.float()

        '''
        #Debug code.
                
        if pre_obj.shape[1] < 18:
            i_0, i_1, i_2 = 0, 11, 14
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
            # # FOCAL loss
            obj_loss = (self.alpha * (((1. - pre_obj)[obj_mask]) ** self.gamma)) * self.bceloss(pre_obj[obj_mask], labels_obj[obj_mask])
            noobj_loss = ((1 - self.alpha) * ((pre_obj[noobj_mask]) ** self.gamma)) * self.bceloss(pre_obj[noobj_mask], labels_obj[noobj_mask])
            obj_loss = torch.sum(obj_loss) / self.batch_size
            noobj_loss = torch.sum(noobj_loss) / self.batch_size

        elif losstype == 'mse' or losstype is None:
            # nomal loss
            obj_loss = self.mseloss(pre_obj[obj_mask], labels_obj[obj_mask]) / self.batch_size
            noobj_loss = 100 * self.mseloss(pre_obj[noobj_mask], labels_obj[noobj_mask]) / self.batch_size
        else:
            print(losstype, 'is not define.')
            obj_loss = 0.
            noobj_loss = 0.
        lxy_loss = self.mseloss(pre_loc_xy[obj_mask], labels_loc_xy[obj_mask]) / self.batch_size
        lwh_loss = self.mseloss(pre_loc_wh[obj_mask], labels_loc_wh[obj_mask]) / self.batch_size
        cls_loss = self.mseloss(pre_cls[obj_mask], labels_cls[obj_mask]) / self.batch_size
        loc_loss = lxy_loss + lwh_loss
        return obj_loss, noobj_loss, cls_loss, loc_loss

    def _focal_loss(self, pred, target):  # not used
        ce = self.bceloss(pred, target)
        alpha = target * self.alpha + (1. - target) * (1. - self.alpha)
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_loss = alpha * (1. - pt) ** self.gamma * ce
        return focal_loss

    def Loss_Call(self, f_maps, dataset, losstype=None):
        images, labels, datainfos = dataset
        noobj_loss, obj_loss, cls_loss, loc_loss = 0.0, 0.0, 0.0, 0.0
        for f_id, f_map in enumerate(f_maps):
            _obj_loss, _noobj_loss, _cls_loss, _loc_loss = self._loss_cal_one_Fmap(f_map, f_id, labels, losstype)
            obj_loss += _obj_loss
            noobj_loss += _noobj_loss
            cls_loss += _cls_loss
            loc_loss += _loc_loss
        return obj_loss, noobj_loss, cls_loss, loc_loss


'''
yolov3 test of voc2007
mAP:  0.028912177841298514
F1SCORE:  [0.17535545 0.10291595 0.01410437 0.06827309 0.00269179 0.04761904
 0.25238854 0.09503239 0.06493507 0.13333333 0.0247678  0.04930662
 0.10367171 0.08       0.2844862  0.0083682  0.14888337 0.03023758
 0.06417113 0.11498974]
map: [0.05944339 0.01411944 0.00080376 0.01195635 0.00013837 0.01090248
 0.10537232 0.02183381 0.00954061 0.04768166 0.0083612  0.00670902
 0.03323201 0.02257409 0.12607616 0.00022342 0.04919026 0.00216555
 0.01138146 0.03653822]
prec: [0.33333334 0.15463917 0.03759398 0.16190477 0.01162791 0.175
 0.32646757 0.23655914 0.08951407 0.30769232 0.16666667 0.13445379
 0.3529412  0.22222222 0.29788572 0.024      0.32608697 0.10447761
 0.16666667 0.22222222]
rec: [0.11897106 0.07712083 0.00868056 0.043257   0.00152207 0.02755906
 0.20571057 0.05945946 0.05094614 0.08510638 0.01337793 0.03018868
 0.06075949 0.04878049 0.27224028 0.00506757 0.09646302 0.01767677
 0.0397351  0.07756232]
 '''