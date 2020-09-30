"""Loss calculation based on yolo."""
import torch
import torch.nn as nn
import math
import numpy as np
from util.util_iou import iou_xywh, xywh2xyxy, iou_xyxy, _iou_wh, bbox_GDCiou
from util.util_get_cls_names import _get_class_names
from lgdet.postprocess.parse_prediction import ParsePredict

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
        self.device = self.cfg.TRAIN.DEVICE
        self.anchors = torch.Tensor(cfg.TRAIN.ANCHORS)
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = cfg.TRAIN.CLASSES_NUM

        self.mseloss = torch.nn.MSELoss()
        self.bceloss = torch.nn.BCELoss()

        self.parsepredict = ParsePredict(cfg)
        self.multiply_area_scale = 0  # whether multiply loss to area_scale.

        self.alpha = 0.25
        self.gamma = 2
        self.use_hard_noobj_loss = False

    def build_targets(self, pre_obj, labels, f_id):
        # x1y1x2y2 to xywh:
        targets = labels.clone()
        targets[..., 2:4] = (labels[..., 2:4] + labels[..., 4:6]) / 2
        targets[..., 4:6] = (labels[..., 4:6] - labels[..., 2:4])
        B, C, H, W = pre_obj.shape

        mask = np.arange(self.anc_num) + self.anc_num * f_id  # be care of the relationship between anchor size and the feature map size.
        anchors_raw = self.anchors[mask]
        anchors = torch.Tensor([(a_w / self.cfg.TRAIN.IMG_SIZE[1] * W, a_h / self.cfg.TRAIN.IMG_SIZE[0] * H) for a_w, a_h in anchors_raw]).to(self.device)

        num_target = targets.shape[0]
        gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
        gain[2:] = torch.tensor([W, H, W, H])  # xyxy gain
        t = targets * gain

        self.anc_num = anchors.shape[0]  # number of anchors
        at = torch.arange(self.anc_num).view(self.anc_num, 1).repeat(1, num_target)  # anchor tensor, same as .repeat_interleave(num_target)

        # Match targets to anchors
        if num_target:
            j = _iou_wh(anchors, t[:, 4:6]) > 0.2  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            a, t = at[j], t.repeat(self.anc_num, 1, 1)[j]  # filter
        # Define
        batch, tcls = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = gxy.long()
        gi, gj = gij.T  # grid xy indices

        indices = [batch, a, gj, gi]  # image, anchor, grid indices
        tbox = torch.cat((gxy - gij, gwh), 1)  # box
        anch = anchors[a]  # anchors
        if tcls.shape[0]:  # if any targets
            assert tcls.max() < self.cls_num, 'cls_num is beyound the classes.'
        return tcls, tbox, indices, anch

    def _loss_cal_one_Fmap(self, f_map, f_id, labels, losstype=None):
        """Calculate the loss."""
        # init loss.
        lcls, lbox, lobj = [torch.FloatTensor([0]).to(self.device) for _ in range(3)]

        pre_obj, pre_cls, pre_loc_xy, pre_loc_wh = self.parsepredict._parse_yolo_predict_fmap(f_map, f_id)
        tcls, tbox, indices, anchors = self.build_targets(pre_obj, labels, f_id)  # targets

        tobj = torch.zeros_like(pre_obj)  # target obj
        nb = indices[0].shape[0]  # number of targets
        loss_ratio = {'box': 1, 'cls': 1, 'obj': 1}
        if nb:
            pre_cls, pre_xy, pre_wh = [i[indices] for i in [pre_cls, pre_loc_xy, pre_loc_wh]]
            t_cls = torch.zeros_like(pre_cls)  # targets
            t_cls[range(nb), tcls] = 1
            lcls = self.bceloss(pre_cls, t_cls)  # BCE

            if losstype == 'mse':
                # mse:
                twh = torch.log(tbox[..., 2:] / anchors + 1e-16)  # torch.log(gw / anchors[best_n][:, 0] + 1e-16)
                lxy = self.mseloss(pre_xy, tbox[..., :2])
                lwh = self.mseloss(pre_wh, twh)
                lbox = lxy + lwh
                tobj[indices] = 1.0

            elif losstype == 'giou':
                pre_wh = pre_wh.exp().clamp(max=1E3) * anchors
                pbox = torch.cat((pre_xy, pre_wh), 1)  # predicted box
                giou = bbox_GDCiou(pbox.t(), tbox, x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
                lbox = (1.0 - giou).mean()  # giou loss
                # Obj
                gr = 0.0
                tobj[indices] = (1.0 - gr) + gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio
                loss_ratio['obj'] = 64.3
                loss_ratio['cls'] = 9.35
                loss_ratio['box'] = 3.54

            elif losstype == 'focalloss':
                ...
            else:
                print('sorry, no such a loss type...')
        lobj = self.bceloss(pre_obj, tobj)  # obj loss
        # obj_loss = self.bceloss(pre_obj[indices], tobj[indices])

        total_loss = loss_ratio['obj'] * lobj + loss_ratio['cls'] * lcls + loss_ratio['box'] * lbox
        metrics = {'obj_loss': [lobj.item()],
                   # 'obj_loss': [obj_loss.item()],
                   # 'noobj_loss': [noobj_loss.item()],
                   'cls_loss': [lcls.item()],
                   'box_loss': [lbox.item()],
                   }
        return total_loss, metrics

    def Loss_Call(self, f_maps, dataset, kwargs):
        losstype = kwargs['losstype']
        images, labels, datainfos = dataset
        metrics = {}
        total_loss = torch.FloatTensor([0]).to(self.device)
        for f_id, f_map in enumerate(f_maps):
            _loss, _metrics = self._loss_cal_one_Fmap(f_map, f_id, labels, losstype)
            total_loss += _loss
            if f_id == 0:
                metrics = _metrics
            else:
                for k, v in metrics.items():
                    metrics[k].append(_metrics[k][0])

        for k, v in metrics.items():
            metrics[k] = '%.3f' % np.asarray(v, np.float32).mean()

        return {'total_loss': total_loss, 'metrics': metrics}
