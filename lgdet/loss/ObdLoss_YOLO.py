"""Loss calculation based on yolo."""
import torch
import torch.nn as nn
import math
import numpy as np
from util.util_iou import iou_xywh, xywh2xyxy, iou_xyxy, _iou_wh
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
        lcls, lxy, lwh, lobj = [torch.FloatTensor([0]).to(self.device) for _ in range(4)]

        pre_obj, pre_cls, pre_loc_xy, pre_loc_wh = self.parsepredict._parse_yolo_predict_fmap(f_map, f_id)
        tcls, tbox, indices, anchors = self.build_targets(pre_obj, labels, f_id)  # targets

        tobj = torch.zeros_like(pre_obj)  # target obj
        tobj[indices] = 1.0
        nb = indices[0].shape[0]  # number of targets
        if nb:
            pre_cls, pre_xy, pre_wh = [i[indices] for i in [pre_cls, pre_loc_xy, pre_loc_wh]]

            # mse:
            twh = torch.log(tbox[..., 2:] / anchors + 1e-16)  # torch.log(gw / anchors[best_n][:, 0] + 1e-16)
            lxy = self.mseloss(pre_xy, tbox[..., :2])
            lwh = self.mseloss(pre_wh, twh)

            t_cls = torch.zeros_like(pre_cls)  # targets
            t_cls[range(nb), tcls] = 1
            lcls = self.bceloss(pre_cls, t_cls)  # BCE

        lobj = self.bceloss(pre_obj, tobj)  # obj loss

        total_loss = lobj + lcls + lxy + lwh
        metrics = {'obj_loss': [lobj.item()],
                   'cls_loss': [lcls.item()],
                   'xy_loss': [lcls.item()],
                   'wh_loss': [lcls.item()],
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
            metrics[k] = '%.3f' % np.asarray(v).mean()

        return {'total_loss': total_loss, 'metrics': metrics}
