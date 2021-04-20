"""Parse the predictions."""
import torch
import numpy as np
from lgdet.util.util_nms.util_nms_python import NMS


class ParsePredict_yolo:
    # TODO: 分解parseprdict.
    def __init__(self, cfg):
        self.cfg = cfg
        self.anchors = torch.Tensor(cfg.TRAIN.ANCHORS)
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = cfg.TRAIN.CLASSES_NUM
        self.NMS = NMS(cfg)
        self.device = self.cfg.TRAIN.DEVICE
        self.scale_x_y = 2  # 1.05

    def parse_predict(self, f_maps):
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
            _pre_obj = _pre_obj.unsqueeze(-1)
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

        labels_predict = []
        BN = pre_obj.shape[0]
        pre_obj_thresh = 0
        if pre_obj_thresh:
            for bi in range(BN):
                # score is conf*cls,but obj is pre_obj>thresh.
                mask = pre_obj[bi] > self.cfg.TEST.SCORE_THRESH
                if torch.sum(mask) > 0:
                    pre_obj_i = pre_obj[bi][mask]
                    pre_cls_i = pre_cls[bi][mask.squeeze()]
                    pre_cls_score_i, pre_cls_id = pre_cls_i.max(-1)
                    pre_score = pre_obj_i * pre_cls_score_i  # score of obj * score of class.
                    pre_loc_i = pre_loc[bi][mask.squeeze()]
                    labels_predict.append(self.NMS._nms(pre_score, pre_cls_id, pre_loc_i, xywh2x1y1x2y2=True))
                else:
                    labels_predict.append([])
        else:
            #  conf*cls>thresh

            pre_score = pre_obj * pre_cls
            labels_predict = self.NMS.forward(pre_score, pre_loc)

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

        ###1: pre deal feature mps
        B, C, H, W = f_map.shape
        f_map = f_map.view(B, self.anc_num, self.cls_num + 5, H, W)
        _permiute = (0, 1, 3, 4, 2)  # ((0, 1, 3, 4, 2),   (0, 3, 4, 1, 2))
        f_map = f_map.permute(_permiute).contiguous()

        pred_conf = f_map[..., 0]  # NO sigmoid()
        pred_xy = torch.sigmoid(f_map[..., 1:3])  # Center x
        # Grid Sensitive
        if self.scale_x_y > 1:
            pred_xy = self.scale_x_y * pred_xy - 0.5 * (self.scale_x_y - 1.0)  # Grid Sensitive
        pred_wh = (f_map[..., 3:5].sigmoid() * 2) ** 2  # Width
        pred_cls = f_map[..., 5:].sigmoid()  # Cls pred.

        mask = np.arange(self.anc_num) + self.anc_num * f_id
        anchors_raw = self.anchors[mask]
        anchors = torch.Tensor([(a_w / self.cfg.TRAIN.IMG_SIZE[1] * W, a_h / self.cfg.TRAIN.IMG_SIZE[0] * H) for a_w, a_h in anchors_raw]).to(self.device)
        grid_wh = torch.Tensor([W, H, W, H]).to(self.device)
        grid_x = torch.arange(W).repeat(H, 1).view([1, H, W, 1]).to(self.device)
        grid_y = torch.arange(H).repeat(W, 1).t().view([1, H, W, 1]).to(self.device)
        '''
        yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
        self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()
        '''
        if _permiute == (0, 1, 3, 4, 2):
            anchor_ch = anchors.view(1, self.anc_num, 1, 1, 2)
            grid_xy = torch.cat([grid_x, grid_y], -1).unsqueeze(1)
        else:
            anchor_ch = anchors.view(1, 1, 1, self.anc_num, 2)
            grid_xy = torch.cat([grid_x, grid_y], -1).unsqueeze(3).expand(1, H, W, self.anc_num, 2).to(self.device)

        pre_xy = pred_xy + grid_xy.float()
        pre_wh = pred_wh * anchor_ch
        pre_box = torch.cat([pre_xy, pre_wh], -1) / grid_wh

        if not tolabel:  # training
            return pred_conf, pred_cls, pred_xy, pred_wh, pre_box
        else:  # test
            pred_conf = torch.sigmoid(pred_conf)
            return pred_conf, pred_cls, pre_box

    def _parse_yolo_predict_fmap_old(self, f_map, f_id, tolabel=False):
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
        f_map = f_map.permute(0, 2, 3, 1)
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
            expand(1, shape[0], shape[1], self.anc_num, 2).expand_as(pre_loc_xy).type(torch.FloatTensor).to(loc_pred.device)

        # prepare gird xy
        box_ch = torch.Tensor([shape[1], shape[0]]).to(loc_pred.device)
        pre_realtive_xy = (pre_loc_xy + grid_xy) / box_ch
        anchor_ch = anchors.view(1, 1, 1, self.anc_num, 2).expand(1, shape[0], shape[1], self.anc_num, 2).to(loc_pred.device)
        pre_realtive_wh = pre_loc_wh.exp() * anchor_ch

        pre_relative_box = torch.cat([pre_realtive_xy, pre_realtive_wh], -1)

        if tolabel:
            return pre_obj, pre_cls, pre_relative_box
        return pre_obj, pre_cls, pre_relative_box, pre_loc_xy, pre_loc_wh, grid_xy, shape
