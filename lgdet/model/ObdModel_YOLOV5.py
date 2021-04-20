"""Yolo v3 net."""
import torch
import torch.nn as nn
from lgdet.model.backbone.yolov5_backbone import YOLOV5BACKBONE
from lgdet.model.neck.yolov5_neck import YOLOV5NECK
from ..registry import MODELS


@MODELS.registry()
class YOLOV5(nn.Module):
    """Constructs a darknet-21 model.
    """

    def __init__(self, cfg):
        super(YOLOV5, self).__init__()
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = cfg.TRAIN.CLASSES_NUM
        self.final_out = self.anc_num * (1 + 4 + self.cls_num)
        self.layers_out_filters = [64, 128, 256, 512, 1024]

        self.yolov5_type = cfg.TRAIN.TYPE

        if self.yolov5_type == 's':
            backbone_chs = [32, 64, 64, 128, 128, 256, 256, 512, 512, 512, 256]
            backbone_csp = [1, 3, 3, 1]
            neck_chs = [512, 256, 128, 128, 128, 256, 256, 512]
            neck_csp = [1, 1, 1, 1]
            deteck = [128, 256, 512]
        elif self.yolov5_type == 'm':
            ...
        elif self.yolov5_type == 'l':
            backbone_chs = [64, 128, 128, 256, 256, 512, 512, 1024, 1024, 1024, 512]
            backbone_csp = [3, 9, 9, 3]
            neck_chs = [1024, 512, 256, 256, 256, 512, 512, 1024]
            neck_csp = [3, 3, 3, 3]
            deteck = [256, 512, 1024]
        elif self.yolov5_type == 'x':
            ...

        self.backbone = YOLOV5BACKBONE(chs=backbone_chs, csp=backbone_csp)
        self.neck = YOLOV5NECK(chs=neck_chs, csp=neck_csp)

    def forward(self, **args):
        x = args['input_x']
        backbone = self.backbone(x)
        featuremap0, featuremap1, featuremap2 = self.neck(backbone)
        return featuremap0, featuremap1, featuremap2
