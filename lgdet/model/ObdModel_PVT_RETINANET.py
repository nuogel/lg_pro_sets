import torch
import torch.nn as nn
import numpy as np
from lgdet.util.util_anchor_maker import Anchors
from lgdet.model.backbone.resnet import resnet50
from lgdet.model.head.retina_head import RetinaHead
from lgdet.model.neck.fpn_neck import FPN
from ..registry import MODELS
from functools import partial
from lgdet.model.backbone.pvt import PyramidVisionTransformer
from lgdet.model.backbone.pvt_v2 import PyramidVisionTransformerV2

@MODELS.registry()
class PVT_RETINANET(nn.Module):
    def __init__(self, config=None):
        super(PVT_RETINANET, self).__init__()
        self.config = config
        use_p5 = True
        fpn_out_channels = 256
        self.config.fpn_out_channels = 256
        self.config.use_GN_head = True
        self.config.freeze_bn = False
        self.config.freeze_stage_1 = False

        pvt = 'PyramidVisionTransformer'  # PyramidVisionTransformerV2
        self.backbone = eval(pvt)(num_stages=4,
                                  patch_size=4, embed_dims=[128, 128, 256, 256], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4],
                                  qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
                                  sr_ratios=[8, 4, 2, 1], drop_rate=0.1, drop_path_rate=0.1)

        self.fpn = FPN(features=fpn_out_channels, use_p5=use_p5)
        self.head = RetinaHead(config=self.config)
        self.anchors = Anchors()

    def train(self, mode=True):
        super(PVT_RETINANET, self).train(mode=True)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            class_name = module.__class__.__name__
            if class_name.find('BatchNorm') != -1:
                for p in module.parameters():
                    p.requires_grad = False

        if self.config.freeze_bn:
            self.apply(freeze_bn)
            print("success freeze bn")
        if self.config.freeze_stage_1:
            self.backbone.freeze_stages(1)
            print("success freeze stage 1")

    def forward(self, input_x, **args):
        x = input_x
        _, C3, C4, C5 = self.backbone(x)
        all_p_level = self.fpn([C3, C4, C5])
        cls_logits, reg_preds = self.head(all_p_level)
        anchors = self.anchors(input_x)
        # cls_logits = cls_logits.sigmoid()
        return cls_logits, reg_preds, anchors

    def weights_init(self):
        pass
