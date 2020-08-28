"""Yolo v3 net."""
import torch
import torch.nn as nn
from collections import OrderedDict
from lgdet.model.aid_Models.DARKNET import DarkNet


def darknet_53():
    return DarkNet([1, 2, 8, 8, 4])

from ..registry import MODELS


@MODELS.registry()
class YOLOV3(nn.Module):
    """Constructs a darknet-21 model.
    """

    def __init__(self, cfg):
        super(YOLOV3, self).__init__()
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = cfg.TRAIN.CLASSES_NUM
        self.final_out = self.anc_num * (1 + 4 + self.cls_num)
        self.layers_out_filters = [64, 128, 256, 512, 1024]
        self.backbone = darknet_53()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv_set0 = self._conv_set([512, 1024], self.layers_out_filters[-1], self.final_out)
        self.conv_set1 = self._conv_set([256, 512], self.layers_out_filters[-2] + 256, self.final_out)
        self.conv_set2 = self._conv_set([128, 256], self.layers_out_filters[-3] + 128, self.final_out)

        self.conv_next1 = self._conv(512, 256, 1)
        self.conv_next2 = self._conv(256, 128, 1)

    def _conv(self, _in, _out, k_size):
        pad = (k_size - 1) // 2 if k_size else 0
        _set_out = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(_in, _out, k_size, stride=1, padding=pad, bias=False)),
            ('bn', nn.BatchNorm2d(_out)),
            ('relu', nn.LeakyReLU(0.1))
        ]))
        return _set_out

    def _conv_set(self, in_out, _in, _out):
        mlist = nn.ModuleList([
            self._conv(_in, in_out[0], 1),
            self._conv(in_out[0], in_out[1], 3),
            self._conv(in_out[1], in_out[0], 1),
            self._conv(in_out[0], in_out[1], 3),
            self._conv(in_out[1], in_out[0], 1),
            self._conv(in_out[0], in_out[1], 3),
        ])
        mlist.add_module("conv_out", nn.Conv2d(in_out[1], _out, 1, 1, 0, True))
        return mlist

    def forward(self, **args):
        def _branch(_embedding, _in):
            out_branch = None
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch

        x = args['input_x']
        # x2, x1, x0: 256, 512, 1024
        x2, x1, x0 = self.backbone(x)
        # featuremap0:144, net_out0:512
        featuremap0, net_out0 = _branch(self.conv_set0, x0)
        # in_x1:256
        in_x1 = self.conv_next1(net_out0)
        in_x1 = self.upsample(in_x1)
        # in_x1:768
        in_x1 = torch.cat([in_x1, x1], 1)
        # featuremap1:144, net_out1:256
        featuremap1, net_out1 = _branch(self.conv_set1, in_x1)
        in_x2 = self.conv_next2(net_out1)
        in_x2 = self.upsample(in_x2)
        # in_x2:256+128=384
        in_x2 = torch.cat([in_x2, x2], 1)
        # featuremap2: 144
        featuremap2, _ = _branch(self.conv_set2, in_x2)
        # net:[N, 24, 60, 144]
        featuremap0 = featuremap0.permute([0, 2, 3, 1])
        featuremap1 = featuremap1.permute([0, 2, 3, 1])
        featuremap2 = featuremap2.permute([0, 2, 3, 1])

        return featuremap0, featuremap1, featuremap2
