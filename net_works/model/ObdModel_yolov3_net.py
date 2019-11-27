"""Yolo v3 net."""
import torch
import torch.nn as nn
import math
from collections import OrderedDict


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        self.layer1 = self._make_layer([32, 64], layers[0])
        self.layer2 = self._make_layer([64, 128], layers[1])
        self.layer3 = self._make_layer([128, 256], layers[2])
        self.layer4 = self._make_layer([256, 512], layers[3])
        self.layer5 = self._make_layer([512, 1024], layers[4])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        layers = []
        #  downsample
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3,
                                            stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        #  blocks
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out2 = self.layer3(x)  # 256
        out1 = self.layer4(out2)  # 512
        out0 = self.layer5(out1)  # 1024

        return out2, out1, out0


def darknet_21():
    return DarkNet([1, 1, 2, 2, 1])


class YoloV3(nn.Module):
    """Constructs a darknet-21 model.
    """

    def __init__(self, cfg):
        super(YoloV3, self).__init__()
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = len(cfg.TRAIN.CLASSES)
        self.final_out = self.anc_num * (1 + 4 + self.cls_num)
        self.layers_out_filters = [64, 128, 256, 512, 1024]
        self.backbone = darknet_21()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # self.final_in = 512
        self.cls_pred_prob = torch.nn.Softmax(-1)
        self.obj_pred = torch.nn.Sigmoid()

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

    def forward(self, train_data, **args):
        def _branch(_embedding, _in):
            out_branch = None
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch

        if isinstance(train_data, tuple):
            x, lab = train_data
        else:
            x = train_data
        x = x.permute([0, 3, 1, 2])

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
