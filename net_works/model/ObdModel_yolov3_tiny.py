import torch
import torch.nn as nn
from collections import OrderedDict


def make_layer(in_ch, out_ch, ksize=3, max_pool=2, conv=True, up_samping=0, last_layer=False):
    layers = []
    if conv:
        padding = (ksize - 1) // 2
        layers.append(('conv2d', nn.Conv2d(in_ch, out_ch, kernel_size=ksize, padding=padding)))
        if not last_layer:  # the last layer don't need this activation functions.
            layers.append(('batchnorm', nn.BatchNorm2d(out_ch)))
            layers.append(('leakyrelu', nn.LeakyReLU(0.1)))
    if max_pool:
        layers.append(('max_pool', nn.MaxPool2d(2, stride=max_pool)))
    if up_samping:
        layers.append(('up_samping', nn.Upsample(scale_factor=up_samping, mode='bilinear', align_corners=True)))

    return nn.Sequential(OrderedDict(layers))


class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()
        self.out_chs = [16, 32, 64, 128, 256, 512, 1024]

        self.layer1 = make_layer(3, self.out_chs[0])
        self.layer2 = make_layer(self.out_chs[0], self.out_chs[1])
        self.layer3 = make_layer(self.out_chs[1], self.out_chs[2])
        self.layer4 = make_layer(self.out_chs[2], self.out_chs[3])
        self.layer5_conv = make_layer(self.out_chs[3], self.out_chs[4], max_pool=0, up_samping=False)
        self.layer5_pool = make_layer(in_ch=0, out_ch=0, max_pool=2, conv=False, up_samping=False)

        self.layer6 = make_layer(self.out_chs[4], self.out_chs[5], max_pool=0)
        self.layer7 = make_layer(self.out_chs[5], self.out_chs[6], max_pool=0)

    def forward(self, input):
        x = self.layer1(input)  # 208
        x = self.layer2(x)  # 104
        x = self.layer3(x)  # 52
        x = self.layer4(x)  # 26
        x = self.layer5_conv(x)  # 26
        f2 = x
        x = self.layer5_pool(x)  # 13
        x = self.layer6(x)
        f1 = self.layer7(x)
        return f1, f2  # f1:13x13x1024; f2:26x26x256


class YoloV3_Tiny(nn.Module):
    def __init__(self, cfg):
        super(YoloV3_Tiny, self).__init__()
        self.anc_num = cfg.TRAIN.FMAP_ANCHOR_NUM
        self.cls_num = len(cfg.TRAIN.CLASSES)
        out_ch = self.anc_num * (1 + 4 + self.cls_num)
        self.ch_1 = [1024, 256, 512, out_ch]
        self.ch_2 = [256, 128, 384, 256, out_ch]

        self.backbone = BackBone()
        self.bb1_1 = make_layer(self.ch_1[0], self.ch_1[1], ksize=1, max_pool=0)
        self.bb1_2 = make_layer(self.ch_1[1], self.ch_1[2], max_pool=0)
        self.bb1_3 = make_layer(self.ch_1[-2], self.ch_1[-1], ksize=1, max_pool=0, last_layer=True)

        self.bb2_1 = make_layer(self.ch_2[0], self.ch_2[1], ksize=1, max_pool=0, up_samping=2)
        self.bb2_cat = torch.cat
        self.bb2_2 = make_layer(self.ch_2[2], self.ch_2[3], max_pool=0)
        self.bb2_3 = make_layer(self.ch_2[-2], self.ch_2[-1], ksize=1, max_pool=0, last_layer=True)

    def forward(self, train_data):
        if isinstance(train_data, tuple):
            x, lab = train_data
        else:
            x = train_data
        input = x.permute([0, 3, 1, 2])

        f1, f2 = self.backbone(input)  # jump2
        net1 = self.bb1_1(f1)
        net2 = net1  # jump1

        net1 = self.bb1_2(net1)
        net1 = self.bb1_3(net1)  # 13x13x108

        net2 = self.bb2_1(net2)  # jump1_TO   26x26x128
        net2 = self.bb2_cat((f2, net2), 1)  # jump2_TO    26x26x256+26x26x128=26x26x384
        net2 = self.bb2_2(net2)
        net2 = self.bb2_3(net2)

        f_map1 = net1.permute([0, 2, 3, 1])
        f_map2 = net2.permute([0, 2, 3, 1])

        return [f_map1, f_map2, ]
