"""Definition of the apollo yolo network."""
from collections import OrderedDict
import torch

"""
Definition of the apollo yolo network,
"""
import torch
import torch.nn as nn


class ApolloYoloV2(torch.nn.Module):
    """
    apollo yolo 2d
    """

    def __init__(self, cfg):
        super(ApolloYoloV2, self).__init__()
        anc_num, cls_num = cfg.TRAIN.ANCHOR_FMAP_NUM, len(cfg.TRAIN.CLASSES)
        self.anc_num = anc_num
        self.cls_num = cls_num
        self.power = torch.pow
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, 1, 1, bias=True)
        self.conv1_relu = torch.nn.ReLU(1)
        self.pool1 = torch.nn.MaxPool2d(2, 2, 0)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 1, 1, 1, bias=True)
        self.conv2_relu = torch.nn.ReLU(1)
        self.pool2 = torch.nn.MaxPool2d(2, 2, 0)
        self.conv3_1 = torch.nn.Conv2d(32, 64, 3, 1, 1, 1, bias=True)
        self.conv3_1_relu = torch.nn.ReLU(1)
        self.conv3_2 = torch.nn.Conv2d(64, 32, 1, 1, 0, 1, bias=True)
        self.conv3_2_relu = torch.nn.ReLU(1)
        self.conv3_3 = torch.nn.Conv2d(32, 64, 3, 1, 1, 1, bias=True)
        self.conv3_3_relu = torch.nn.ReLU(1)
        self.pool3 = torch.nn.MaxPool2d(2, 2, 0)
        self.conv4_1 = torch.nn.Conv2d(64, 128, 3, 1, 1, 1, bias=True)
        self.conv4_1_relu = torch.nn.ReLU(1)
        self.conv4_2 = torch.nn.Conv2d(128, 64, 1, 1, 0, 1, bias=True)
        self.conv4_2_relu = torch.nn.ReLU(1)
        self.conv4_3 = torch.nn.Conv2d(64, 128, 3, 1, 1, 1, bias=True)
        self.conv4_3_relu = torch.nn.ReLU(1)
        self.pool4 = torch.nn.MaxPool2d(2, 2, 0)
        self.conv5_1 = torch.nn.Conv2d(128, 256, 3, 1, 1, 1, bias=True)
        self.conv5_1_relu = torch.nn.ReLU(1)
        self.conv5_2 = torch.nn.Conv2d(256, 128, 1, 1, 0, 1, bias=True)
        self.conv5_2_relu = torch.nn.ReLU(1)
        self.conv5_3 = torch.nn.Conv2d(128, 256, 3, 1, 1, 1, bias=True)
        self.conv5_3_relu = torch.nn.ReLU(1)
        self.conv5_4 = torch.nn.Conv2d(256, 128, 1, 1, 0, 1, bias=True)
        self.conv5_4_relu = torch.nn.ReLU(1)
        self.conv5_5 = torch.nn.Conv2d(128, 256, 3, 1, 1, 1, bias=True)
        self.conv5_5_relu = torch.nn.ReLU(1)
        self.pool5 = torch.nn.MaxPool2d(3, 1, 1)
        self.conv6_1_nodilate = torch.nn.Conv2d(256, 512, 5, 1, 2, 1, bias=True)
        self.conv6_1_relu = torch.nn.ReLU(1)
        self.conv6_2 = torch.nn.Conv2d(512, 256, 1, 1, 0, 1, bias=True)
        self.conv6_2_relu = torch.nn.ReLU(1)
        self.conv6_3 = torch.nn.Conv2d(256, 512, 3, 1, 1, 1, bias=True)
        self.conv6_3_relu = torch.nn.ReLU(1)
        self.conv6_4 = torch.nn.Conv2d(512, 256, 1, 1, 0, 1, bias=True)
        self.conv6_4_relu = torch.nn.ReLU(1)
        self.conv6_5 = torch.nn.Conv2d(256, 512, 3, 1, 1, 1, bias=True)
        self.conv6_5_relu = torch.nn.ReLU(1)
        self.conv7_1 = torch.nn.Conv2d(512, 512, 3, 1, 1, 1, bias=True)
        self.conv7_1_relu = torch.nn.ReLU(1)
        self.conv7_2 = torch.nn.Conv2d(512, 512, 3, 1, 1, 1, bias=True)
        self.conv7_2_relu = torch.nn.ReLU(1)
        self.concat8 = torch.cat
        self.conv9 = torch.nn.Conv2d(768, 512, 3, 1, 1, 1, bias=True)
        self.conv9_relu = torch.nn.ReLU(1)
        self.conv_final = torch.nn.Conv2d(512, anc_num * (4 + 1 + cls_num), 1, 1, 0, bias=True)

    def forward(self, train_data, **args):
        if isinstance(train_data, tuple):
            x, lab = train_data
        else:
            x = train_data
        x = x.permute([0, 3, 1, 2, ])
        x = self.power(x * 0.00392156885937 + 0.0, 1.0)
        x = self.conv1_relu(self.conv1(x))
        x = self.pool1(x)
        x = self.conv2_relu(self.conv2(x))
        x = self.pool2(x)
        x = self.conv3_1_relu(self.conv3_1(x))
        x = self.conv3_2_relu(self.conv3_2(x))
        x = self.conv3_3_relu(self.conv3_3(x))
        x = self.pool3(x)
        x = self.conv4_1_relu(self.conv4_1(x))
        x = self.conv4_2_relu(self.conv4_2(x))
        x = self.conv4_3_relu(self.conv4_3(x))
        x = self.pool4(x)
        x = self.conv5_1_relu(self.conv5_1(x))
        x = self.conv5_2_relu(self.conv5_2(x))
        x = self.conv5_3_relu(self.conv5_3(x))
        x = self.conv5_4_relu(self.conv5_4(x))
        cat = self.conv5_5_relu(self.conv5_5(x))
        x = self.pool5(cat)
        x = self.conv6_1_relu(self.conv6_1_nodilate(x))
        x = self.conv6_2_relu(self.conv6_2(x))
        x = self.conv6_3_relu(self.conv6_3(x))
        x = self.conv6_4_relu(self.conv6_4(x))
        x = self.conv6_5_relu(self.conv6_5(x))
        x = self.conv7_1_relu(self.conv7_1(x))
        x = self.conv7_2_relu(self.conv7_2(x))
        x = self.concat8((cat, x), 1)
        x = self.conv9_relu(self.conv9(x))
        x = self.conv_final(x)
        net_final = x.permute([0, 2, 3, 1])
        return [net_final, ]

#
# class MyNet(torch.nn.Module):
#     """Apollo yolo net for 2d box."""
#
#     def __init__(self):
#         """Yolo net init."""
#         super(MyNet, self).__init__()
#         self.power = torch.pow
#         self.conv1 = torch.nn.Sequential(OrderedDict([
#             ("conv1", torch.nn.Conv2d(3, 16, 3, 1, 1, 1, bias=True)),  # 384,960
#             ("conv1_relu", torch.nn.ReLU(1)),
#             ("pool1", torch.nn.MaxPool2d(2, 2, 0)),  # 192,480
#             ("conv2", torch.nn.Conv2d(16, 32, 3, 1, 1, 1, bias=True)),  # 192,480
#             ("conv2_relu", torch.nn.ReLU(1)),
#             ("pool2", torch.nn.MaxPool2d(2, 2, 0)),  # 96,240
#             ("conv3_1", torch.nn.Conv2d(32, 64, 3, 1, 1, 1, bias=True)),  # 96,240
#             ("conv3_1_relu", torch.nn.ReLU(1)),
#             ("conv3_2", torch.nn.Conv2d(64, 32, 1, 1, 0, 1, bias=True)),  # 96,240
#             ("conv3_2_relu", torch.nn.ReLU(1)),
#             ("conv3_3", torch.nn.Conv2d(32, 64, 3, 1, 1, 1, bias=True)),  # 96,240
#             ("conv3_3_relu", torch.nn.ReLU(1)),
#             ("pool3", torch.nn.MaxPool2d(2, 2, 0)),  # 48,120
#             ("conv4_1", torch.nn.Conv2d(64, 128, 3, 1, 1, 1, bias=True)),  # 48,120
#             ("conv4_1_relu", torch.nn.ReLU(1)),
#             ("conv4_2", torch.nn.Conv2d(128, 64, 1, 1, 0, 1, bias=True)),  # 48,120
#             ("conv4_2_relu", torch.nn.ReLU(1)),
#             ("conv4_3", torch.nn.Conv2d(64, 128, 3, 1, 1, 1, bias=True)),  # 48,120
#             ("conv4_3_relu", torch.nn.ReLU(1)),
#             ("pool4", torch.nn.MaxPool2d(2, 2, 0)),  # 24,60
#             ("conv5_1", torch.nn.Conv2d(128, 256, 3, 1, 1, 1, bias=True)),  # 24,60
#             ("conv5_1_relu", torch.nn.ReLU(1)),
#             ("conv5_2", torch.nn.Conv2d(256, 128, 1, 1, 0, 1, bias=True)),  # 24,60
#             ("conv5_2_relu", torch.nn.ReLU(1)),
#             ("conv5_3", torch.nn.Conv2d(128, 256, 3, 1, 1, 1, bias=True)),  # 24,60
#             ("conv5_3_relu", torch.nn.ReLU(1)),
#             ("conv5_4", torch.nn.Conv2d(256, 128, 1, 1, 0, 1, bias=True)),  # 24,60
#             ("conv5_4_relu", torch.nn.ReLU(1)),
#             ("conv5_5", torch.nn.Conv2d(128, 256, 3, 1, 1, 1, bias=True)),  # 24,60
#             ("conv5_5_relu", torch.nn.ReLU(1))
#         ]))
#         self.conv2 = torch.nn.Sequential(OrderedDict([
#             ("pool5", torch.nn.MaxPool2d(3, 1, 1)),  # 24,60
#             ("conv6_1_nodilate", torch.nn.Conv2d(256, 512, 5, 1, 2, 1, bias=True)),  # 24,60
#             ("conv6_1_relu", torch.nn.ReLU(1)),
#             ("conv6_2", torch.nn.Conv2d(512, 256, 1, 1, 0, 1, bias=True)),  # 24,60
#             ("conv6_2_relu", torch.nn.ReLU(1)),
#             ("conv6_3", torch.nn.Conv2d(256, 512, 3, 1, 1, 1, bias=True)),  # 24,60
#             ("conv6_3_relu", torch.nn.ReLU(1)),
#             ("conv6_4", torch.nn.Conv2d(512, 256, 1, 1, 0, 1, bias=True)),  # 24,60
#             ("conv6_4_relu", torch.nn.ReLU(1)),
#             ("conv6_5", torch.nn.Conv2d(256, 512, 3, 1, 1, 1, bias=True)),  # 24,60
#             ("conv6_5_relu", torch.nn.ReLU(1)),
#             ("conv7_1", torch.nn.Conv2d(512, 512, 3, 1, 1, 1, bias=True)),  # 24,60
#             ("conv7_1_relu", torch.nn.ReLU(1)),
#             ("conv7_2", torch.nn.Conv2d(512, 512, 3, 1, 1, 1, bias=True)),  # 24,60
#             ("conv7_2_relu", torch.nn.ReLU(1))
#         ]))
#         self.concat8 = torch.cat
#         self.conv3 = torch.nn.Sequential(OrderedDict([
#             ("conv9", torch.nn.Conv2d(768, 512, 3, 1, 1, 1, bias=True)),  # 24,60
#             ("conv9_relu", torch.nn.ReLU(1)),
#             ("conv_final", torch.nn.Conv2d(512, 144, 1, 1, 0, bias=True))  # 24,60
#         ]))
#         self.cls_pred_prob = torch.nn.Softmax(-1)
#         self.obj_pred = torch.nn.Sigmoid()
#
#     def forward(self, x):
#         # pylint: disable=arguments-differ
#         """Yolo Net forward process."""
#         x = x.permute([0, 3, 1, 2, ])
#         x = self.power(x * 0.00392156885937 + 0.0, 1.0)
#         cat = self.conv1(x)
#         x = self.conv2(cat)
#         x = self.concat8((cat, x), 1)
#         conv_final = self.conv3(x)
#         conv_final_permute = conv_final.permute([0, 2, 3, 1])
#         loc_pred, obj_perm, cls_perm = torch.split(conv_final_permute, [64, 16, 64], 3)
#         # obj_perm = self.obj_pred(obj_perm)   # sigmoid might lead to gradient losing.
#         cls_reshape = torch.reshape(cls_perm, (-1, 4))
#         cls_pred_prob = self.cls_pred_prob(cls_reshape)
#         cls_pred = torch.reshape(cls_pred_prob, cls_perm.shape)
#         return obj_perm, cls_pred, loc_pred
#
#     def weights_init(self, layer):
#         # pylint: disable=no-self-use
#         """Init weights."""
#         classname = layer.__class__.__name__
#         if classname.find('Conv') != -1:
#             torch.nn.init.xavier_normal_(layer.weight.data)
#             torch.nn.init.constant_(layer.bias.data, 0.0)
