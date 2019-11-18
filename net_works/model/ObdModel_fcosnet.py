import torch.nn as nn
import torch
import math


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet([3, 4, 6, 3], **kwargs) # original model
    model = ResNet([1, 1, 2, 2, 2, 1], **kwargs)

    if pretrained:
        print('please add pretrained weight.')
        # model.load_state_dict(torch.load(model.modelPath))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet([3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model.modelPath))
    return model


class ResNet(nn.Module):
    """
    block: A sub module
    """

    def __init__(self, layers, num_classes=1000, model_path="model.pkl"):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.modelPath = model_path
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stack1 = self.make_stack(64, layers[0])
        self.stack2 = self.make_stack(128, layers[1], stride=2)
        self.stack3 = self.make_stack(256, layers[2], stride=2)
        self.stack4 = self.make_stack(512, layers[3], stride=2)
        self.stack5 = self.make_stack(512, layers[4], stride=2)
        self.stack6 = self.make_stack(512, layers[5], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        # initialize parameters
        # self.init_param()

    # def init_param(self):
    #     # The following is initialization
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             n = m.weight.shape[0] * m.weight.shape[1]
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             m.bias.data.zero_()

    def make_stack(self, planes, blocks, stride=1):
        downsample = None
        layers = []

        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):  # [N ,3, 384, 960] 800 1024
        x = self.conv1(x)  # [N ,3, 192, 480] 400 512
        x = self.bn1(x)  # [N ,3, 192, 480] 400 512
        # print(self.bn1.running_mean)
        # print(self.bn1.running_var)
        x = self.relu(x)  # [N ,64, 192, 480]400 512
        x = self.maxpool(x)  # [N ,64, 96, 240]200 256

        x = self.stack1(x)  # [N ,256, 96, 240]200 256
        x = self.stack2(x)  # [N ,512, 48, 120]100 128
        c3 = x
        x = self.stack3(x)  # [N ,1024, 24, 60]50 64
        c4 = x
        x = self.stack4(x)  # [N ,2048, 12, 30]25 32
        c5 = x
        x = self.stack5(x)  # [N ,2048, 12, 30]25 32
        c6 = x
        x = self.stack6(x)  # [N ,2048, 12, 30]25 32
        c7 = x
        return c3, c4, c5, c6, c7


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 1, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.resnet50 = resnet50()
        self.c5upsamping = nn.Upsample(scale_factor=2, mode='bilinear')
        self.p54 = self.conv2d_bn_relu(768, 256)
        self.c4upsamping = nn.Upsample(scale_factor=2, mode='bilinear')
        self.p43 = self.conv2d_bn_relu(384, 128)

    def conv2d_bn_relu(self, in_channel, out_channel):
        return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1),
                              nn.BatchNorm2d(out_channel),
                              nn.ReLU(inplace=True))
    
    def forward(self, x):
        c3, c4, p5, p6, p7 = self.resnet50(x)
        # TODO: add the upsamping codes.
        up54 = self.c5upsamping(p5)
        p54_cat = torch.cat([up54, c4], 1)
        p4 = self.p54(p54_cat)
        
        up43 = self.c4upsamping(p4)
        p43_cat = torch.cat([up43, c3], 1)
        p3 = self.p43(p43_cat)

        return p3, p4, p5, p6, p7


class FCOS(nn.Module):
    def __init__(self):
        super(FCOS, self).__init__()
        self.power = torch.pow
        self.fpn = FPN()

        # never use a list to contain functions.

        # self.cls_shape = [nn.Conv2d(128, 256, kernel_size=1),
        #                          nn.Conv2d(256, 256, kernel_size=1),
        #                          nn.Conv2d(512, 256, kernel_size=1),
        #                          nn.Conv2d(512, 256, kernel_size=1),
        #                          nn.Conv2d(512, 256, kernel_size=1)]
        # self.loc_shape = [nn.Conv2d(128, 256, kernel_size=1),
        #                   nn.Conv2d(256, 256, kernel_size=1),
        #                   nn.Conv2d(512, 256, kernel_size=1),
        #                   nn.Conv2d(512, 256, kernel_size=1),
        #                   nn.Conv2d(512, 256, kernel_size=1),]
        self.cls_shape0 = self.in_256_out_cnn(128, 4)
        self.cls_shape1 = self.in_256_out_cnn(256, 4)
        self.cls_shape2 = self.in_256_out_cnn(512, 4)
        self.cls_shape3 = self.in_256_out_cnn(512, 4)
        self.cls_shape4 = self.in_256_out_cnn(512, 4)
        self.cls_shape = [self.cls_shape0,
                          self.cls_shape1,
                          self.cls_shape2,
                          self.cls_shape3,
                          self.cls_shape4]

        self.center_shape0 = self.in_256_out_cnn(128, 1)
        self.center_shape1 = self.in_256_out_cnn(256, 1)
        self.center_shape2 = self.in_256_out_cnn(512, 1)
        self.center_shape3 = self.in_256_out_cnn(512, 1)
        self.center_shape4 = self.in_256_out_cnn(512, 1)
        self.center_shape = [self.center_shape0,
                             self.center_shape1,
                             self.center_shape2,
                             self.center_shape3,
                             self.center_shape4]

        self.loc_shape0 = self.in_256_out_cnn(128, 4)
        self.loc_shape1 = self.in_256_out_cnn(256, 4)
        self.loc_shape2 = self.in_256_out_cnn(512, 4)
        self.loc_shape3 = self.in_256_out_cnn(512, 4)
        self.loc_shape4 = self.in_256_out_cnn(512, 4)
        self.loc_shape = [self.loc_shape0,
                          self.loc_shape1,
                          self.loc_shape2,
                          self.loc_shape3,
                          self.loc_shape4]

    def in_256_out_cnn(self, in_channel, out_channel):
        return nn.Sequential(nn.Conv2d(in_channel, 256, kernel_size=1),
                              nn.BatchNorm2d(256),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(256, out_channel, kernel_size=1),
                             )


    def forward(self, x):
        x = x.permute([0, 3, 1, 2, ])
        x = self.power(x * 0.00392156885937 + 0.0, 1.0)
        x = self.fpn(x)
        feature_all = []
        for i in range(len(x)):
            feature_one = []
            x_i = x[i]
            x_cls = self.cls_shape[i](x_i)  #
            feature_one.append(x_cls)

            x_center = self.center_shape[i](x_i)
            feature_one.append(x_center)

            x_loc = self.loc_shape[i](x_i)
            feature_one.append(x_loc)

            feature_all.append(feature_one)
        return feature_all

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # if m.bias is not None:
                #     m.bias.data.zero_()
                weight = m.weight
                torch.nn.init.kaiming_normal_(weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == "__main__":
    for i in range(100):
        img = torch.rand((2, 800, 1024, 3)).cuda()
        model = FCOS().cuda()
        x_cls, x_center, x_loc = model.forward(img)
        print(x_cls.shape, x_center.shape, x_loc.shape)
