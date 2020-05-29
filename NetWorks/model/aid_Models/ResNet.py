import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class ResNet(nn.Module):
    def __init__(self, type='50', pretrained=True):
        super(ResNet, self).__init__()
        if type == '50':
            self.resnet_model = resnet50(pretrained=pretrained)
        elif type == '101':
            ...
        # TODO:。。。
        self.backbone = self.resnet_model[:-1]

    def forward(self, x):
        x = self.backbone(x)
        return x


if __name__ == '__main__':
    res = ResNet(pretrained=False)
    input = torch.rand((4, 3, 200, 200))
    out = res(input)

    a = 0
