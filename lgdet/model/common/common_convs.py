import torch.nn as nn
import torch
from .activation_funs import *


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


'''
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
'''


class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class Conv2D_Norm_Activation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='mish', norm_type='BN', num_groups=32, bias=False):
        super(Conv2D_Norm_Activation, self).__init__()
        pad = (kernel_size - 1) // 2  # kernel size is 3 or 0
        self.sigmoid = False
        assert norm_type in (None, 'BN', 'GN'), 'norm type just support BN or GN'
        self.norm_type = norm_type
        self.darknetConv = nn.ModuleList()
        self.darknetConv.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias))
        if norm_type == 'BN':
            self.darknetConv.add_module('bn', nn.BatchNorm2d(out_channels))
        elif norm_type == 'GN':
            if not isinstance(num_groups, int):
                num_groups = 32
            self.darknetConv.add_module('gn', nn.GroupNorm(out_channels, num_groups=num_groups))
        if activation == 'relu':
            self.darknetConv.add_module('relu', nn.ReLU(inplace=True))
        elif activation == 'leaky':
            self.darknetConv.add_module('leaky', nn.LeakyReLU(0.1, inplace=True))
        elif activation == 'swish':
            self.darknetConv.add_module('swish', Swish())
        elif activation == 'mish':
            self.darknetConv.add_module('mish', Mish())
        elif activation == 'logistic':
            self.darknetConv.add_module('logistic', torch.nn.Sigmoid())
        else:
            pass

    def forward(self, x):
        for dc in self.darknetConv:
            x = dc(x)
        return x


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, with_r=False, activation='leaky', norm_type='BN', num_groups=None, coord_conv=True):
        super().__init__()
        self.coord_conv = coord_conv
        if self.coord_conv:
            self.addcoords = AddCoords(with_r=with_r)
            # in_size = in_channels
            in_channels += 2
            if with_r:
                in_channels += 1
        self.conv = Conv2D_Norm_Activation(in_channels, out_channels, kernel_size=kernel_size, activation=activation, norm_type=norm_type, num_groups=num_groups)

    def forward(self, x):
        if self.coord_conv:
            x = self.addcoords(x)
        # ret = x
        x = self.conv(x)
        return x


if __name__ == '__main__':
    x = torch.ones([4, 24, 200, 200])
    Add = AddCoords()
    y= Add(x)