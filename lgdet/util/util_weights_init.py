import torch
import torch.nn
import numpy as np
import math

'''
weights initiation is so important to a model, it decide where you start to downing the weights, also decide the time and the results of your model.
I suddenly find out that the loss of super resolution is so big by use the initiation of object detection which is kaiming initiation, so wrong ,so bad .after several weeks 
I found that the mistake is the weights initiation. ——LuoGeng 2020.02.27.
'''


def weights_init(Modle, cfg):
    if cfg.TRAIN.MODEL in ['SRDN', 'srdn', 'retinanet', ]:
        torch.manual_seed(123)
    else:
        for name, m in Modle.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                # if 'conv_list' in name or "header" in name:
                variance_scaling_(m.weight.data)
                # else:
                #     torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

                if m.bias is not None:
                    if "classifier.header" in name:
                        bias_value = -np.log((1 - 0.01) / 0.01)
                        torch.nn.init.constant_(m.bias, bias_value)
                    else:
                        m.bias.data.zero_()

            elif isinstance(m, torch.nn.RNN) or isinstance(m, torch.nn.LSTM) or isinstance(m, torch.nn.GRU):
                torch.nn.init.kaiming_normal_(m.weight_ih_l0, a=0, mode='fan_in', nonlinearity='leaky_relu')
                torch.nn.init.kaiming_normal_(m.weight_hh_l0, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias_ih_l0 is not None:
                    m.bias_ih_l0.data.zero_()
                if m.bias_hh_l0 is not None:
                    m.bias_hh_l0.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.GroupNorm) or isinstance(m, torch.nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()


def variance_scaling_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""
    initializer for SeparableConv in Regressor/Classifier
    reference: https://keras.io/zh/initializers/  VarianceScaling
    """
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(gain / float(fan_in))

    return torch.nn.init._no_grad_normal_(tensor, 0., std)
