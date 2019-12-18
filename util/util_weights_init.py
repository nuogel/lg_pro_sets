import torch
import torch.nn

def weights_init(Modle):
    for m in Modle.modules():
        if isinstance(m, torch.nn.Conv2d):
            # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # m.weight.data.normal_(0, math.sqrt(2. / n))
            # if m.bias is not None:
            #     m.bias.data.zero_()
            weight = m.weight
            torch.nn.init.kaiming_normal_(weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.RNN) or isinstance(m, torch.nn.LSTM)or isinstance(m, torch.nn.GRU):
            torch.nn.init.kaiming_normal_(m.weight_ih_l0, a=0, mode='fan_in', nonlinearity='leaky_relu')
            torch.nn.init.kaiming_normal_(m.weight_hh_l0, a=0, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias_ih_l0 is not None:
                m.bias_ih_l0.data.zero_()
            if m.bias_hh_l0 is not None:
                m.bias_hh_l0.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
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
