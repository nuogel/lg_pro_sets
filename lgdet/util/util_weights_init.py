import torch
import torch.nn

'''
weights initiation is so important to a model, it decide where you start to downing the weights, also decide the time and the results of your model.
I suddenly find out that the loss of super resolution is so big by use the initiation of object detection which is kaiming initiation, so wrong ,so bad .after several weeks 
I found that the mistake is the weights initiation. ——LuoGeng 2020.02.27.
'''


def weights_init(Modle, cfg):
    if cfg.TRAIN.MODEL in ['SRDN', 'srdn']:
        torch.manual_seed(123)
    else:
        for m in Modle.modules():
            if isinstance(m, torch.nn.Conv2d):
                weight = m.weight
                torch.nn.init.kaiming_normal_(weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.RNN) or isinstance(m, torch.nn.LSTM) or isinstance(m, torch.nn.GRU):
                torch.nn.init.kaiming_normal_(m.weight_ih_l0, a=0, mode='fan_in', nonlinearity='leaky_relu')
                torch.nn.init.kaiming_normal_(m.weight_hh_l0, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias_ih_l0 is not None:
                    m.bias_ih_l0.data.zero_()
                if m.bias_hh_l0 is not None:
                    m.bias_hh_l0.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.GroupNorm) or isinstance(m, torch.nn.SyncBatchNorm):
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
    # return Modle