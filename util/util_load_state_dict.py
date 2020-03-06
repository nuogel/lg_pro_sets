import torch
from collections import OrderedDict


def load_state_dict(model, checkpoint, device):
    new_dic = OrderedDict()
    state_dict = torch.load(checkpoint, map_location=device)
    for k, v in state_dict.items():
        if 'module.' == k[:8]:
            k = k.replace('module.', '')
        new_dic[k] = v
    model.load_state_dict(new_dic)
    return model
