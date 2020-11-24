import torch
import os
from collections import OrderedDict


def weight16_17():
    weight_path_16 = '../../tmp/checkpoints/lrf300/now.pkl'
    name_17 = os.path.basename(weight_path_16).split('.')[0]
    weight_path_17 = os.path.join(os.path.dirname(weight_path_16), name_17 + '_17.pkl')
    weight_16 = torch.load(weight_path_16)
    torch.save(weight_16, weight_path_17, _use_new_zipfile_serialization=False)


def weight_discard_module():
    checkpoint = '/media/lg/SSD_WorkSpace/LG/OBD/FCOS-PyTorch-37.2AP/checkpoint/voc_77.8.pth'
    name = os.path.basename(checkpoint).split('.')[0]
    new_path = os.path.join(os.path.dirname(checkpoint), name + '_without_module.pkl')
    new_dic = OrderedDict()
    state_dict = torch.load(checkpoint)
    for k, v in state_dict.items():
        if 'module.' == k[:7]:
            k = k.replace('module.', '')
        new_dic[k] = v
    torch.save(new_dic, new_path)
    print('saved to ', new_path)


if __name__ == '__main__':
    weight_discard_module()