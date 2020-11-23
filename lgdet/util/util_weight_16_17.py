import torch
import os

weight_path_16 = '../../tmp/checkpoints/lrf300/now.pkl'
name_17 = os.path.basename(weight_path_16).split('.')[0]
weight_path_17 = os.path.join(os.path.dirname(weight_path_16), name_17+'_17.pkl')
weight_16 = torch.load(weight_path_16)
torch.save(weight_16, weight_path_17, _use_new_zipfile_serialization=False)
