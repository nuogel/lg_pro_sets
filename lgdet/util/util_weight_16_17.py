import torch
import os

weight_path_16 = '../../tmp/checkpoints/efficientdet/now.pkl'
weight_path_17 = os.path.join(os.path.dirname(weight_path_16), 'trans_weight_17.pkl')
torch.save(torch.load(weight_path_16), weight_path_17, _use_new_zipfile_serialization=False)
