import torch
from matplotlib import pyplot as plt
import numpy as np

path = '../../tmp/checkpoint/190626_175_lg_anchors16.pkl'
weight = torch.load(path)

v_all = 0
lt_num_all = 0
cat_array=[]
for k, v in weight.items()[:2]:
    show_one_lay=False
    lt_num = 0
    lt_thresh = 0.0001
    v_array = v.reshape(-1)
    cat_array.append(v_array)
    lt_num += sum(abs(v_array) < lt_thresh)
    lt_num_all+=lt_num
    idx = np.arange(len(v_array))
    v_all += len(idx)
    # to_ = -1
    # plt.bar(idx[:to_], v_array[:to_])
    print(k,'value num:',len(v_array),'; less than',lt_thresh,'is:',lt_num, '; the less persent is:', lt_num/len(v_array))
# plt.show()
out = torch.stack(cat_array)
print(lt_num_all, v_all, 'the less persent is:', lt_num_all/v_all)
