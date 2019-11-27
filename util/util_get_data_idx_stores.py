import os
import glob
from sklearn.model_selection import train_test_split
import torch
from util.util_get_cls_names import _get_class_names
import xml.etree.ElementTree as ET


def _get_data_idx_stores(lab_dir, idx_stores_dir, test_train_ratio, cfg):
    label_files = glob.glob('{}/*'.format(lab_dir))
    label_files_list = []
    is_name_in_dict = True  # if the image has this class in it ?
    if cfg.BELONGS == 'OBD':
        if is_name_in_dict:
            class_map = _get_class_names(cfg.PATH.CLASSES_PATH)
            for label_file in label_files:
                if os.path.basename(label_file).split('.')[-1] == 'txt':
                    f = open(label_file, 'r')
                    # print('checking file %s about:%s' % (label_file, cfg.TRAIN.CLASSES))
                    for line in f.readlines():
                        name = line.split(' ')
                        if name[0] not in class_map:
                            continue
                        if class_map[name[0]] in cfg.TRAIN.CLASSES:
                            label_files_list.append(label_file)
                            break
                elif os.path.basename(label_file).split('.')[-1] == 'xml':
                    tree = ET.parse(label_file)
                    root = tree.getroot()
                    for object in root.findall('object'):
                        name = object.find('name').text
                        if name not in class_map:
                            continue
                        if class_map[name] in cfg.TRAIN.CLASSES:
                            label_files_list.append(label_file)
                            break
        else:
            label_files_list = label_files
    else:
        label_files_list = label_files

    data_idx = list(set([str(os.path.basename(x).split('.')[0]) for x in label_files_list]))
    assert len(data_idx) >= 1, 'No data found!'

    train_set, test_set = train_test_split(data_idx, test_size=test_train_ratio, random_state=1)
    os.makedirs(idx_stores_dir, exist_ok=True)
    torch.save(train_set, os.path.join(idx_stores_dir, 'train_set'))
    torch.save(test_set, os.path.join(idx_stores_dir, 'test_set'))
    print('saving train_set&test_set to %s' % idx_stores_dir)

    return train_set, test_set


if __name__ == '__main__':
    lab_dir = 'E://datasets//kitti//training//label_2//'
    idx_stores_dir = 'E://LG//programs//lg_pro_sets//tmp//'
    test_train_ratio = 0.1
    catgeorys = ['Car', ]
    _get_data_idx_stores(lab_dir, idx_stores_dir, test_train_ratio)
