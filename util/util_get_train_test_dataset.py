import os
import glob
from sklearn.model_selection import train_test_split
import torch
from util.util_get_cls_names import _get_class_names
import xml.etree.ElementTree as ET


def _get_train_test_dataset(x_dir, y_dir, idx_stores_dir, test_train_ratio, cfg):
    label_files = glob.glob('{}/*'.format(y_dir))
    x_files = glob.glob('{}/*'.format(x_dir))
    label_files_list = []
    x_extend_name = os.path.basename(x_files[0]).split('.')[-1]
    y_extend_name = os.path.basename(label_files[0]).split('.')[-1]

    is_name_in_dict = True  # if the image has this class in it ?
    if cfg.BELONGS == 'OBD':
        if is_name_in_dict:
            class_map = _get_class_names(cfg.PATH.CLASSES_PATH)
            for label_file in label_files:
                if y_extend_name == 'txt':
                    f = open(label_file, 'r')
                    # print('checking file %s about:%s' % (label_file, cfg.TRAIN.CLASSES))
                    for line in f.readlines():
                        name = line.split(' ')[0]
                        if name not in class_map:
                            continue
                        if class_map[name] in cfg.TRAIN.CLASSES:
                            label_files_list.append(label_file)
                            break
                elif y_extend_name == 'xml':
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

    data_idx = sorted(list(set([str(os.path.basename(x).split('.')[0]) for x in label_files_list])))
    for idx in data_idx:
        if not os.path.isfile(os.path.join(x_dir, idx + '.'+str(x_extend_name))):
            data_idx = data_idx.remove(idx)

    assert len(data_idx) >= 1, 'No data found!'

    train_set, test_set = train_test_split(data_idx, test_size=test_train_ratio, random_state=1)
    os.makedirs(idx_stores_dir, exist_ok=True)
    _wrte_dataset_txt((train_set, test_set), idx_stores_dir, [x_dir, x_extend_name, y_dir, y_extend_name])
    # torch.save(train_set, os.path.join(idx_stores_dir, 'train_set'))
    # torch.save(test_set, os.path.join(idx_stores_dir, 'test_set'))
    print('saving train_set&test_set to %s' % idx_stores_dir)

    return train_set, test_set


def _wrte_dataset_txt(dataset, idx_stores_dir, path_info):
    train_set, test_set = dataset
    train_set_txt = ''
    test_set_txt = ''
    for i in train_set:
        train_set_txt += str(i) + ';' + os.path.join(path_info[0], str(i) + '.' + str(path_info[1])) + ';' + os.path.join(path_info[2], str(i) + '.' + str(path_info[3])) + '\n'
    for i in test_set:
        test_set_txt += str(i) + ';' + os.path.join(path_info[0], str(i) + '.' + str(path_info[1])) + ';' + os.path.join(path_info[2], str(i) + '.' + str(path_info[3])) + '\n'
    f = open(os.path.join(idx_stores_dir, 'train_set.txt'), 'w')
    f.write(train_set_txt)
    f.close()
    f = open(os.path.join(idx_stores_dir, 'test_set.txt'), 'w')
    f.write(test_set_txt)
    f.close()


def _read_train_test_dataset(idx_stores_dir):
    print('reading train_set&test_set from %s' % idx_stores_dir)
    f = open(os.path.join(idx_stores_dir, 'train_set.txt'), 'r')
    train_set = [line.split(';')[0] for line in f.readlines()]
    f = open(os.path.join(idx_stores_dir, 'test_set.txt'), 'r')
    test_set = [line.split(';')[0] for line in f.readlines()]
    return train_set, test_set


if __name__ == '__main__':
    y_dir = 'E://datasets//kitti//training//label_2//'
    idx_stores_dir = 'E://LG//programs//lg_pro_sets//tmp//'
    test_train_ratio = 0.1
    catgeorys = ['Car', ]
    _get_train_test_dataset(y_dir, idx_stores_dir, test_train_ratio)
