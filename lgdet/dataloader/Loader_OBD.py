import os
import torch
import random
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from util.util_get_cls_names import _get_class_names
from util.util_show_img import _show_img
from torch.utils.data import DataLoader
from ..registry import DATALOADERS
from util.util_lg_transformer import LgTransformer
import lmdb
import pickle


@DATALOADERS.registry()
class OBD_Loader(DataLoader):
    def __init__(self, cfg, dataset, is_training):
        super(OBD_Loader, self).__init__(object)
        self.cfg = cfg
        self.dataset_txt = dataset
        self.is_training = is_training
        self.one_test = cfg.TEST.ONE_TEST
        self.one_name = cfg.TEST.ONE_NAME
        self.train_batch_num = 100
        self.test_batch_num = 1
        self.class_name = _get_class_names(cfg.PATH.CLASSES_PATH)
        self.print_path = self.cfg.TRAIN.SHOW_TRAIN_NAMES
        self.cls2idx = dict(zip(cfg.TRAIN.CLASSES, range(cfg.TRAIN.CLASSES_NUM)))
        self.write_images = self.cfg.TRAIN.WRITE_IMAGES
        self.lgtransformer = LgTransformer(self.cfg)
        if self.cfg.TRAIN.USE_LMDB:
            mod = 'train' if self.is_training else 'val'
            lmdb_path = self.cfg.PATH.LMDB_PATH.format(mod)
            env = lmdb.open(lmdb_path, max_dbs=2, lock=False)
            db_image = env.open_db('image'.encode())
            db_label = env.open_db('label'.encode())
            self.txn_image = env.begin(write=False, db=db_image)
            self.txn_label = env.begin(write=False, db=db_label)

    def __len__(self):
        if self.one_test:
            if self.is_training:
                length = int(self.cfg.TEST.ONE_TEST_TRAIN_STEP)
            else:
                length = 1
        else:
            length = len(self.dataset_txt)
        return length

    def __getitem__(self, index):
        img, label, data_info = self._load_gt(index)
        # DOAUG:
        if self.cfg.TRAIN.DO_AUG and self.is_training:
            img, label = self.lgtransformer.data_aug(img, label)

        # PAD TO SIZE:
        if (self.cfg.TRAIN.PADTOSIZE and self.is_training) or (self.cfg.TEST.PADTOSIZE and not self.is_training):
            img, label = self.lgtransformer.pad_to_size(img, label, self.cfg.TRAIN.IMG_SIZE, decode=False)

        # resize
        if (self.cfg.TRAIN.RESIZE and self.is_training) or (self.cfg.TEST.RESIZE and not self.is_training):
            img, label = self.lgtransformer.resize(img, label, self.cfg.TRAIN.IMG_SIZE)

        if self.cfg.TRAIN.RELATIVE_LABELS:
            img, label = self.lgtransformer.relative_label(img, label)

        img, label = self.lgtransformer.transpose(img, label)

        if self.cfg.TRAIN.SHOW_INPUT > 0:
            _img = img.copy()
            _label = label.clone()
            show_img = self.lgtransformer.imdenormalize(_img, self.cfg.mean, self.cfg.std, to_bgr=True)
            _show_img(show_img, _label.numpy(), cfg=self.cfg, show_time=self.cfg.TRAIN.SHOW_INPUT, pic_path=data_info[2])

        if self.write_images > 0 and self.is_training:
            _img = img.copy()
            _label = label.clone()
            show_img = self.lgtransformer.imdenormalize(_img, self.cfg.mean, self.cfg.std, to_bgr=True)
            img_write = _show_img(show_img, _label.numpy(), cfg=self.cfg, show_time=-1)
            self.cfg.writer.tbX_addImage('GT_' + data_info[0], img_write)
            self.write_images -= 1

        return img, label, data_info  # only need the labels  label_after[x1y1x2y2]

    def _set_group_flag(self):  # LG: this is important
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _load_gt(self, index):
        img = None
        label = None

        while img is None or label is None:  # if there is no data in img or label
            if self.one_test:
                data_info = self.dataset_txt[0]
            else:
                data_info = self.dataset_txt[index]

            data_info[1] = os.path.join(self.cfg.PATH.INPUT_PATH, data_info[1])
            data_info[2] = os.path.join(self.cfg.PATH.INPUT_PATH, data_info[2])
            img, label = self._read_datas(data_info)  # labels: (x1, y1, x2, y2) & must be absolutely labels.
            index = random.randint(1, len(self.dataset_txt) - 1)

            if not self.is_training and not label:  # for test
                break
        return img, label, data_info

    def _add_dataset(self, dataset, is_training):
        self.dataset_txt = dataset
        self.is_training = is_training

    def pre_load_data_infos(self):
        if 'COCO' in self.cfg.TRAIN.TRAIN_DATA_FROM_FILE:
            from util.util_load_coco import COCODataset
            if self.is_training:
                annfile = os.path.join(self.cfg.PATH.INPUT_PATH, 'annotations/instances_train2017.json')
            else:
                annfile = os.path.join(self.cfg.PATH.INPUT_PATH, 'annotations/instances_val2017.json')
            self.coco = COCODataset(ann_file=annfile, cfg=self.cfg)

            self.data_infos = []
            self.flag = np.zeros(len(self), dtype=np.uint8)
            for i, datainfo in enumerate(self.dataset_txt):
                img_info, ann_info = self.coco.prepare_data(datainfo[0])
                bboxes = self._read_labels(datainfo[2], datainfo)
                self.data_infos.append({'img_name': datainfo[0],
                                        'img_path': datainfo[1],
                                        'lab_path': datainfo[2],
                                        'img_info': img_info,
                                        'ann_info': ann_info,
                                        'bboxes': bboxes})
                if img_info['width'] / img_info['height'] > 1:
                    self.flag[i] = 1
        else:
            self.data_infos = []
            for datainfo in self.dataset_txt:
                self.data_infos.append({'imgname': datainfo[0],
                                        'imgpath': datainfo[1],
                                        'labpath': datainfo[2]})

    def _read_datas(self, data_info):
        '''
        Read images and labels depend on the idx.
        :param image: if there is a image ,the just return the image.
        :return: images, labels, image_size
        '''
        if self.cfg.TRAIN.USE_LMDB:
            img_name = data_info[0]
            try:
                image_bin = self.txn_image.get(img_name.encode())
                label_bin = self.txn_label.get(img_name.encode())
            except:
                print(data_info, 'faild in lmdb loader')
            image_buf = np.frombuffer(image_bin, dtype=np.uint8)
            img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)

            label_infos = pickle.loads(label_bin)
            bboxes = label_infos['bboxes']
            cls_names = label_infos['cls_names']
            label = []
            for i, bbx in enumerate(bboxes):
                cls_name = cls_names[i]
                if self.class_name[cls_name] not in self.class_name:
                    continue
                label.append([self.cls2idx[self.class_name[cls_name]], bbx[0], bbx[1], bbx[2], bbx[3]])
        else:
            x_path = data_info[1]
            y_path = data_info[2]
            if self.print_path:
                print(x_path, '<--->', y_path)
            if not (os.path.isfile(x_path) and os.path.isfile(y_path)):
                print('ERROR, NO SUCH A FILE.', x_path, '<--->', y_path)
                exit()
            # labels come first.
            label = self._load_labels(y_path, data_info=data_info)
            if label == [[]] or label == []:
                print('loader obd :none label at:', y_path)
                label = None
            # then add the images.
            img = cv2.imread(x_path)

        return img, label

    def _is_finedata(self, xyxy):
        '''
        area < min ?
        :param xyxy:
        :return:
        '''
        x1, y1, x2, y2 = xyxy
        for point in xyxy:
            if point < 0.: return False
        if x2 - x1 <= 0.: return False
        if y2 - y1 <= 0.: return False
        if (x2 - x1) * (y2 - y1) < self.cfg.TRAIN.MIN_AREA: return False
        return True

    def _load_labels(self, path, data_info=None, predicted_line=False, pass_obj=['DontCare', ]):
        """
        Parse the labels from file.

        :param pass_obj: pass the labels in the list.
                        e.g. pass_obj=['Others','Pedestrian', 'DontCare']
        :param path: the path of file that need to parse.
        :return:lists of the classes and the key points============ [x1, y1, x2, y2].
        """
        bbs = []

        if os.path.basename(path).split('.')[-1] == 'txt' and not predicted_line:
            file_open = open(path, 'r')
            for line in file_open.readlines():
                if 'UCAS_AOD' in self.cfg.TRAIN.TRAIN_DATA_FROM_FILE:
                    tmps = line.strip().split('\t')
                    box_x1 = float(tmps[9])
                    box_y1 = float(tmps[10])
                    box_x2 = box_x1 + float(tmps[11])
                    box_y2 = box_y1 + float(tmps[12])
                    if not self._is_finedata([box_x1, box_y1, box_x2, box_y2]): continue
                    bbs.append([1, box_x1, box_y1, box_x2, box_y2])

                elif 'VisDrone2019' in self.cfg.TRAIN.TRAIN_DATA_FROM_FILE:
                    name_dict = {'0': 'ignored regions', '1': 'pedestrian', '2': 'people',
                                 '3': 'bicycle', '4': 'car', '5': 'van', '6': 'truck',
                                 '7': 'tricycle', '8': 'awning-tricycle', '9': 'bus',
                                 '10': 'motor', '11': 'others'}
                    tmps = line.strip().split(',')
                    realname = name_dict[tmps[5]]
                    if realname not in self.class_name:
                        continue
                    if realname in pass_obj:
                        continue
                    box_x1 = float(tmps[0])
                    box_y1 = float(tmps[1])
                    box_x2 = box_x1 + float(tmps[2])
                    box_y2 = box_y1 + float(tmps[3])

                    if not self._is_finedata([box_x1, box_y1, box_x2, box_y2]): continue
                    bbs.append([self.cls2idx[self.class_name[realname]], box_x1, box_y1, box_x2, box_y2])
                elif 'TS02' in self.cfg.TRAIN.TRAIN_DATA_FROM_FILE:
                    tmps = line.strip().split(' ')
                    realname = 'car'
                    if realname not in self.class_name:
                        continue
                    if realname in pass_obj:
                        continue
                    img_w = 1920
                    img_h = 1080
                    x = float(tmps[1]) * img_w
                    y = float(tmps[2]) * img_h
                    w = float(tmps[3]) * img_w
                    h = float(tmps[4]) * img_h

                    box_x1 = x - w / 2.
                    box_y1 = y - h / 2.
                    box_x2 = x + w / 2.
                    box_y2 = y + h / 2.

                    if not self._is_finedata([box_x1, box_y1, box_x2, box_y2]): continue
                    bbs.append([self.cls2idx[self.class_name[realname]], box_x1, box_y1, box_x2, box_y2])
                else:
                    tmps = line.strip().split(' ')
                    if tmps[0] not in self.class_name:
                        continue
                    if self.class_name[tmps[0]] in pass_obj:
                        continue
                    box_x1 = float(tmps[4])
                    box_y1 = float(tmps[5])
                    box_x2 = float(tmps[6])
                    box_y2 = float(tmps[7])
                    if not self._is_finedata([box_x1, box_y1, box_x2, box_y2]): continue
                    bbs.append([self.cls2idx[self.class_name[tmps[0]]], box_x1, box_y1, box_x2, box_y2])
        elif os.path.basename(path).split('.')[-1] == 'xml' and not predicted_line:
            tree = ET.parse(path)
            root = tree.getroot()
            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                if cls_name not in self.class_name:
                    print(cls_name, 'is passed')
                    continue
                if self.class_name[cls_name] in pass_obj:
                    continue
                bbox = obj.find('bndbox')
                box_x1 = float(bbox.find('xmin').text)
                box_y1 = float(bbox.find('ymin').text)
                box_x2 = float(bbox.find('xmax').text)
                box_y2 = float(bbox.find('ymax').text)
                if not self._is_finedata([box_x1, box_y1, box_x2, box_y2]): continue
                bbs.append([self.cls2idx[self.class_name[cls_name]], box_x1, box_y1, box_x2, box_y2])

        elif os.path.basename(path).split('.')[-1] == 'txt' and predicted_line:
            f_path = open(path, 'r')
            print(path)
            for line in f_path.readlines():
                tmp = line.split()
                cls_name = tmp[0]
                bbs.append([float(tmp[1]), self.cls2idx[self.class_name[cls_name]],
                            [float(tmp[2]), float(tmp[3]), float(tmp[4]), float(tmp[5])]])

        elif 'COCO' in self.cfg.TRAIN.TRAIN_DATA_FROM_FILE:
            img_name = data_info[0].strip()
            img_info, ann_info = self.coco.prepare_data(img_name)
            for i, bbx in enumerate(ann_info['bboxes']):
                cls_name = ann_info['labels'][i]
                if cls_name not in self.class_name:
                    print(cls_name, 'is passed')
                    continue
                if self.class_name[cls_name] in pass_obj:
                    continue
                if not self._is_finedata(bbx): continue
                bbs.append([self.cls2idx[self.class_name[cls_name]], bbx[0], bbx[1], bbx[2], bbx[3]])

        return bbs

    def collate_fun(self, batch):
        '''
        collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
        其中default_collate会将labels分割合并转换成tensor。
        !!!***if not use my own collect_fun ,the labels will be wrong orders.***
        :param batch:
        :return:
        '''
        imgs, labels, infos = zip(*batch)
        imgs = torch.from_numpy(np.asarray(imgs))
        imgs = imgs.permute(0, 3, 1, 2)
        if isinstance(labels[0], torch.Tensor):
            for i, label in enumerate(labels):
                try:
                    label[:, 0] = i
                except:
                    print(infos[i])
            try:
                labels = torch.cat(labels, 0)
            except:
                print(labels, infos)
        return imgs, labels, list(infos)
