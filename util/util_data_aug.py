"""Collection of data augumentation functions."""
import os
import random
import logging

import cv2
import torch
import imgaug as ia
from imgaug import augmenters as iaa
import xml.etree.ElementTree as ET

from util.util_show_img import _show_img
from util.util_get_cls_names import _get_class_names

LOG = logging.getLogger(__name__)


class Dataaug:
    '''
    Data augmentation.
    '''

    def __init__(self, cfg):
        self.cfg = cfg
        self.img_path = cfg.PATH.IMG_PATH
        self.lab_path = cfg.PATH.LAB_PATH
        self.cls2idx = dict(zip(self.cfg.TRAIN.CLASSES, range(len(self.cfg.TRAIN.CLASSES))))
        self.area_ratio = cfg.TRAIN.AREAR_RATIO
        self.min_area = cfg.TRAIN.MIN_AREAR
        self.class_name = _get_class_names(cfg.PATH.CLASSES_PATH)
        self.print_path = False

    def _read_line(self, path, pass_obj=['DontCare', ]):
        """
        Parse the labels from file.
    
        :param pass_obj: pass the labels in the list.
                        e.g. pass_obj=['Others','Pedestrian', 'DontCare']
        :param path: the path of file that need to parse.
        :return:lists of the classes and the key points.
        """
        bbs = []
        if os.path.basename(path).split('.')[-1] == 'txt':
            file_open = open(path, 'r')
            for line in file_open.readlines():
                tmps = line.strip().split(' ')
                if tmps[0] not in self.class_name:
                    continue
                if self.class_name[tmps[0]] in pass_obj:
                    continue
                box_x1 = float(tmps[4])
                box_y1 = float(tmps[5])
                box_x2 = float(tmps[6])
                box_y2 = float(tmps[7])
                bbs.append(ia.BoundingBox(box_x1, box_y1, box_x2, box_y2, label=self.class_name[tmps[0]]))
        elif os.path.basename(path).split('.')[-1] == 'xml':
            tree = ET.parse(path)
            root = tree.getroot()
            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                if cls_name not in self.class_name:
                    continue
                if self.class_name[cls_name] in pass_obj:
                    continue
                bbox = obj.find('bndbox')
                box_x1 = float(bbox.find('xmin').text)
                box_y1 = float(bbox.find('ymin').text)
                box_x2 = float(bbox.find('xmax').text)
                box_y2 = float(bbox.find('ymax').text)
                bbs.append(ia.BoundingBox(box_x1, box_y1, box_x2, box_y2, label=self.class_name[cls_name]))
        return bbs

    def _read_datas(self, idx, image=None):
        '''
        Read images and labels depend on the idx.
        :param image: if there is a image ,the just return the image.
        :return: images, labels, image_size
        '''
        images = []
        labels = []
        if image is None:
            idx_remove = idx.copy()
            for img_idx in idx:
                # labels come first.
                if os.path.isfile(self.lab_path + img_idx + '.txt'):
                    lab_name = img_idx + '.txt'
                else:
                    lab_name = img_idx + '.xml'
                if os.path.isfile(self.img_path + img_idx + '.png'):
                    img_name = img_idx + '.png'
                else:
                    img_name = img_idx + '.jpg'
                if self.print_path:
                    print(img_name, '==>>>', lab_name)
                label = self._read_line(self.lab_path + lab_name)
                while not label:  # whether the label is empty?
                    print('warning: no label...')
                    if img_idx in idx_remove:  # if img_idx has been removed then label is empty
                        idx_remove.remove(img_idx)
                    if not idx_remove: break
                    img_idx = random.choice(idx_remove)
                    label = self._read_line(self.lab_path + lab_name)
                labels.append(label)
                # then add the images.
                img = cv2.imread(self.img_path + img_name)
                if img is not None:
                    image_size = img.shape[:-1]  # the last pic as the shape of a batch.
                    images.append(img)
                else:
                    print('imread img is NONE.')
        else:
            images.append(image)
            image_size = image.shape[:-1]

        if labels == [[]]:
            labels = None
        return images, labels, image_size

    def _parse_bbs(self, bbs_aug, relative=True):
        """
        Parse the key points from augmentation.
    
        :param bbs_aug: key points from augmentation.
        :param relative: the relative location of bounding boxes
        :return:labs are in the shape of [[class, left, top, button, right],..]
        """
        labs = []
        height = bbs_aug.height
        width = bbs_aug.width
        for bb_aug in bbs_aug.bounding_boxes:
            _bb_x1, _bb_y1, _bb_x2, _bb_y2, cls = bb_aug.x1, bb_aug.y1, \
                                                  bb_aug.x2, bb_aug.y2, \
                                                  bb_aug.label
            bb_x1 = min(max(1, _bb_x1), width - 1)
            bb_y1 = min(max(1, _bb_y1), height - 1)
            bb_x2 = min(max(1, _bb_x2), width - 1)
            bb_y2 = min(max(1, _bb_y2), height - 1)
            # calculate the ratio of area
            area_before = (_bb_x2 - _bb_x1) * (_bb_y2 - _bb_y1)
            area_after = (bb_x2 - bb_x1) * (bb_y2 - bb_y1)
            if area_after / area_before < self.area_ratio or area_after < self.min_area:
                continue

            # get the lab
            if relative:
                lab = [self.cls2idx[cls],
                       bb_x1 / width,
                       bb_y1 / height,
                       bb_x2 / width,
                       bb_y2 / height]
            else:
                lab = [self.cls2idx[cls], int(bb_x1), int(bb_y1), int(bb_x2), int(bb_y2)]
            labs.append(lab)
        return labs

    def _augmenting(self, idx=None, do_aug=True, relative=True, resize=None, image_for_aug=None):
        """Create augmentation images from kitti.
    
        :param idx:  the index of kitti images ,
         index is in the shape of [1, 3, 555, 1033...]
        :param do_aug: if do_aug is False , then do nothing about images,and labels.
        :param relative: the relative location of bounding boxes
        :param image_for_aug: if there is a single image,the augment it without labels
        :return: return a np array of images, and return a list about
         labs are in the shape of [[class, left, top, button, right],..]
        """
        # print(idx)
        # prepare the augmentation functions
        images, labels, image_size = self._read_datas(idx, image=image_for_aug)
        if image_for_aug is None and labels is None:  # if there is no any label in classes,the return none.
            return images, labels

        if isinstance(resize, list):
            resize_size = resize
        else:
            resize_size = image_size
        aug_funs = [iaa.Resize({"height": resize_size[0], "width": resize_size[1]})]
        if do_aug:
            # weather_aug = [random.choice([iaa.Snowflakes(flake_size=(0.4, 0.75),
            #                                              speed=(0.001, 0.03)),
            #                               iaa.Fog(),
            #                               iaa.Clouds(),
            #                               ])]  # choose one weather augmentation
            base_funs = [iaa.Fliplr(.5),
                         # iaa.Grayscale(alpha=(0, 1)),
                         # iaa.ChangeColorspace('BGR'),
                         # iaa.GaussianBlur((0, 2)),
                         # iaa.Add((-50, 50)),
                         # iaa.Dropout(0.02, per_channel=0.5),
                         # iaa.GammaContrast(gamma=(0.5, 1.5), per_channel=True),
                         iaa.Affine(scale=(0.9, 1.1), translate_percent=(-.01, 0.01), rotate=(-3, 3))]
            # base_funs += weather_aug
            # random.shuffle(base_funs)
            base_funs = [random.choice(base_funs)]
            aug_funs = base_funs + aug_funs

        # do the augmentation
        seq_det = iaa.Sequential(aug_funs)
        seq_det = seq_det.to_deterministic()
        labels = [ia.BoundingBoxesOnImage(x, shape=images[i].shape) for i, x in enumerate(labels)]
        images = seq_det.augment_images(images)
        bbs_aug = seq_det.augment_bounding_boxes(labels)
        labels = [self._parse_bbs(x, relative) for x in bbs_aug]
        return images, labels

    def _show_imgaug(self, idx, do_aug=True, resize=True, relative=False, show_img=True):
        """
        Show the images with data augmentation.
    
        :param idx: idx of images from kitti.
        :return: show the images.
        """
        images, labels = self.augmentation(idx, do_aug=do_aug, relative=relative, resize=resize)
        images = torch.Tensor(images)
        LOG.debug(labels)
        return _show_img(images, labels, show_img=show_img, cfg=self.cfg)

    def augmentation(self, idx=None, do_aug=True, relative=True, resize=True, for_one_image=None, show_img=False):
        """Create augmentation images from kitti.
    
        :param idx:  the index of kitti images ,
         index is in the shape of [1, 3, 555, 1033...]
        :param do_aug: if do_aug is False , then do nothing about images,and labels,
                        but corp to fixed size.
        :param relative: the relative location of bounding boxes
    
        :return: return a np array of images, and return a list about
                labs are in the shape of [[class, left, top, button, right],..]
        """
        if resize:
            resize = random.choice(self.cfg.TRAIN.MULTI_SIZE_RATIO) * self.cfg.TRAIN.IMG_SIZE
        images, labels = self._augmenting(idx, do_aug, relative, resize=resize, image_for_aug=for_one_image)

        if labels is not None:  # if labels is not none ,then check the label in labels,whether the label is none.
            for i, label in enumerate(labels):
                time = 0
                total_time = 50
                while not label:  # check every label, whether there is a label is empty.
                    print('no label at NO.%s, repeating %d/%d' % (idx[i], time, total_time))
                    time += 1
                    re_aug_idx = [idx[i]] if time < total_time else [random.choice(idx)]
                    image, label = self._augmenting(re_aug_idx, do_aug, relative, resize=resize,
                                                    image_for_aug=for_one_image)
                    label = label[0]
                    if label:
                        images[i] = image[0]
                        labels[i] = label
        if show_img:
            _show_img(images, labels, show_img=True, cfg=self.cfg)
        return images, labels
