import os
import torch
from util.util_Save_Parmeters import TxbLogger
from util.util_logger import load_logger
import numpy as np


def prepare_cfg(cfg, args, is_training=True):
    torch.backends.cudnn.benchmark = True
    os.makedirs(cfg.PATH.TMP_PATH, exist_ok=True)

    print('torch version: ', torch.__version__)
    print('torch.version.cuda: ', torch.version.cuda)

    cfg = common_cfg(cfg)

    cfg.writer = TxbLogger(cfg)
    cfg.logger = load_logger(cfg, args)

    # if not is_training:
    #     cfg.TRAIN.TARGET_PREDEEL = 0
    #     cfg.TRAIN.INPUT_PREDEEL = 0

    if args.batch_size != 0:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    if cfg.TEST.ONE_TEST:
        cfg.TRAIN.DO_AUG = 0
        cfg.TRAIN.USE_LMDB=0

    try:
        if cfg.TEST.ONE_TEST:
            args.number_works = 0
    except:
        pass

    anchor_yolov2 = [[10, 13],  # [W,H]
                     [16, 30],
                     [33, 23],
                     [30, 61],
                     [62, 45],
                     [59, 119],
                     [116, 90],
                     [156, 198],
                     [373, 326]]

    anchor_yolov3_tiny = [[347., 286.],
                          [158., 211.],
                          [124., 94.],
                          [67., 132.],
                          [45., 59.],
                          [20., 28.],
                          ]  # VOC2007

    anchor_yolov3 = [[373, 326],
                     [156, 198],
                     [116, 90],
                     [59, 119],
                     [62, 45],
                     [30, 61],
                     [33, 23],
                     [16, 30],
                     [10, 13],
                     ]  # yolov3_tiny yolov3 writer's anchors

    # yolov3 writer's anchors: [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]

    VISDRONE_anchors = [[104., 92.],
                        [57., 41.],
                        [27., 43.],
                        [31., 20.],
                        [15., 23.],
                        [8., 11.],
                        ]
    # anchors should be 倒序。

    cfg.PATH.CLASSES_PATH = cfg.PATH.CLASSES_PATH.format(cfg.TRAIN.TRAIN_DATA_FROM_FILE[0].lower())

    cfg.TRAIN.ANCHORS = anchor_yolov3
    if cfg.TEST.ONE_TEST:
        if cfg.TEST.ONE_NAME != []:
            cfg.TRAIN.BATCH_SIZE = len(cfg.TEST.ONE_NAME)
        else:
            cfg.TRAIN.BATCH_SIZE = 1
        cfg.TRAIN.BATCH_BACKWARD_SIZE = 1

    if 'yolov3_tiny' in cfg.TRAIN.MODEL and ('KITTI' in cfg.TRAIN.TRAIN_DATA_FROM_FILE):
        cfg.TRAIN.FMAP_ANCHOR_NUM = 3
        cfg.TRAIN.ANCHORS = anchor_yolov3_tiny
    elif 'yolov3_tiny' in cfg.TRAIN.MODEL and (cfg.TRAIN.TRAIN_DATA_FROM_FILE[0] in ['VISDRONE', 'AUTOAIR']):
        cfg.TRAIN.FMAP_ANCHOR_NUM = 3
        cfg.TRAIN.ANCHORS = VISDRONE_anchors
    if 'yolov3_tiny' in cfg.TRAIN.MODEL:
        cfg.TRAIN.FMAP_ANCHOR_NUM = 3
        cfg.TRAIN.ANCHORS = anchor_yolov3_tiny
    elif cfg.TRAIN.MODEL in ['yolov3', 'yolonano']:
        cfg.TRAIN.FMAP_ANCHOR_NUM = 3
        cfg.TRAIN.ANCHORS = anchor_yolov3

    elif 'yolov2' in cfg.TRAIN.MODEL:
        cfg.TRAIN.FMAP_ANCHOR_NUM = len(anchor_yolov2)
        cfg.TRAIN.ANCHORS = anchor_yolov2

    try:
        from util.util_get_cls_names import _get_class_names
        class_dict= _get_class_names(cfg.PATH.CLASSES_PATH)
        class_names = []
        for k,v in class_dict.items():
            if v not in class_names:
                class_names.append(v)
        cfg.TRAIN.CLASSES_NUM = len(class_names)
        cfg.TRAIN.CLASSES = class_names
    except:
        print('cfg.py trying get class number and classes faild.')

    return cfg, args


def common_cfg(cfg):
    mean = np.asarray([123.675, 116.28, 103.53])
    std = np.asarray([58.395, 57.12, 57.375])
    cfg.mean = mean
    cfg.std = std
    return cfg
