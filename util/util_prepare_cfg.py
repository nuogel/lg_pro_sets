def prepare_cfg(cfg):
    if cfg.TEST.ONE_TEST:
        cfg.TRAIN.BATCH_SIZE = len(cfg.TEST.ONE_NAME)

    if 'yolov3_tiny' in cfg.TRAIN.MODEL:
        cfg.TRAIN.FMAP_ANCHOR_NUM = 3
    elif 'yolov2' in cfg.TRAIN.MODEL:
        cfg.TRAIN.FMAP_ANCHOR_NUM = 6
    elif 'yolov3' == cfg.TRAIN.MODEL:
        cfg.TRAIN.FMAP_ANCHOR_NUM = 2

    return cfg
