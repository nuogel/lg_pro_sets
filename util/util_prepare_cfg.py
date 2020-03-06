def prepare_cfg(cfg, arg):
    if arg.batch_size != 0:
        cfg.TRAIN.BATCH_SIZE = arg.batch_size
    anchor_yolov2_apollo = [[0.0772422, 0.0632077],
                            [0.0332185, 0.0699152],
                            [0.3039470, 0.7423017],
                            [0.0491545, 0.1041431],
                            [0.2348479, 0.0989017],
                            [0.0193353, 0.1177316],
                            [0.0864546, 0.1502465],
                            [0.0378630, 0.0336919],
                            [0.0057380, 0.0268776],
                            [0.0211715, 0.0501949],
                            [0.0141959, 0.0302731],
                            [0.0120658, 0.0846409],
                            [0.0639976, 0.3795098],
                            [0.0079226, 0.0525171],
                            [0.1565745, 0.2865745],
                            [0.0307940, 0.1949102], ]  # apollo_anchors 16
    anchor_yolov3 = [[0.0240385, 0.0336538],
                     [0.0552885, 0.0649038],
                     [0.0889423, 0.1394231],
                     [0.1947115, 0.1971154],
                     [0.3245192, 0.4062500],
                     [0.8269231, 0.7668269]]  # yolov3_tiny
    # yolov3 writer's anchors: [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]
    if cfg.TEST.ONE_TEST:
        cfg.TRAIN.BATCH_SIZE = len(cfg.TEST.ONE_NAME)
        cfg.TRAIN.BATCH_BACKWARD_SIZE = 1

    if 'yolov3_tiny' in cfg.TRAIN.MODEL:
        cfg.TRAIN.FMAP_ANCHOR_NUM = 3
        cfg.TRAIN.ANCHORS = anchor_yolov3
    elif 'yolov3' == cfg.TRAIN.MODEL:
        cfg.TRAIN.FMAP_ANCHOR_NUM = 2
        cfg.TRAIN.ANCHORS = anchor_yolov3
    elif 'yolov2' in cfg.TRAIN.MODEL:
        cfg.TRAIN.FMAP_ANCHOR_NUM = 6
        cfg.TRAIN.ANCHORS = anchor_yolov2_apollo
    return cfg
