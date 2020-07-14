def prepare_cfg(cfg, arg, is_training=True):
    if not is_training:
        cfg.TRAIN.TARGET_PREDEEL = 0
        cfg.TRAIN.INPUT_PREDEEL = 0

    if arg.batch_size != 0:
        cfg.TRAIN.BATCH_SIZE = arg.batch_size
    if cfg.TEST.ONE_TEST:
        cfg.TRAIN.DO_AUG = 0
    if cfg.TEST.ONE_TEST or cfg.TRAIN.SHOW_INPUT or cfg.BELONGS == 'VID':
        arg.number_works = 0
        cfg.TRAIN.SAVE_STEP = 50

    anchor_yolov2 = [[10, 13],   # [W,H]
                     [16, 30],
                     [33, 23],
                     [30, 61],
                     [62, 45],
                     [59, 119],
                     [116, 90],
                     [156, 198],
                     [373, 326]]

    # anchor_ratio_min = [[0.0240385, 0.0336538],
    #                       [0.0552885, 0.0649038],
    #                       [0.0889423, 0.1394231],
    #                       [0.1947115, 0.1971154],
    #                       [0.3245192, 0.4062500],
    #                       [0.8269231, 0.7668269]]  # yolov3_tiny

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

    # anchor_yolov3 = [[363., 289.],
    #                  [175., 236.],
    #                  [163., 109.],
    #                  [87., 155.],
    #                  [76., 60.],
    #                  [43., 99.],
    #                  [44., 31.],
    #                  [24., 53.],
    #                  [17., 21.],
    #                  ]  ## VOC2007
    # yolov3 writer's anchors: [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]

    VISDRONE_anchors = [[104., 92.],
                        [57., 41.],
                        [27., 43.],
                        [31., 20.],
                        [15., 23.],
                        [8., 11.],
                        ]
    # anchors should be 倒序。
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

    return cfg, arg
