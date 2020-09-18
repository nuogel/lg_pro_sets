import os
import logging
from colorlog import ColoredFormatter
import datetime


def load_logger(cfg, args):
    log_path = os.path.join(cfg.PATH.TMP_PATH, 'txt_logs', str(datetime.date.today()) + '.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.isfile(log_path):
        os.mknod(log_path)

    logging.basicConfig(filename=log_path,
                        level=logging.DEBUG,  # if args.debug else logging.INFO
                        format='%(asctime)s %(levelname)-8s %(filename)s[line:%(lineno)d]: %(message)s',
                        # 定义输出log的格式
                        datefmt='%Y-%m-%d %H:%M:%S', )

    formatter = ColoredFormatter(
        "%(asctime)s %(log_color)s%(levelname)-8s %(reset)s %(filename)s[line:%(lineno)d]: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'blue',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        })

    logger = logging.getLogger('LG-PRO')
    # handler = logging.StreamHandler()
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    logger.info('logger init finished')
    return logger
