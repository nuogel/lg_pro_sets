import logging
from colorlog import ColoredFormatter
import datetime

def load_logger(args):
    log_path = args.log_file_path+str(datetime.date.today())+'.txt'
    logging.basicConfig(filename=log_path,
                        level=logging.DEBUG if args.debug else logging.INFO,
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
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info('logger init finished')
    return logger
