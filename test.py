#! /usr/bin/env python3
"""Script to test the apollo yolo."""

import sys
from argparse import ArgumentParser
from util.util_yml_parse import parse_yaml
from NetWorks.test_solver import Test


def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--yml_path', default='SRDN'  # OBD SR_DN
                        , type=str, help='yml_path')
    parser.add_argument('--checkpoint', default=
    # 'tmp/checkpoint/now.pkl'
    # 'F:/test_results/saved_tbx_log_yolov3_tiny_clean/checkpoint.pkl',
    # 'F:/test_results/tbx_log_ssd_coco/checkpoint.pkl'
    # 'F:/test_results/tbx_log_vdsr/checkpoint.pkl'
    # 'F:/test_results/tbx_log_rcan/checkpoint.pkl'
    # 'F:/test_results/tbx_log_cbdnet/checkpoint.pkl'
    # 'F:/test_results/tbx_log_edsr/checkpoint.pkl'
    'F:/test_results/tbx_log_rdn/checkpoint.pkl'

                        , help='Path to the checkpoint to be loaded to the model')
    parser.add_argument('--batch_size', '--bz', default=1, type=int, help='batch size')

    return parser.parse_args()


def main():
    """Run the script."""
    exit_code = 0
    # file_s = None
    file_s = 'E:/LG/GitHub/lg_pro_sets/util/util_tmp/3.txt'
    file_s = 'E:/datasets/test_dataset/crop/5.png'
    file_s = 'F:/datasets/SR/REDS4/train_sharp_part_x4/train_sharp_part_x4/000/00000000.png'
    # file_s = 'E:/LG/GitHub/lg_pro_sets/tmp/generated_labels/cbdnet_predict.png'
    score = False
    args = _parse_arguments()
    cfg = parse_yaml(args.yml_path)
    # file_s = cfg.TEST.ONE_NAME[0]
    test = Test[cfg.BELONGS](cfg, args)
    test.test_run(file_s)
    if score:
        test.score(txt_info=file_s, pre_path=cfg.PATH.GENERATE_LABEL_SAVE_PATH)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
