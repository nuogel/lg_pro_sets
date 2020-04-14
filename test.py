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
    'tmp/checkpoint/now.pkl'
    # 'F:/test_results/saved_tbx_log_yolov3_tiny_clean/checkpoint.pkl',
    # 'F:/test_results/tbx_log_ssd_coco/checkpoint.pkl'
    # 'F:/test_results/tbx_log_vdsr/checkpoint.pkl'
    # 'F:/test_results/tbx_log_rcan/checkpoint.pkl'
    # 'F:/test_results/tbx_log_cbdnet/checkpoint.pkl'
    # 'F:/test_results/tbx_log_edsr/checkpoint.pkl'
    # 'F:/test_results/saved_tbx_log_rdn_youku_clean/checkpoint.pkl'
    # 'E:/LG/GitHub/lg_pro_sets/tmp/tbx_log_rdn/checkpoint.pkl'
                        , help='Path to the checkpoint to be loaded to the model')
    parser.add_argument('--batch_size', '--bz', default=1, type=int, help='batch size')
    parser.add_argument('--number_works', '--n_w', default=0, type=int, help='number works of DataLoader')
    return parser.parse_args()


def main():
    """Run the script."""
    exit_code = 0
    # files = None
    # files = 'E:/LG/GitHub/lg_pro_sets/util/util_tmp/3.txt'
    # files = 'E:/datasets/test_dataset/crop/5.png'
    # files = 'F:/datasets/SR/REDS4/train_sharp_part_x4/train_sharp_part_x4/000/00000000.png'
    # files = 'E:/LG/GitHub/lg_pro_sets/tmp/generated_labels/cbdnet_predict.png'
    # files = 'E:/LG/GitHub/lg_pro_sets/tmp/generated_labels/raw'
    # files = 'F:/datasets/SR/REDS4/train_sharp_part_x4/000'
    files = 'DataLoader/datasets/SRDN_idx_stores/FILMS/FILMS_test_w.txt'

    score = False
    args = _parse_arguments()
    cfg = parse_yaml(args.yml_path)
    # files = cfg.TEST.ONE_NAME[0]
    test = Test[cfg.BELONGS](cfg, args)
    test.test_run(files)
    if score:
        test.score(txt_info=files, pre_path=cfg.PATH.GENERATE_LABEL_SAVE_PATH)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
