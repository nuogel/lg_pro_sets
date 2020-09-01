#! /usr/bin/env python3
"""Script to test the apollo yolo."""

import sys
from argparse import ArgumentParser
from util.util_yml_parse import parse_yaml
from solver.test_solver import Test


def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--type', default='OBD'  # SR_DN
                        , type=str, help='yml_path')
    parser.add_argument('--checkpoint', default=
    'tmp/checkpoint/now.pkl'
                        # 'F:/saved_weight/saved_tbx_log_yolov3_tiny_clean/checkpoint.pkl',
                        # 'F:/saved_weight/tbx_log_ssd_coco/checkpoint.pkl'
                        # 'F:/saved_weight/tbx_log_vdsr/checkpoint.pkl'
                        # 'F:/saved_weight/tbx_log_cbdnet/checkpoint.pkl'
                        # 'F:/saved_weight/tbx_log_edsr/checkpoint.pkl'
                        # 'F:/saved_weight/saved_tbx_log_rdn_youku_clean/checkpoint.pkl'
                        # 'F:/LG/GitHub/lg_pro_sets/tmp/tbx_log_rdn/checkpoint.pkl'
                        # 'F:/saved_weight/tbx_log_rcan_without_dataaug/checkpoint.pkl'
                        # 'F:\saved_weight\saved_tbx_log_efficientdet_kitti_car_512x768/checkpoint.pkl'
                        , help='Path to the checkpoint to be loaded to the model')
    parser.add_argument('--batch_size', '--bz', default=1, type=int, help='batch size')
    parser.add_argument('--number_works', '--n_w', default=0, type=int, help='number works of DataLoader')
    parser.add_argument('--debug', '--d', action='store_true', default=False, help='Enable verbose info')
    parser.add_argument('--log_file_path', default='tmp/logs/', help='log_file_path')

    return parser.parse_args()


def main():
    """Run the script."""
    exit_code = 0
    files = None
    files = 'datasets/OBD_idx_stores/COCO/COCO_test.txt'
    score = False
    args = _parse_arguments()
    cfg = parse_yaml(args)
    test = Test[cfg.BELONGS](cfg, args, train=False)
    test.test_run(files)
    if score:
        test.score(txt_info=files, pre_path=cfg.PATH.GENERATE_LABEL_SAVE_PATH)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
