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
    parser.add_argument('--checkpoint', default= 1
                        , help='Path to the checkpoint to be loaded to the model')
    parser.add_argument('--batch_size', '--bz', default=1, type=int, help='batch size')
    parser.add_argument('--number_works', '--n_w', default=0, type=int, help='number works of DataLoader')
    parser.add_argument('--debug', '--d', action='store_true', default=False, help='Enable verbose info')
    parser.add_argument('--log_file_path', default='tmp/logs/', help='log_file_path')

    return parser.parse_args()


def main():
    """Run the script."""
    exit_code = 0
    files = 'one_name'
    files = 'datasets/OBD_idx_stores/VOC/VOC_test.txt'
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
