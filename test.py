#! /usr/bin/env python3
"""Script to test the apollo yolo."""

import sys
from argparse import ArgumentParser
from lgdet.util.util_yml_parse import parse_yaml
from lgdet.solver.test_solver import Test


def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--type', default='OBD'  # SR_DN
                        , type=str, help='yml_path')
    parser.add_argument('--model', type=str, help='yml_path')
    parser.add_argument('--checkpoint', '--cp', default=1
                        , help='Path to the checkpoint to be loaded to the model')
    parser.add_argument('--pre_trained', '--pt', default=0
                        , help='Epoch of continue training')
    parser.add_argument('--batch_size', '--bz', default=1, type=int, help='batch size')
    parser.add_argument('--gpu', help='number works of dataloader')
    parser.add_argument('--number_works', '--nw', default=0, type=int, help='number works of dataloader')
    parser.add_argument('--debug', '--d', action='store_true', default=False, help='Enable verbose info')
    return parser.parse_args()


def main():
    """Run the script."""
    exit_code = 0
    # files = 'one_name'

    # files = 'datasets/OBD_idx_stores/VOC/VOC_test.txt'
    files = 'datasets/OBD_idx_stores/COCO/COCO_test.txt'  #
    # files = 'datasets/OBD_idx_stores/KITTI/KITTI_test.txt'
    score = False
    args = _parse_arguments()
    cfg = parse_yaml(args)
    test = Test[cfg.BELONGS](cfg, args, train=None)
    test.test_run(files)
    if score:
        test.score(txt_info=files, pre_path=cfg.PATH.GENERATE_LABEL_SAVE_PATH)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
