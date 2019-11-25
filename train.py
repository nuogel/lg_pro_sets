#! /usr/bin/env python3
"""Script to train the apollo yolo model."""

import sys
import os

# print(sys.path)
# sys.path.append('/home/lg/new_disk/deep_learning/Lg_Pro_Set')
import logging
from argparse import ArgumentParser
from net_works.train_solver import Solver
from cfg.yml_parse import parse_yaml


def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--yml_path', default='cfg/SR.yml'  #'cfg/yolov2.yml'#'cfg/ASR.yml'  #
                        , type=str, help='yml_path')
    parser.add_argument('--checkpoint', default='tmp/checkpoint/399.pkl' # None  #'tmp/checkpoint/40.pkl'#
                        , help='Path to the checkpoint to be loaded to the model')
    parser.add_argument('--lr', default=0.00001, type=float,
                        help='Learning rate')
    parser.add_argument('--epoch-continue', default=None, type=int,
                        help='Epoch of continue training')
    parser.add_argument('-d', '--debug', action='store_true', default=True,
                        help='Enable verbose info')
    return parser.parse_args()


def main():
    """Main. entry of this script."""
    exit_code = 0
    args = _parse_arguments()
    real_path = os.path.realpath('')
    yml_path = os.path.join(real_path, args.yml_path)
    cfg = parse_yaml(yml_path)

    logging.getLogger().level = logging.DEBUG
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format=cfg.TRAIN.LOG_FORMAT)

    solver = Solver(cfg, args)
    solver.train()
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
