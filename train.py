#! /usr/bin/env python3
"""Script to train the apollo yolo model."""

import sys
import os

# print(sys.path)
# sys.path.append('/home/lg/new_disk/deep_learning/Lg_Pro_Set')
import logging
from argparse import ArgumentParser
from NetWorks.train_solver import Solver
from util.util_yml_parse import parse_yaml


def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--type', default='SRDN', type=str, help='yml_path')
    parser.add_argument('--checkpoint', '--cp', default='tmp/checkpoint/now.pkl'  #0  #
                        , help='Path to the checkpoint to be loaded to the model')
    parser.add_argument('--batch_size', '--bz', default=32, type=int, help='batch size')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--lr_continue', '--lr_c', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--epoch-continue', default=None, type=int, help='Epoch of continue training')
    parser.add_argument('--debug', '--d', action='store_true', default=False, help='Enable verbose info')
    parser.add_argument('--test_only', '--to', default=False, type=bool, help='test only')
    return parser.parse_args()


def main():
    """Main. entry of this script."""
    exit_code = 0
    args = _parse_arguments()
    # real_path = os.path.realpath('')
    # yml_type = os.path.join(real_path, args.type)
    cfg = parse_yaml(args.type)

    logging.getLogger().level = logging.DEBUG
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format=cfg.TRAIN.LOG_FORMAT)

    solver = Solver(cfg, args)
    solver.train()
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
