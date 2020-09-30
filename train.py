#! /usr/bin/env python3
"""Script to train the apollo yolo model."""

import sys

# print(sys.path)
# sys.path.append('/home/lg/new_disk/deep_learning/lg_Pro_Set')
from argparse import ArgumentParser
from solver.train_solver import Solver
from util.util_yml_parse import parse_yaml


def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--type', default='OBD', type=str, help='yml_path')
    parser.add_argument('--checkpoint', '--cp', default=0
                        , help='Path to the checkpoint to be loaded to the model')
    parser.add_argument('--batch_size', '--bz', default=4, type=int, help='batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_continue', '--lr_c', default=0, type=float, help='learning rate')
    parser.add_argument('--number_works', '--n_w', default=4, type=int, help='number works of dataloader')

    parser.add_argument('--epoch-continue', default=None, type=int, help='Epoch of continue training')
    parser.add_argument('--debug', '--d', action='store_true', default=False, help='Enable verbose info')
    parser.add_argument('--test_only', '--to', default=False, type=bool, help='test only')
    return parser.parse_args()


def main():
    """Main. entry of this script."""
    exit_code = 0
    args = _parse_arguments()
    print(args)
    cfg = parse_yaml(args)
    solver = Solver(cfg, args, train=True)
    solver.train()
    # try:
    #     solver.train()
    # except:
    #     print('warning: please do not close this, saving checkpoints......')
    #     solver._save_checkpoint()
    # return exit_code


if __name__ == '__main__':
    sys.exit(main())
