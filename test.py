#! /usr/bin/env python3
"""Script to test the apollo yolo."""

import sys
from argparse import ArgumentParser
from cfg.yml_parse import parse_yaml
from net_works.test_solver import Test
from evasys.Score_Dict import Score


def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--yml_path', default='cfg/yolov3.yml'
                        , type=str, help='yml_path')
    parser.add_argument('--checkpoint', default='tmp/checkpoint/5.pkl',
                        help='Path to the checkpoint to be loaded to the model')
    return parser.parse_args()


def main():
    """Run the script."""
    exit_code = 0
    # file_s = 'E://datasets//kitti//training//image_2//000000.png'
    # file_s = 'E://datasets//kitti//training//image_2//'
    # # file_s = 'datasets/kitti/training/image_2/000000.png'
    file_s = 'E:/datasets/Noise_Images/NOISE_kitti/level_5/images/KITTI_005066_1.png'
    # file_s = 'evasys/voc_mAP/occ_good.txt'
    # file_s = 'E:/datasets/NLP/THCHS/data_thchs30/data/A2_0.wav'  # /'

    args = _parse_arguments()
    cfg = parse_yaml(args.yml_path)
    test = Test[cfg.TRAIN.BELONGS](cfg, args)
    test.test_run(file_s)
    # score
    is_score = False
    if is_score:
        score = Score[cfg.TRAIN.BELONGS](cfg)
        if cfg.TRAIN.BELONGS == 'img' and cfg.TEST.SAVE_LABELS is True:
            pre_labels, gt_labels = score.get_labels_txt(cfg.PATH.GENERATE_LABEL_SAVE_PATH, cfg.PATH.LAB_PATH)
            score.cal_score(pre_labels, gt_labels, from_net=False)
            score.score_out()
        elif cfg.TRAIN.BELONGS is 'ASR':
            ...
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
