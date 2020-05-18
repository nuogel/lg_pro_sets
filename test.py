#! /usr/bin/env python3
"""Script to test the apollo yolo."""

import sys
from argparse import ArgumentParser
from util.util_yml_parse import parse_yaml
from NetWorks.test_solver import Test


def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--type', default='OBD'  # OBD SR_DN
                        , type=str, help='yml_path')
    parser.add_argument('--checkpoint', default=
    'tmp/checkpoint/ssd_4.6_124.pkl'
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
    return parser.parse_args()


def main():
    """Run the script."""
    exit_code = 0
    files = None
    # files = 'E:/LG/GitHub/lg_pro_sets/util/util_tmp/3.txt'
    # files = 'E:/datasets/test_dataset/crop/5.png'
    # files = 'F:/datasets/SR/REDS4/train_sharp_part_x4/train_sharp_part_x4/000/00000000.png'
    # files = 'E:/LG/GitHub/lg_pro_sets/tmp/generated_labels/cbdnet_predict.png'
    # files = 'F:/LG/GitHub/lg_pro_sets/tmp/generated_labels/raw/'
    # files = 'F:/LG/GitHub/lg_pro_sets/DataLoader/datasets/SRDN_idx_stores/YOUKU/YOUKU_test_set_w.txt'
    # files = 'F:/LG/GitHub/lg_pro_sets/DataLoader/datasets/SRDN_idx_stores/RED4/RED4_train_set_w.txt'
    # files = 'F:/datasets/SR/REDS4/train_sharp_part_x4/000'
    # files = 'DataLoader/datasets/SRDN_idx_stores/FILMS/FILMS_test_w.txt'
    # files = 'E:/datasets/SR/youku/youku_00150_00199_l/Youku_00150_l/Youku_00150_l_001.png'
    # files = 'E:/datasets/youku/youku_00200_00249_l/Youku_00203_l/Youku_00203_l_001.png'
    files = 'F:/Projects/auto_Airplane/TS02/20191217_153659_10/'
    # files = 'E:/datasets/VisDrone2019/VisDrone2019-VID-val/sequences/uav0000182_00000_v/'
    # files = 'F:/LG/GitHub/lg_pro_sets/DataLoader/datasets/OBD_idx_stores/KITTI/KITTI_train_set_w.txt'
    # files = 'F:/LG/GitHub/lg_pro_sets/DataLoader/datasets/OBD_idx_stores/VISDRONE/VISDRONE_test_set_w.txt'
    score = False
    args = _parse_arguments()
    cfg = parse_yaml(args)
    test = Test[cfg.BELONGS](cfg, args)
    dataset = test.prase_file(files)
    test.test_run(dataset)
    if score:
        test.score(txt_info=files, pre_path=cfg.PATH.GENERATE_LABEL_SAVE_PATH)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
