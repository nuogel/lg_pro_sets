import cv2
from lgdet.postprocess.parse_factory import ParsePredict
from lgdet.util.util_show_img import _show_img
from lgdet.util.util_time_stamp import Time
from lgdet.solver.test_pakage._test_base import TestBase
from torch2trt import TRTModule
import torch
from lgdet.util.util_yml_parse import parse_yaml

def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--type', default='OBD'  # SR_DN
                        , type=str, help='yml_path')
    parser.add_argument('--model', type=str, help='yml_path')
    parser.add_argument('--checkpoint', '--cp', default=1
                        , help='Path to the checkpoint to be loaded to the model')
    parser.add_argument('--pre_trained', '--pt', default=0  # 'saved/checkpoint/fcos_voc_77_new.pkl'
                        , help='Epoch of continue training')
    parser.add_argument('--batch_size', '--bz', default=1, type=int, help='batch size')
    parser.add_argument('--gpu', help='number works of dataloader')
    parser.add_argument('--ema', default=0, type=int, help='ema')
    parser.add_argument('--number_works', '--nw', default=0, type=int, help='number works of dataloader')
    parser.add_argument('--debug', '--d', action='store_true', default=False, help='Enable verbose info')
    parser.add_argument('--score_thresh', '--st', default=0.1, type=float, help='score_thresh')

    return parser.parse_args()




class YOLOV5FORWARD:
    def __init__(self, cfg):
        self.parsepredict = ParsePredict(cfg)
        self.apolloclass2num = dict(zip(self.cfg.TRAIN.CLASSES, range(len(self.cfg.TRAIN.CLASSES))))
        self.model_trt_2 = TRTModule()
        self.model_trt_2.load_state_dict(torch.load('others/model_compression/torch2tensorrt/yolov5_with_model.pth.onnx.statedict_trt'))
        self.imgp = '/media/dell/data/voc/VOCdevkit/VOC2007/JPEGImages/000005.jpg'

    def run(self, img):


        labels_pres = self.parsepredict.parse_predict(predicts)
        labels_pres = self.parsepredict.predict2labels(labels_pres, data_infos)
        batches = 1
        for i in range(batches):
            img_raw = [cv2.imread(data_infos[i]['img_path'])]
            img_in = inputs[i]
            _show_img(img_raw, labels_pres, img_in=img_in, pic_path=data_infos[i]['img_path'], cfg=self.cfg,
                      is_training=False, relative_labels=False)


if __name__ == '__main__':
    score = False
    args = _parse_arguments()
    cfg = parse_yaml(args)