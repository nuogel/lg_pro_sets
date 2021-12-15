import os.path
import tensorrt
import torch
from torch2trt import torch2trt, TRTModule
import onnx
import onnxruntime
import cv2
import numpy as np
import onnx_tensorrt.backend as backend
import time
import cv2
from lgdet.postprocess.parse_factory import ParsePredict
from lgdet.util.util_show_img import _show_img
from lgdet.util.util_time_stamp import Time
from torch2trt import TRTModule
import torch
from lgdet.util.util_yml_parse import parse_yaml
from argparse import ArgumentParser
from lgdet.util.util_lg_transformer import LgTransformer
import sys
from lgdet.config.cfg import prepare_cfg
from lgdet.util.util_prepare_device import load_device

sys.path.append('/home/dell/lg/code/lg_pro_sets')


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
    parser.add_argument('--score_thresh', '--st', default=0.2, type=float, help='score_thresh')

    return parser.parse_args()


class YOLOV5:
    def __init__(self, cfg, trtpath=None, onnxpath=None):
        self.cfg = cfg
        self.inputshape = (640, 640)
        self.parsepredict = ParsePredict(cfg)
        self.lgtransformer = LgTransformer(cfg)
        self.apolloclass2num = dict(zip(self.cfg.TRAIN.CLASSES, range(len(self.cfg.TRAIN.CLASSES))))
        if trtpath:
            self.model_trt_2 = TRTModule()
            self.model_trt_2.load_state_dict(torch.load(trtpath))
        if onnxpath:
            self.sess = onnxruntime.InferenceSession(onnxpath)
            self.input_name = self.sess.get_inputs()[0].name
            self.output_name = [self.sess.get_outputs()[0].name, self.sess.get_outputs()[1].name, self.sess.get_outputs()[2].name]

    def preprocess(self, img):
        img, label, data_info = self.lgtransformer.letter_box(img, label=[], new_shape=self.inputshape, auto=False, scaleup=True)
        img, label = self.lgtransformer.transpose(img, None)
        img = torch.unsqueeze(img, 0).cuda()
        return img, label, data_info

    def postprocess(self, predicts, img_input, data_info):
        labels_pres = self.parsepredict.parse_predict(predicts)
        labelsp = self.parsepredict.predict2labels(labels_pres, [data_info])
        img_raw = [img]
        _show_img(img_raw, labelsp, img_in=img_input, cfg=self.cfg, is_training=False, relative_labels=False)

    def forward_trt(self, img):
        img_input, labels, data_info = self.preprocess(img)
        predicts = self.model_trt_2.forward(img_input)
        self.postprocess(predicts, img_input, data_info)

    def forward_onnx(self, imgdata):
        img_input, labels, data_info = self.preprocess(img)
        predicts = self.sess.run(self.output_name, {self.input_name: imgdata})
        self.postprocess(predicts, img_input, data_info)


if __name__ == '__main__':
    score = False
    trtpath = '/home/dell/lg/code/lg_pro_sets/others/model_compression/torch2tensorrt/tmp/yolov5_with_model.pth.onnx.statedict_trt'
    onnxpath = '/home/dell/lg/code/lg_pro_sets/others/model_compression/torch2tensorrt/tmp/yolov5_with_model.pth.onnx'
    args = _parse_arguments()
    cfg = parse_yaml(args)
    cfg, args = prepare_cfg(cfg, args, is_training=False)
    load_device(cfg)
    yolov5 = YOLOV5(cfg, trtpath=trtpath, onnxpath=onnxpath)

    imgp = '/media/dell/data/voc/VOCdevkit/VOC2007/JPEGImages/000005.jpg'
    img = cv2.imread(imgp)
    yolov5.forward_trt(img)
    yolov5.forward_onnx(img)
