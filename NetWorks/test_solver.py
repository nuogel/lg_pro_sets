"""
Test.py is used for marking things with the weight file which we trained.

With a box outside of thing and with a label of what it is ,
and with it's score at the left top of the box.
"""
import os
import torch
import cv2
import numpy as np
from NetWorks.Model_Loss_Dict import ModelDict
from DataLoader.DataLoaderDict import DataLoaderDict
import glob
from util.util_prepare_device import load_device
from util.util_parse_prediction import ParsePredict
from util.util_data_aug import Dataaug
from util.util_show_img import _show_img
from util.util_parse_SR_img import parse_Tensor_img
from NetWorks.NetworksConfigFactory import Score
from util.util_prepare_cfg import prepare_cfg


class Test_Base(object):
    def __init__(self, cfg, args):
        self.cfg = prepare_cfg(cfg, args)
        self.args = args
        self.apolloclass2num = dict(zip(self.cfg.TRAIN.CLASSES, range(len(self.cfg.TRAIN.CLASSES))))
        self.cfg.TRAIN.DEVICE, self.device_ids = load_device(self.cfg)
        self.Model = ModelDict[self.cfg.TRAIN.MODEL](self.cfg)
        if self.args.checkpoint:
            self.model_path = self.args.checkpoint
        else:
            self.model_path = self.cfg.PATH.TEST_WEIGHT_PATH
        self.Model.load_state_dict(torch.load(self.model_path))
        self.Model = self.Model.to(self.cfg.TRAIN.DEVICE)
        if len(self.device_ids) > 1:
            self.Model = torch.nn.DataParallel(self.Model, device_ids=self.device_ids)
        self.Model.eval()

    def test_backbone(self, test_path):
        pass

    def test_run(self, file_s):
        """
        Test images in the file_s.

        :param file_s:
        :return:
        """
        if os.path.isfile(file_s):
            if file_s.split('.')[1] == 'txt':  # .txt
                lines = open(file_s, 'r').readlines()
                FILES = []
                GTS = []
                for line in lines:
                    tmp = line.split(";")
                    FILES.append(tmp[1])
                    if len(tmp) > 1:  # ground truth.
                        GTS.append(tmp[2].strip())
            else:  # xx.jpg
                FILES = [file_s]
        else:
            if os.path.isdir(file_s):
                FILES = glob.glob('{}/*.*'.format(file_s))

            elif isinstance(list, file_s):
                FILES = file_s
            else:
                FILES = None

        _len = len(FILES)
        for i, file_path in enumerate(FILES):
            # TODO: make a matrix instead of feed them one by one.
            print('testing [{}/{}] {}'.format(i + 1, _len, file_path))
            self.test_backbone(file_path)


class Test_OBD(Test_Base):
    def __init__(self, cfg, args):
        super(Test_OBD, self).__init__(cfg, args)
        self.dataaug = Dataaug(cfg)
        self.parsepredict = ParsePredict(cfg)
        self.DataLoader = DataLoaderDict[cfg.BELONGS](cfg)
        self.SCORE = Score[self.cfg.BELONGS](self.cfg)
        self.SCORE.init_parameters()

    def test_backbone(self, test_picture_path):
        """Test."""
        # prepare paramertas

        img_raw = cv2.imread(test_picture_path)
        if img_raw is None:
            print('ERROR：no such a image')
        if self.cfg.TEST.DO_AUG:
            img_aug, _ = self.dataaug.augmentation(for_one_image=img_raw)
            img_aug = img_aug[0]
        elif self.cfg.TEST.RESIZE:
            img_aug = cv2.resize(img_raw, (int(self.cfg.TRAIN.IMG_SIZE[1]), int(self.cfg.TRAIN.IMG_SIZE[0])))
        else:
            img_aug = img_raw
        img_in = torch.from_numpy(img_aug).unsqueeze(0).type(torch.FloatTensor).to(self.cfg.TRAIN.DEVICE)
        img_raw = torch.from_numpy(img_raw).unsqueeze(0).type(torch.FloatTensor)
        img_in = img_in.permute([0, 3, 1, 2, ])
        img_in = img_in / 127.5 - 1.
        predict = self.Model.forward(input_x=img_in, is_training=False)
        labels_pre = self.parsepredict._parse_predict(predict)
        return _show_img(img_raw, labels_pre, img_in=img_in[0], pic_path=test_picture_path, cfg=self.cfg)

    def score(self, txt_info, pre_path):
        pre_path_list = glob.glob(pre_path + '/*.*')
        lines = open(txt_info, 'r').readlines()
        gt_labels = []
        pre_labels = []
        for line in lines:
            tmp = line.split(";")
            gt_name = tmp[0].strip()
            for pre_path_i in pre_path_list:
                if gt_name == os.path.basename(pre_path_i).split('.')[0]:
                    gt_path = tmp[2].strip()
                    gt_labels.append(self.DataLoader._read_line(gt_path))
                    pre_labels.append(self.DataLoader._read_line(pre_path_i, predicted_line=True))
                    break

        self.SCORE.cal_score(pre_labels, gt_labels, from_net=False)
        return self.SCORE.score_out()


class Test_ASR(Test_Base):
    def __init__(self, cfg, args):
        super(Test_ASR, self).__init__(cfg, args)
        self.DataLoader = DataLoaderDict[cfg.BELONGS](cfg)

    def test_backbone(self, wav_path):
        """Test."""
        # prepare paramertas
        self.cfg.TRAIN.BATCH_SIZE = 1
        test_data = self.DataLoader.get_one_data_for_test(wav_path)
        predict = self.Model.forward(test_data, is_training=False)
        for k, v in predict.items():
            print('pre:', k, self.DataLoader._number2pinying(v[:-1]))


class Test_SR_DN(Test_Base):
    def __init__(self, cfg, args):
        super(Test_SR_DN, self).__init__(cfg, args)

    def test_backbone(self, img_path):
        """Test."""
        # prepare paramertas
        test_img = cv2.imread(img_path)
        test_img = torch.from_numpy(np.asarray((test_img - self.cfg.TRAIN.PIXCELS_NORM[0]) * 1.0 / self.cfg.TRAIN.PIXCELS_NORM[1])).unsqueeze(0).type(torch.FloatTensor)
        test_img = test_img.to(self.cfg.TRAIN.DEVICE)
        self.cfg.TRAIN.BATCH_SIZE = 1
        predict = self.Model.forward(test_img, is_training=False)
        if self.cfg.TEST.SAVE_LABELS == True:
            if not os.path.isdir(self.cfg.PATH.GENERATE_LABEL_SAVE_PATH):
                os.mkdir(self.cfg.PATH.GENERATE_LABEL_SAVE_PATH)
            save_path = os.path.join(self.cfg.PATH.GENERATE_LABEL_SAVE_PATH, os.path.basename(img_path).split('.')[0] + '.png')  # .format(self.cfg.TRAIN.UPSCALE_FACTOR))
        else:
            save_path = None
        parse_Tensor_img(predict, pixcels_norm=self.cfg.TRAIN.PIXCELS_NORM, save_path=save_path, show_time=self.cfg.TEST.SHOW_EVAL_TIME)


Test = {'OBD': Test_OBD,
        'ASR': Test_ASR,
        'SR_DN': Test_SR_DN,
        }
