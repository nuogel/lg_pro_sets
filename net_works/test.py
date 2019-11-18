"""
Test.py is used for marking things with the weight file which we trained.

With a box outside of thing and with a label of what it is ,
and with it's score at the left top of the box.
"""
import os
import torch
import cv2
import numpy as np
from net_works.model.Model_Loss_Dict import ModelDict, LossDict
from dataloader.DataLoaderDict import DataLoaderDict
import glob
from util.util_parse_prediction import ParsePredict
from util.util_data_aug import Dataaug
from util.util_show_img import _show_img
from evasys.F1Score.f1score import F1Score


class Test_img:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.Model = ModelDict[cfg.TRAIN.MODEL](cfg)
        self.dataaug = Dataaug(cfg)
        self.parsepredict = ParsePredict(cfg)

    def test_backbone(self, test_picture_path):
        """Test."""
        # prepare paramertas
        if self.args.checkpoint:
            model_path = self.args.checkpoint
        else:
            model_path = self.cfg.PATH.TEST_WEIGHT_PATH

        self.Model.load_state_dict(torch.load(model_path))
        self.Model = self.Model.cuda()
        self.Model.eval()
        img_raw = cv2.imread(test_picture_path)
        if img_raw is None:
            print('ERROR：no such a image')
        if self.cfg.TEST.DO_AUG:
            img_aug, _ = self.dataaug.augmentation(for_one_image=img_raw, do_aug=self.cfg.TEST.DO_AUG,
                                                   resize=self.cfg.TEST.RESIZE)
            img_aug = img_aug[0]
        elif self.cfg.TEST.RESIZE:
            img_aug = cv2.resize(img_raw, (int(self.cfg.TRAIN.IMG_SIZE[1]), int(self.cfg.TRAIN.IMG_SIZE[0])))
        else:
            img_aug = img_raw
        img_in = torch.from_numpy(img_aug).unsqueeze(0).type(torch.FloatTensor).cuda()
        img_raw = torch.from_numpy(img_raw).unsqueeze(0).type(torch.FloatTensor)
        predict = self.Model.forward(img_in)
        labels_pre = self.parsepredict._parse_predict(predict)
        return _show_img(img_raw, labels_pre, img_in=img_in[0], pic_path=test_picture_path, cfg=self.cfg)

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
                for line in lines:
                    tmp = line.split(";")
                    FILES.append(tmp[1])
            else:  # xx.jpg
                FILES = [file_s]
        else:
            if os.path.isdir(file_s):
                FILES = os.listdir(file_s)
            elif isinstance(list, file_s):
                FILES = file_s

        _len = len(FILES)
        for i, file in enumerate(FILES):
            print('testing {}...{}/{}'.format(file, i + 1, _len))
            img_path = file
            self.test_backbone(img_path)

    def test_set(self):
        set_path = '../tmp/checkpoint/apollo_lg_1_test_set'
        test_set = torch.load(set_path)
        do_aug = False
        resize = True
        for img_idx in test_set:
            if os.path.isfile(self.cfg.IMGPATH + '%06d.png' % img_idx):
                file_name = '%06d.png' % img_idx
            else:
                file_name = '%06d.jpg' % img_idx
            imagepath = self.cfg.IMGPATH + file_name
            # label_path = cfg.LABPATH + '%06d.txt' % img_idx
            image1 = self.test_backbone(imagepath, do_aug, resize)
            # imgs, labels = self.augmentation([img_idx], do_aug=False, resize=False)
            image2 = _show_img([img_idx], do_aug, resize)
            image_cat = np.vstack((image1, image2))
            cv2.imshow('img', image_cat)
            cv2.waitKey()

    def calculate_f1_score(self, cfg):
        print('calculating the F1score...')
        f1sore = F1Score(cfg)
        pre_labels, gt_labels = f1sore.get_labels_txt(cfg.pre_path, cfg.lab_path)
        f1sore.cal_tp_fp(pre_labels, gt_labels, from_net=False)
        f1_sore, prec, rec = f1sore.f1_score()
        print('f1_sore: {}\nprec: {}\nrec: {}'.format(f1_sore, prec, rec))


class Test_asr:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.DataLoader = DataLoaderDict[cfg.TRAIN.BELONGS](cfg)
        self.Model = ModelDict[cfg.TRAIN.MODEL](cfg)

    def test_backbone(self, wav_path):
        """Test."""
        # prepare paramertas
        if self.args.checkpoint:
            model_path = self.args.checkpoint
        else:
            model_path = self.cfg.PATH.TEST_WEIGHT_PATH

        self.Model.load_state_dict(torch.load(model_path))
        self.Model = self.Model.cuda()
        self.Model.eval()
        self.cfg.TRAIN.BATCH_SIZE = 1
        test_data = self.DataLoader.get_one_data_for_test(wav_path)
        predict = self.Model.forward(test_data, eval=True)
        for k, v in predict.items():
            print('pre:', k, self.DataLoader._number2pinying(v[:-1]))

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
                for line in lines:
                    tmp = line.split(";")
                    FILES.append(tmp[1])
            else:  # xx.jpg
                FILES = [file_s]
        else:
            if os.path.isdir(file_s):
                FILES = glob.glob('{}/*.wav'.format(file_s))

            elif isinstance(list, file_s):
                FILES = file_s

        _len = len(FILES)
        for i, wav_path in enumerate(FILES):
            print('testing [{}/{}] {}'.format(i + 1, _len, wav_path))
            self.test_backbone(wav_path)


Test = {'img': Test_img,
        'ASR': Test_asr}
