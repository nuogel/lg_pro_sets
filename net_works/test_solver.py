"""
Test.py is used for marking things with the weight file which we trained.

With a box outside of thing and with a label of what it is ,
and with it's score at the left top of the box.
"""
import os
import torch
import cv2
import numpy as np
from net_works.Model_Loss_Dict import ModelDict
from dataloader.DataLoaderDict import DataLoaderDict
import glob
from util.util_parse_prediction import ParsePredict
from util.util_data_aug import Dataaug
from util.util_show_img import _show_img
from util.util_is_use_cuda import _is_use_cuda
from evasys.Score_OBD_F1 import F1Score
from util.util_parse_SR_img import parse_Tensor_img
from evasys.Score_Dict import Score


class Test_Base(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.Model = ModelDict[cfg.TRAIN.MODEL](cfg)
        if self.args.checkpoint:
            self.model_path = self.args.checkpoint
        else:
            self.model_path = self.cfg.PATH.TEST_WEIGHT_PATH
        self.Model.load_state_dict(torch.load(self.model_path))
        if _is_use_cuda():
            self.Model = self.Model.cuda()
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
                for line in lines:
                    tmp = line.split(";")
                    FILES.append(tmp[1])
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

        is_score = False
        if is_score:
            score = Score[self.cfg.BELONGS](self.cfg)
            if self.cfg.BELONGS == 'OBD' and self.cfg.TEST.SAVE_LABELS is True:
                pre_labels, gt_labels = score.get_labels_txt(self.cfg.PATH.GENERATE_LABEL_SAVE_PATH, self.cfg.PATH.LAB_PATH)
                score.cal_score(pre_labels, gt_labels, from_net=False)
                return score.score_out()
            elif self.cfg.BELONGS is 'ASR':
                ...


class Test_OBD(Test_Base):
    def __init__(self, cfg, args):
        super(Test_OBD, self).__init__(cfg, args)
        self.dataaug = Dataaug(cfg)
        self.parsepredict = ParsePredict(cfg)
        self.DataLoader = DataLoaderDict[cfg.BELONGS](cfg)

    def test_backbone(self, test_picture_path):
        """Test."""
        # prepare paramertas

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


class Test_ASR(Test_Base):
    def __init__(self, cfg, args):
        super(Test_ASR, self).__init__(cfg, args)
        self.DataLoader = DataLoaderDict[cfg.BELONGS](cfg)

    def test_backbone(self, wav_path):
        """Test."""
        # prepare paramertas
        self.cfg.TRAIN.BATCH_SIZE = 1
        test_data = self.DataLoader.get_one_data_for_test(wav_path)
        predict = self.Model.forward(test_data, eval=True)
        for k, v in predict.items():
            print('pre:', k, self.DataLoader._number2pinying(v[:-1]))


class Test_SR(Test_Base):
    def __init__(self, cfg, args):
        super(Test_SR, self).__init__(cfg, args)

    def test_backbone(self, img_path):
        """Test."""
        # prepare paramertas
        test_img = cv2.imread(img_path)
        test_img = torch.from_numpy(np.asarray((test_img - self.cfg.TRAIN.PIXCELS_NORM[0]) * 1.0 / self.cfg.TRAIN.PIXCELS_NORM[1])).unsqueeze(0).type(torch.FloatTensor)
        if _is_use_cuda():
            test_img = test_img.cuda()
        self.cfg.TRAIN.BATCH_SIZE = 1
        predict = self.Model.forward(test_img, eval=True)
        if self.cfg.TEST.SAVE_LABELS == True:
            if not os.path.isdir(self.cfg.PATH.GENERATE_LABEL_SAVE_PATH):
                os.mkdir(self.cfg.PATH.GENERATE_LABEL_SAVE_PATH)
            save_path = os.path.join(self.cfg.PATH.GENERATE_LABEL_SAVE_PATH, os.path.basename(img_path).split('.')[0] + '_X{}.jpg'.format(self.cfg.TRAIN.UPSCALE_FACTOR))
        else:
            save_path = None
        parse_Tensor_img(predict, pixcels_norm=self.cfg.TRAIN.PIXCELS_NORM, save_path=save_path, show_time=10000)


Test = {'OBD': Test_OBD,
        'ASR': Test_ASR,
        'SR': Test_SR,
        }
