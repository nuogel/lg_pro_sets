"""
Test.py is used for marking things with the weight file which we trained.

With a box outside of thing and with a label of what it is ,
and with it's score at the left top of the box.
"""
import os
import torch
import cv2
from util.util_ConfigFactory_Classes import get_model_class, get_score_class, get_loader_class
from DataLoader.DataLoaderFactory import dataloader_factory
import glob
from util.util_load_state_dict import load_state_dict
from util.util_prepare_device import load_device
from util.util_parse_prediction import ParsePredict
from util.util_data_aug import Dataaug
from util.util_show_img import _show_img
from util.util_parse_SR_img import parse_Tensor_img
from util.util_prepare_cfg import prepare_cfg
from util.util_img_block import img_cut
from util.util_nms_for_img_block import NMS_block
from util.util_time_stamp import Time
from DataLoader.DataLoaderFactory import dataloader_factory


class Test_Base(object):
    def __init__(self, cfg, args):
        self.cfg, self.args = prepare_cfg(cfg, args, is_training=False)
        self.cfg.TRAIN.DEVICE, self.device_ids = load_device(self.cfg)
        self.Model = get_model_class(self.cfg.BELONGS, self.cfg.TRAIN.MODEL)(self.cfg)
        self.dataloader_factory = dataloader_factory(self.cfg, self.args)
        self.DataLoader = get_loader_class(cfg.BELONGS)(self.cfg)
        self.SCORE = get_score_class(self.cfg.BELONGS)(self.cfg)
        self.SCORE.init_parameters()
        if self.args.checkpoint:
            self.model_path = self.args.checkpoint
        else:
            self.model_path = self.cfg.PATH.TEST_WEIGHT_PATH
        self.Model = load_state_dict(self.Model, self.args.checkpoint, self.cfg.TRAIN.DEVICE)
        self.Model = self.Model.to(self.cfg.TRAIN.DEVICE)
        # self.Model.eval()

    def test_backbone(self, DataSet):
        pass

    def test_run(self, dataset):
        """
        Test images in the file_s.

        :param file_s:
        :return:
        """

        DataSet = self.dataloader_factory.make_dataset(None, dataset)[1]
        self.test_backbone(DataSet)

    def prase_file(self, file_s):
        dataset = []
        if file_s is None:
            dataset = self.cfg.TEST.ONE_NAME

        elif os.path.isfile(file_s):
            if file_s.split('.')[1] == 'txt':  # .txt
                lines = open(file_s, 'r').readlines()
                for line in lines:
                    tmp = line.strip().split(";")
                    dataset.append(tmp)
            else:  # xx.jpg
                dataset.append([os.path.basename(file_s), file_s, file_s])

        elif os.path.isdir(file_s):
            files = glob.glob('{}/*.*'.format(file_s))
            for i, path in enumerate(files):
                # img = cv2.imread(path)
                name = os.path.basename(path)
                dataset.append([name, path, path])

        elif isinstance(list, file_s):
            dataset = file_s

        else:
            dataset = None
        return dataset


class Test_OBD(Test_Base):
    def __init__(self, cfg, args):
        super(Test_OBD, self).__init__(cfg, args)
        self.dataaug = Dataaug(cfg)
        self.parsepredict = ParsePredict(cfg)
        self.apolloclass2num = dict(zip(self.cfg.TRAIN.CLASSES, range(len(self.cfg.TRAIN.CLASSES))))

    def test_backbone(self, DataSet):
        """Test."""
        loader = iter(DataSet)
        timer = Time()

        for i in range(DataSet.__len__()):
            test_data = next(loader)
            timer.time_start()
            test_data = self.dataloader_factory.to_devce(test_data)
            inputs, targets, data_infos = test_data
            if self.cfg.TEST.IMG_BLOCK:
                raw_inputs = inputs.clone()
                [n, c, h, w] = raw_inputs.shape
                target_size = (512, 768)
                img_cuts_pixcel = img_cut(h, w, gap=300, target_size=target_size)
                labels_pres = [[]]
                for bbox in img_cuts_pixcel:
                    input = raw_inputs[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    predict = self.Model.forward(input_x=input, is_training=False)
                    labels_pre = self.parsepredict._parse_predict(predict)
                    for i, pre in enumerate(labels_pre[0]):
                        if self.cfg.TRAIN.RELATIVE_LABELS:
                            pre[2][0] *= target_size[1]
                            pre[2][1] *= target_size[0]
                            pre[2][2] *= target_size[1]
                            pre[2][3] *= target_size[0]

                            pre[2][0] += bbox[0]
                            pre[2][1] += bbox[1]
                            pre[2][2] += bbox[0]
                            pre[2][3] += bbox[1]

                            pre[2][0] /= w
                            pre[2][1] /= h
                            pre[2][2] /= w
                            pre[2][3] /= h
                        else:
                            pre[2][0] += bbox[0]
                            pre[2][1] += bbox[1]
                            pre[2][2] += bbox[0]
                            pre[2][3] += bbox[1]
                        labels_pres[0].append(pre)
                # TODO:nms for labels
                labels_pres = NMS_block(labels_pres, self.cfg)

            else:
                predicts = self.Model.forward(input_x=inputs, is_training=False)
                labels_pres = self.parsepredict._parse_predict(predicts)
            batches = 1
            timer.time_end()
            print('a batch time is', timer.diff)
            for i in range(batches):
                img_raw = [cv2.imread(data_infos[i][1])]
                img_in = inputs[i]
                _show_img(img_raw, labels_pres, img_in=img_in, pic_path=data_infos[i][1], cfg=self.cfg, is_training=False)

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
        # self.DataLoader = self.dataloader_factory.DataLoaderDict

    def test_backbone(self, wav_path):
        """Test."""
        # prepare paramertas
        self.cfg.TRAIN.BATCH_SIZE = 1
        test_data = self.DataLoader.get_one_data_for_test(wav_path)
        predict = self.Model.forward(test_data, is_training=False)
        for k, v in predict.items():
            print('pre:', k, self.DataLoader._number2pinying(v[:-1]))


class Test_SRDN(Test_Base):
    def __init__(self, cfg, args):
        super(Test_SRDN, self).__init__(cfg, args)
        # self.DataLoader = self.dataloader_factory.DataLoaderDict[cfg.BELONGS](cfg)

    def test_backbone(self, DataSet):
        """Test."""
        loader = iter(DataSet)

        for i in range(DataSet.__len__()):
            test_data = next(loader)
            test_data = self.dataloader_factory.to_devce(test_data)
            inputs, targets, data_infos = test_data
            predicts = self.Model.forward(input_x=inputs, is_training=False)
            predicts = predicts.permute(0, 2, 3, 1)

            batches = inputs.shape[0]
            save_paths = []
            if self.cfg.TEST.SAVE_LABELS:
                for i in range(batches):
                    data_info = data_infos[i]
                    os.makedirs(self.cfg.PATH.GENERATE_LABEL_SAVE_PATH, exist_ok=True)
                    os.makedirs(os.path.join(self.cfg.PATH.GENERATE_LABEL_SAVE_PATH, self.cfg.TRAIN.MODEL), exist_ok=True)
                    save_paths.append(os.path.join(self.cfg.PATH.GENERATE_LABEL_SAVE_PATH, self.cfg.TRAIN.MODEL + '/' + data_info[0]))

            predict_size = (predicts.shape[1], predicts.shape[2])
            inputs = torch.nn.functional.interpolate(inputs, size=predict_size)
            inputs = inputs.permute(0, 2, 3, 1)
            targets = targets.permute(0, 2, 3, 1)
            inputs_join_predicts = 1
            if inputs_join_predicts:
                try:
                    img_cat = torch.cat([inputs, predicts, targets], dim=1)
                except:
                    img_cat = torch.cat([inputs, predicts], dim=1)
            else:
                img_cat = predicts

            parse_Tensor_img(img_cat, pixcels_norm=self.cfg.TRAIN.PIXCELS_NORM, save_paths=save_paths,
                             show_time=self.cfg.TEST.SHOW_EVAL_TIME)


Test = {'OBD': Test_OBD,
        'ASR': Test_ASR,
        'SRDN': Test_SRDN,
        'VID': Test_OBD,
        }
