"""
Test.py is used for marking things with the weight file which we trained.

With a box outside of thing and with a label of what it is ,
and with it's score at the left top of the box.
"""
import os
import torch
import cv2
import glob
from lgdet.postprocess.parse_prediction import ParsePredict
from lgdet.util.util_show_img import _show_img
from lgdet.util.util_parse_SR_img import parse_Tensor_img
from lgdet.util.util_img_block import img_cut
from lgdet.util.util_nms_for_img_block import NMS_block
from lgdet.util.util_time_stamp import Time
from lgdet.util.util_audio import Util_Audio
from .solver_base import BaseSolver


class TestBase(BaseSolver):
    def __init__(self, cfg, args, train):
        super().__init__(cfg, args, train)
        ...

    def test_backbone(self, DataSet):
        pass

    def test_run(self, file_s):
        """
        Test images in the file_s.

        :param file_s:
        :return:
        """
        dataset = self.prase_file(file_s)
        DataSet = self.DataFun.make_dataset(None, dataset)[1]
        self.test_backbone(DataSet)

    def prase_file(self, file_s):
        dataset = []
        if file_s == 'one_name':
            dataset = self.cfg.TEST.ONE_NAME

        elif os.path.isfile(file_s):
            if file_s.split('.')[1] == 'txt':  # .txt
                lines = open(file_s, 'r', encoding='utf-8').readlines()
                for line in lines:
                    tmp = line.strip().split("┣┫")
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


class Test_OBD(TestBase):
    def __init__(self, cfg, args, train):
        super(Test_OBD, self).__init__(cfg, args, train)
        self.parsepredict = ParsePredict(cfg)
        self.apolloclass2num = dict(zip(self.cfg.TRAIN.CLASSES, range(len(self.cfg.TRAIN.CLASSES))))

    def test_backbone(self, DataSet):
        """Test."""
        loader = iter(DataSet)
        timer = Time()
        for i in range(DataSet.__len__()):
            test_data = next(loader)
            timer.time_start()
            test_data = self.DataFun.to_devce(test_data)
            inputs, targets, data_infos = test_data
            if self.cfg.TEST.IMG_BLOCK:
                ...
                # raw_inputs = inputs.clone()
                # [n, c, h, w] = raw_inputs.shape
                # target_size = (512, 768)
                # img_cuts_pixcel = img_cut(h, w, gap=300, target_size=target_size)
                # labels_pres = [[]]
                # for bbox in img_cuts_pixcel:
                #     input = raw_inputs[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]
                #     predict = self.model.forward(input_x=input, is_training=False)
                #     labels_pre = self.parsepredict.parse_predict(predict)
                #     for i, pre in enumerate(labels_pre[0]):
                #         if self.cfg.TRAIN.RELATIVE_LABELS:
                #             pre[2][0] *= target_size[1]
                #             pre[2][1] *= target_size[0]
                #             pre[2][2] *= target_size[1]
                #             pre[2][3] *= target_size[0]
                #
                #             pre[2][0] += bbox[0]
                #             pre[2][1] += bbox[1]
                #             pre[2][2] += bbox[0]
                #             pre[2][3] += bbox[1]
                #
                #             pre[2][0] /= w
                #             pre[2][1] /= h
                #             pre[2][2] /= w
                #             pre[2][3] /= h
                #         else:
                #             pre[2][0] += bbox[0]
                #             pre[2][1] += bbox[1]
                #             pre[2][2] += bbox[0]
                #             pre[2][3] += bbox[1]
                #         labels_pres[0].append(pre)
                # TODO:nms for labels
                # labels_pres = NMS_block(labels_pres, self.cfg)
            else:
                predicts = self.model.forward(input_x=inputs, is_training=False)
                labels_pres = self.parsepredict.parse_predict(predicts)
                labels_pres = self.parsepredict.predict2labels(labels_pres, data_infos)
            batches = 1
            timer.time_end()
            print('a batch time is', timer.diff)
            for i in range(batches):
                img_raw = [cv2.imread(data_infos[i]['img_path'])]
                img_in = inputs[i]
                _show_img(img_raw, labels_pres, img_in=img_in, pic_path=data_infos[i]['img_path'], cfg=self.cfg,
                          is_training=False, relative_labels=False)


class Test_ASR(TestBase):
    def __init__(self, cfg, args, train):
        super(Test_ASR, self).__init__(cfg, args, train)
        # self.DataLoader = self.DataFun.DataLoaderDict

    def test_backbone(self, wav_path):
        """Test."""
        # prepare paramertas
        self.cfg.TRAIN.BATCH_SIZE = 1
        test_data = self.DataLoader.get_one_data_for_test(wav_path)
        predict = self.model.forward(test_data, is_training=False)
        for k, v in predict.items():
            print('pre:', k, self.DataLoader._number2pinying(v[:-1]))


class Test_SRDN(TestBase):
    def __init__(self, cfg, args, train):
        super(Test_SRDN, self).__init__(cfg, args, train)
        # self.DataLoader = self.DataFun.DataLoaderDict[cfg.BELONGS](cfg)

    def test_backbone(self, DataSet):
        """Test."""
        loader = iter(DataSet)

        for i in range(DataSet.__len__()):
            test_data = next(loader)
            test_data = self.DataFun.to_devce(test_data)
            inputs, targets, data_infos = test_data
            predicts = self.model.forward(input_x=inputs, is_training=False)
            predicts = predicts.permute(0, 2, 3, 1)

            batches = inputs.shape[0]
            save_paths = []
            if self.cfg.TEST.SAVE_LABELS:
                for i in range(batches):
                    data_info = data_infos[i]
                    os.makedirs(self.cfg.PATH.GENERATE_LABEL_SAVE_PATH, exist_ok=True)
                    os.makedirs(os.path.join(self.cfg.PATH.GENERATE_LABEL_SAVE_PATH, self.cfg.TRAIN.MODEL),
                                exist_ok=True)
                    save_paths.append(
                        os.path.join(self.cfg.PATH.GENERATE_LABEL_SAVE_PATH, self.cfg.TRAIN.MODEL + '/' + data_info[0]))

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


class Test_TTS(TestBase):
    def __init__(self, cfg, args, train):
        super(Test_TTS, self).__init__(cfg, args, train)
        self.util_audio = Util_Audio(cfg)

    def test_backbone(self, DataSet):
        """Test."""
        loader = iter(DataSet)
        timer = Time()
        self.model.encoder.eval()
        self.model.postnet.eval()
        for i in range(DataSet.__len__()):
            test_data = next(loader)
            timer.time_start()
            test_data = self.DataFun.to_devce(test_data)
            predicted = self.model.forward(input_x=test_data[0], input_data=test_data, is_training=False)
            mel_outputs, linear_outputs, alignments = predicted
            linear_output = linear_outputs[0].cpu().data.numpy()
            # Predicted audio signal
            waveform = self.util_audio._inv_spectrogram(linear_output.T)
            self.util_audio.save_wav(waveform, 'dst_wav_path.wav')


Test = {'OBD': Test_OBD,
        'ASR': Test_ASR,
        'SRDN': Test_SRDN,
        'VID': Test_OBD,
        'TTS': Test_TTS,
        }
