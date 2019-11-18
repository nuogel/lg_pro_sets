import os
import wave
import numpy as np
import torch
import scipy
from scipy import signal
import glob
from python_speech_features import mfcc, delta
from util.util_is_use_cuda import _is_use_cuda


class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.batchsize = cfg.TRAIN.BATCH_SIZE
        # self.wav_length = cfg.TRAIN.WAV_LENGTH
        # self.lab_length = cfg.TRAIN.LAB_LENGTH
        self.win_length = self.cfg.TRAIN.AUDIO_FEATURE_LENGTH
        self.datalist = []
        self.one_test = cfg.TEST.ONE_TEST
        self.one_name = cfg.TEST.ONE_NAME
        self.SymbolNum = 0
        self.list_symbol = self._GetSymbolList()

    def get_data_by_idx(self, idx_store, index_from, index_to):
        '''
        :param idx_store:
        :param index_from:
        :param index_to:
        :return: imags: torch.Float32, relative labels:[[cls, x1, y1, x2, y2],[...],...]
        '''
        data = (None, None)
        idx = idx_store[index_from: index_to]
        if idx:
            if self.one_test:
                idx = self.one_name

            wav_length = []
            lab_length = []
            wav_list = []
            lab_list = []
            for i, name in enumerate(idx):  # read wav and lab
                # name = self.datalist[start_idx + i]
                wav_path = os.path.join(self.cfg.PATH.DATA_PATH, name + '.wav')
                # print(wav_path)
                wavsignal, fs = self._read_wav_data(wav_path)
                wavsignal = wavsignal[0]  # 取一个通道。
                wav_feature = self._get_wav_features(wavsignal, fs)
                ## test for __log_specgram()
                # out = self._log_specgram(wavsignal, fs)
                # wav_feature = self._get_mfcc_feature(wavsignal, fs)
                wav_length.append(wav_feature.shape[0])
                wav_list.append(wav_feature)

                lab_path = os.path.join(self.cfg.PATH.DATA_PATH, name + '.wav.trn')
                lab_compiled = self._get_wav_symbol(lab_path)
                lab_compiled.insert(0, self.SymbolNum - 2)  # 添加 【START]
                lab_compiled.append(self.SymbolNum - 1)  # 添加 【END]
                lab_length.append(len(lab_compiled))
                lab_list.append(lab_compiled)

            # make a matrix 变长序列
            wav_length = torch.LongTensor(wav_length)
            lab_length = torch.LongTensor(lab_length)
            max_wav_len = wav_length.max()
            max_lab_len = lab_length.max()
            wav_input = torch.zeros((self.batchsize, max_wav_len, self.win_length))
            lab_input = torch.zeros(size=(self.batchsize, max_lab_len), dtype=torch.long)
            for i in range(len(wav_list)):
                wav_input[i, 0:wav_length[i]] = torch.LongTensor(wav_list[i])
                lab_input[i, 0:lab_length[i]] = torch.LongTensor(lab_list[i])

            if _is_use_cuda(self.cfg.TRAIN.GPU_NUM):
                wav_input = wav_input.cuda(self.cfg.TRAIN.GPU_NUM)
                lab_input = lab_input.cuda(self.cfg.TRAIN.GPU_NUM)

            data = (wav_input, lab_input, wav_length, lab_length)
        return data

    def get_one_data_for_test(self, wav_path):
        # wav_length = []
        wavsignal, fs = self._read_wav_data(wav_path)
        wavsignal = wavsignal[0]  # 取一个通道。
        wav_feature = self._get_wav_features(wavsignal, fs)
        wav_length = wav_feature.shape[0]
        wav_input = torch.zeros((1, wav_length, self.win_length))
        wav_input[0, 0:len(wav_feature)] = torch.from_numpy(wav_feature)
        wav_length = torch.LongTensor(wav_length)
        if _is_use_cuda(self.cfg.TRAIN.GPU_NUM):
            wav_input = wav_input.cuda(self.cfg.TRAIN.GPU_NUM)
        data = (wav_input, wav_length, None, None)
        return data

    def _read_wav_data(self, filename):
        '''
        读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
        '''
        wav = wave.open(filename, "rb")  # 打开一个wav格式的声音文件流
        num_frame = wav.getnframes()  # 获取帧数
        num_channel = wav.getnchannels()  # 获取声道数
        framerate = wav.getframerate()  # 获取帧速率
        num_sample_width = wav.getsampwidth()  # 获取实例的比特宽度，即每一帧的字节数
        str_data = wav.readframes(num_frame)  # 读取全部的帧
        wav.close()  # 关闭流
        wave_data = np.fromstring(str_data, dtype=np.short)  # 将声音文件数据转换为数组矩阵形式
        wave_data.shape = -1, num_channel  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
        wave_data = wave_data.T  # 将矩阵转置
        return wave_data, framerate

    def _get_wav_features(self, wavsignal, fs):
        '''短时傅里叶变换'''
        if (16000 != fs):
            raise ValueError(
                '[Error] ASRT currently only supports wav audio files with a sampling rate of 16000 Hz, but this audio is ' + str(
                    fs) + ' Hz. ')
        # wav波形 加时间窗以及时移10ms
        window_length = int(fs * self.cfg.TRAIN.CHUNK_DURATION_S)  # 计算窗长度的公式，目前全部为400固定值
        stride_length = int(fs * self.cfg.TRAIN.STRIDE_S)  # 160 步长帧数
        wav_arr = np.array(wavsignal)
        signal = np.squeeze(wav_arr)
        if signal.ndim != 1:
            raise TypeError("enframe input must be a 1-dimensional array.")
        n_frames = 1 + np.int(np.floor((len(signal) - window_length) / float(stride_length)))  # 778
        signal_framed = np.zeros((n_frames, int(window_length / 2)))  # 200
        for i in range(n_frames):
            signal_divide = signal[i * stride_length: i * stride_length + window_length]
            signal_win = signal_divide * np.hamming(window_length)
            signal_fft = np.abs(np.fft.fft(signal_win)) / (window_length / 2)
            signal_framed[i] = signal_fft[0:int(window_length / 2)]
        data_input = np.log(signal_framed + (1e-10))
        return data_input

    def _log_specgram(self, audio, sample_rate, window_size=25,
                      step_size=10, eps=1e-10):
        nperseg = int(window_size * sample_rate / 1e3)
        nsteplap = int(step_size * sample_rate / 1e3)
        _f, _t, spec = signal.spectrogram(audio,  # spec[1, 201, 778] (0,1e14)
                                          fs=sample_rate,
                                          window='hamm',
                                          nperseg=nperseg,  # 窗口长度
                                          noverlap=nperseg - nsteplap,
                                          # 段与段之间的重叠面积 If None, noverlap = nperseg // 8. Defaults to None.
                                          detrend=False)
        out = np.log(spec.T.astype(np.float32) + eps)  # [778, 201, 1]

        # TODO: out = (out -mean)/std
        return out

    def _get_mfcc_feature(self, wavsignal, fs):
        # 获取输入特征
        feat_mfcc = mfcc(wavsignal, fs)
        feat_mfcc_d = delta(feat_mfcc, 2)
        feat_mfcc_dd = delta(feat_mfcc_d, 2)
        # 返回值分别是mfcc特征向量的矩阵及其一阶差分和二阶差分矩阵
        wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
        return wav_feature

    def _get_wav_symbol(self, filename):
        '''
        读取指定数据集中，所有wav文件对应的语音符号
        返回一个存储符号集的字典类型值
        '''
        f = open(filename, 'r', encoding='utf-8')  # 打开文件并读入
        lines = f.readlines()
        pinying = lines[1].split()  # 获取拼音
        f.close()
        labnumber = self._pinying2number(pinying)  # 将拼音转换成数字。
        return labnumber

    def _pinying2number(self, pinying):
        labnumber = []
        for i in pinying:
            labnumber.append(self.list_symbol.index(i))
        return labnumber

    def _GetSymbolList(self, from_file=False, remake=False):
        '''
        加载拼音符号列表，用于标记符号
        返回一个列表list类型变量
        '''
        tmp_dic_savepath = 'tmp//asr//'
        tmp_dic_file = os.path.join(tmp_dic_savepath, 'asr_dict_list_symbol.pt')
        if os.path.isfile(tmp_dic_file) and not remake:
            print('Loading the dict of words...')
            list_symbol = torch.load(tmp_dic_file)
        else:
            print('Preparing the dict of words...')
            if not os.path.isdir('tmp'):
                os.mkdir('tmp')
            if not os.path.isdir(tmp_dic_savepath):
                os.mkdir(tmp_dic_savepath)
            list_symbol = []  # 初始化符号列表
            if from_file:
                txt_obj = open(self.cfg.PATH.CLASSES_PATH, 'r', encoding='UTF-8')  # 打开文件并读入
                txt_text = txt_obj.read()
                txt_lines = txt_text.split('\n')  # 文本分割
                for i in txt_lines:
                    if (i != ''):
                        txt_l = i.split('\t')
                        list_symbol.append(txt_l[0])
                txt_obj.close()
            else:
                lab_files = glob.glob(self.cfg.PATH.DATA_PATH + '*.wav.trn')
                for lab_f in lab_files:
                    txt_text = open(lab_f, 'r', encoding='UTF-8').read()
                    txt_lines = txt_text.split('\n')  # 文本分割
                    for lab in txt_lines[1].split(' '):
                        lab = lab.strip()
                        if lab not in list_symbol:
                            list_symbol.append(lab)
            list_symbol.insert(0, '_')
            list_symbol.append('<START>')
            list_symbol.append('<END>')

            torch.save(list_symbol, tmp_dic_file)
        self.SymbolNum = len(list_symbol)
        if self.cfg.TRAIN.CLASS_LENGTH != self.SymbolNum:
            print('cfg.TRAIN.CLASS_LENGTH != SymbolNum, then changed it cfg.TRAIN.CLASS_LENGTH = SymbolNum')
            self.cfg.TRAIN.CLASS_LENGTH = self.SymbolNum

        return list_symbol

    def _number2pinying(self, num):
        pingyin = []
        for i in num:
            # print(i)
            pingyin.append(self.list_symbol[i])
        return pingyin
