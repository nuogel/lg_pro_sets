import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from pypinyin import lazy_pinyin, Style

from ..registry import DATALOADERS


@DATALOADERS.registry()
class TTS_Loader(DataLoader):
    def __init__(self, cfg):
        super(TTS_Loader, self).__init__(object)
        self.cfg = cfg
        self.datapath = self.cfg.PATH.INPUT_PATH
        self.one_test = cfg.TEST.ONE_TEST
        self.one_name = cfg.TEST.ONE_NAME
        self.label_dict = torch.load('DataLoader/others/bb_dict.pth')
        self.n_vocab = len(self.label_dict)

    def __len__(self):
        if self.one_test:
            if self.is_training:
                length = int(self.cfg.TEST.ONE_TEST_TRAIN_STEP)
            else:
                length = len(self.cfg.TEST.ONE_NAME)
        else:
            length = len(self.dataset_txt)
        return length

    def __getitem__(self, index):
        if self.one_test:
            data_info = self.dataset_txt[0]
        else:
            data_info = self.dataset_txt[index]
        [spec_path, mel_path, length, txt] = data_info
        spec = np.load(os.path.join(self.datapath, spec_path))
        mel = np.load(os.path.join(self.datapath, mel_path))
        txt_seq = self.text_to_sequence(txt)

        return txt_seq, mel, spec

    def _add_dataset(self, dataset, is_training):
        self.dataset_txt = dataset
        self.is_training = is_training

    def collate_fun(self, batch):
        """Create batch"""

        def _pad(seq, max_len):
            return np.pad(seq, (0, max_len - len(seq)),
                          mode='constant', constant_values=0)

        def _pad_2d(x, max_len):
            x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
                       mode="constant", constant_values=0)
            return x

        r = self.cfg.TRAIN.outputs_per_step
        input_lengths = [len(x[0]) for x in batch]
        max_input_len = np.max(input_lengths)
        # Add single zeros frame at least, so plus 1
        max_target_len = np.max([len(x[1]) for x in batch]) + 1
        if max_target_len % r != 0:
            max_target_len += r - max_target_len % r
            assert max_target_len % r == 0

        a = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
        x_batch = torch.LongTensor(a)

        input_lengths = torch.LongTensor(input_lengths)

        b = np.array([_pad_2d(x[1], max_target_len) for x in batch],
                     dtype=np.float32)
        mel_batch = torch.FloatTensor(b)

        c = np.array([_pad_2d(x[2], max_target_len) for x in batch],
                     dtype=np.float32)
        y_batch = torch.FloatTensor(c)

        # Sort by length
        sorted_lengths, indices = torch.sort(input_lengths.view(-1), dim=0, descending=True)
        sorted_lengths = sorted_lengths.long().numpy()
        x_batch, mel_batch, y_batch = x_batch[indices], mel_batch[indices], y_batch[indices]

        return x_batch, sorted_lengths, mel_batch, y_batch

    def label2num2(self, dict_label, txt):
        style = Style.TONE3
        txt_yj = lazy_pinyin(txt, style)
        num_list = []
        for txt_i in txt_yj:
            num_list.append(dict_label[txt_i])
        return num_list

    def text_to_sequence(self, text):
        text = text.strip().replace('“', '').replace('”', '')
        text2 = self.label2num2(self.label_dict, text)
        text2.append(self.label_dict['~'])
        return text2
