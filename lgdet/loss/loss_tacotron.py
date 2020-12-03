import torch
from lgdet.util.util_audio.util_audio import Util_Audio
from lgdet.util.util_audio.util_tacotron_audio import TacotronSTFT
import numpy as np


class TACOTRONLOSS:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.TRAIN.DEVICE
        self.loss_l1 = torch.nn.L1Loss()
        self.mseloss = torch.nn.MSELoss()
        self.bcelogistloss = torch.nn.BCEWithLogitsLoss()
        self.audio = Util_Audio(cfg)
        self.stft = TacotronSTFT(cfg.TRAIN)

    def Loss_Call(self, predicted, train_data, kwargs):
        epoch = kwargs['epoch']
        global_step = kwargs['global_step']
        mel_out_before, mel_out_after, gate_out, _ = predicted
        mel_target, gate_target = [xi.to(self.device) for xi in train_data[1]]
        mel_loss = self.mseloss(mel_out_before, mel_target) + self.mseloss(mel_out_after, mel_target)
        gate_loss = self.bcelogistloss(gate_out.view(-1, 1), gate_target.view(-1, 1))
        total_loss = mel_loss + gate_loss
        metrics = {'mel_loss': mel_loss,
                   'gate_loss': gate_loss}
        if (global_step + 1) % 1000 == 0:
            length = train_data[0][3][0]
            mel_out_after = mel_out_after.cpu()[0][..., :length]
            mel_target = mel_target.cpu()[0][..., :length]
            waveform = self.stft.in_mel_to_wav(mel_target)
            self.audio.write_wav(waveform, 'output/%d-%d_target.wav' % (epoch, global_step))
            waveform = self.stft.in_mel_to_wav(mel_out_after)
            self.audio.write_wav(waveform, 'output/%d-%d_pred.wav' % (epoch, global_step))
        return {'total_loss': total_loss, 'metrics': metrics}
