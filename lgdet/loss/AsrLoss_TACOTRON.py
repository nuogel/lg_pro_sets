import torch
from util.util_audio import Util_Audio


class TACOTRONLOSS:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loss_l1 = torch.nn.L1Loss()
        self.util_audio = Util_Audio(cfg)

    def Loss_Call(self, pre, train_data, kwargs):
        epoch = kwargs['epoch']
        step = kwargs['step']

        x, input_lengths, mel, y = train_data
        mel_outputs, linear_outputs, attn = pre

        mel_loss = self.loss_l1(mel_outputs, mel)
        n_priority_freq = int(3000 / (self.cfg.TRAIN.sample_rate * 0.5) * self.cfg.TRAIN.num_freq)
        linear_loss = 0.5 * self.loss_l1(linear_outputs, y) + \
                      0.5 * self.loss_l1(linear_outputs[:, :, :n_priority_freq], y[:, :, :n_priority_freq])

        if (step + 1) % 25 == 0:
            linear_output = linear_outputs[0].cpu().data.numpy()
            # Predicted audio signal
            waveform = self.util_audio._inv_spectrogram(linear_output.T)
            self.util_audio.save_wav(waveform, 'pred%d-%d_wav_path.wav' % (epoch, step))

        return {'mel_loss': mel_loss / self.cfg.TRAIN.BATCH_SIZE, 'linear_loss': linear_loss / self.cfg.TRAIN.BATCH_SIZE}
