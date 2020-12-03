import librosa.filters
import librosa
import numpy as np
import copy
from scipy import signal
from scipy.io import wavfile
import torch
from pypinyin import lazy_pinyin, Style
import os


class Util_Audio:
    def __init__(self, cfg):
        self.cfg = cfg.TRAIN
        self.label_dict = torch.load(cfg.PATH.CLASSES_PATH)

    def load_wav(self, path):
        return librosa.load(path, sr=self.cfg.sample_rate)[0]

    def write_wav(self, wav, path):
        wav = self._dc_notch_filter(wav)
        wav = wav / np.abs(wav).max() * 0.999
        f1 = 0.8 * 32768 / max(0.01, np.max(np.abs(wav)))
        f2 = np.sign(wav) * np.power(np.abs(wav), 0.95)
        wav = f1 * f2
        dir = os.path.dirname(path)
        os.makedirs(dir, exist_ok=True)
        wavfile.write(path, self.cfg.sample_rate, wav.astype(np.int16))
        print('saved waves to: ', path)

    def preemphasize(self, wav, preemphasis=0.97):
        return signal.lfilter([1, -preemphasis], [1], wav)

    def de_emphasize(self, wav, preemphasis=0.97):
        return signal.lfilter([1], [1, -preemphasis], wav)

    def trim_silence(self, wav):
        return librosa.effects.trim(wav, top_db=self.cfg.trim_top_db, frame_length=self.cfg.trim_fft_size, hop_length=self.cfg.trim_hop_size)[0]

    def _stft(self, y):
        return librosa.stft(y=y, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length, win_length=self.cfg.win_length)

    def _istft(self, y):
        return librosa.istft(y, hop_length=self.cfg.hop_length, win_length=self.cfg.win_length)

    def get_spectrograms(self, fpath):
        '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
        Args:
          sound_file: A string. The full path of a sound file.

        Returns:
          mel: A 2d array of shape (T, n_mels) <- Transposed
          mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
     '''
        # Loading sound file
        y, sample_rate = librosa.load(fpath, sr=self.cfg.sample_rate)

        # Trimming
        y, _ = librosa.effects.trim(y, top_db=self.cfg.top_db)

        # Preemphasis
        y = np.append(y[0], y[1:] - self.cfg.preemphasis * y[:-1])

        # stft
        linear = librosa.stft(y=y,
                              n_fft=self.cfg.n_fft,
                              hop_length=self.cfg.hop_length,
                              win_length=self.cfg.win_length)

        # magnitude spectrogram
        mag = np.abs(linear)  # (1+n_fft//2, T)

        # mel spectrogram
        mel_basis = librosa.filters.mel(sample_rate, self.cfg.n_fft, self.cfg.n_mels)  # (n_mels, 1+n_fft//2)
        mel = np.dot(mel_basis, mag)  # (n_mels, t)

        # to decibel
        mel = 20 * np.log10(np.maximum(1e-5, mel))
        mag = 20 * np.log10(np.maximum(1e-5, mag))

        # normalize
        mel = np.clip((mel - self.cfg.ref_db + self.cfg.max_db) / self.cfg.max_db, 1e-8, 1)
        mag = np.clip((mag - self.cfg.ref_db + self.cfg.max_db) / self.cfg.max_db, 1e-8, 1)

        # Transpose
        mel = mel.T.astype(np.float32)  # (T, n_mels)
        mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

        return mel, mag

    def melspectrogram2wav(self, mel):
        '''# Generate wave file from spectrogram'''
        # transpose
        mel = mel.T

        # de-noramlize
        mel = (np.clip(mel, 0, 1) * self.cfg.max_db) - self.cfg.max_db + self.cfg.ref_db

        # to amplitude
        mel = np.power(10.0, mel * 0.05)
        m = self._mel_to_linear_matrix()
        mag = np.dot(m, mel)

        # wav reconstruction
        wav = self.griffin_lim(mag)

        # de-preemphasis
        wav = signal.lfilter([1], [1, -self.cfg.preemphasis], wav)

        # trim
        wav, _ = librosa.effects.trim(wav)

        return wav.astype(np.float32)

    def magspectrogram2wav(self, mag):
        '''# Generate wave file from spectrogram'''
        # transpose
        mag = mag.T

        # de-noramlize
        mag = (np.clip(mag, 0, 1) * self.cfg.max_db) - self.cfg.max_db + self.cfg.ref_db

        # to amplitude
        mag = np.power(10.0, mag * 0.05)

        # wav reconstruction
        wav = self.griffin_lim(mag)

        # de-preemphasis
        wav = signal.lfilter([1], [1, -self.cfg.preemphasis], wav)

        # trim
        wav, _ = librosa.effects.trim(wav)

        return wav.astype(np.float32)

    def _mel_to_linear_matrix(self):
        m = librosa.filters.mel(self.cfg.sample_rate, self.cfg.n_fft, self.cfg.n_mels)
        m_t = np.transpose(m)
        p = np.matmul(m, m_t)
        d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
        return np.matmul(m_t, np.diag(d))

    def griffin_lim(self, spectrogram):
        '''Applies Griffin-Lim's raw.
        '''
        X_best = copy.deepcopy(spectrogram)
        for i in range(50):
            X_t = self._invert_spectrogram(X_best)
            est = librosa.stft(X_t, self.cfg.n_fft, self.cfg.hop_length, win_length=self.cfg.win_length)
            phase = est / np.maximum(1e-8, np.abs(est))
            X_best = spectrogram * phase
        X_t = self._invert_spectrogram(X_best)
        y = np.real(X_t)

        return y

    def _invert_spectrogram(self, spectrogram):
        '''
        spectrogram: [f, t]
        '''
        return librosa.istft(spectrogram, self.cfg.hop_length, win_length=self.cfg.win_length, window="hann")

    def _dc_notch_filter(self, wav):
        # code from speex
        notch_radius = 0.982
        den = notch_radius ** 2 + 0.7 * (1 - notch_radius) ** 2
        b = np.array([1, -2, 1]) * notch_radius
        a = np.array([1, -2 * notch_radius, den])
        return signal.lfilter(b, a, wav)

    def _txt2num2(self, dict_label, txt):
        style = Style.TONE3
        txt_yj = lazy_pinyin(txt, style)
        num_list = []
        for txt_i in txt_yj:
            try:
                num_list.append(dict_label[txt_i])
            except:
                pass
        if txt[-1] != '。':
            num_list.append(dict_label['。'])
        return num_list

    def text_to_sequence(self, text):
        text = text.strip().replace('“', '').replace('”', '').replace(',', '，')
        text2 = self._txt2num2(self.label_dict, text)
        text2.append(self.label_dict['~'])
        return text2


if __name__ == '__main__':
    wav_path = '/home/lg/datasets/BZNSYP_kedaTTS/waves/0_000001.wav'
    from lgdet.util.reference.util_voice import cfg

    _mel_basis = None

    ua = Util_Audio(cfg)
    mel, mag = ua.get_spectrograms(wav_path)
    # voice1 = ua.inv_spectrogram1(mel)
    voice2 = ua.melspectrogram2wav(mel)
    # ua.write_wav(voice1, 'voice1.wav')
    ua.write_wav(voice2, 'voice2.wav')
