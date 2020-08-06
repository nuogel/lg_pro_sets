import librosa
import librosa.filters
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
import lws


class Util_Audio:
    def __init__(self, cfg):
        self.cfg = cfg

    def load_wav(self, path):
        return librosa.core.load(path, sr=self.cfg.TRAIN.sample_rate)[0]

    def save_wav(self, wav, path):
        wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, self.cfg.TRAIN.sample_rate, wav.astype(np.int16))

    def preemphasis(self, x):
        from nnmnkwii.preprocessing import preemphasis
        return preemphasis(x, self.cfg.TRAIN.preemphasis)

    def inv_preemphasis(self, x):
        from nnmnkwii.preprocessing import inv_preemphasis
        return inv_preemphasis(x, self.cfg.TRAIN.preemphasis)

    def spectrogram(self, y):
        D = self._lws_processor().stft(self.preemphasis(y)).T
        S = self._amp_to_db(np.abs(D)) - self.cfg.TRAIN.ref_level_db
        return self._normalize(S)

    def _inv_spectrogram(self, spectrogram):
        '''Converts spectrogram to waveform using librosa'''
        S = self._db_to_amp(self._denormalize(spectrogram) + self.cfg.TRAIN.ref_level_db)  # Convert back to linear
        processor = self._lws_processor()
        D = processor.run_lws(S.astype(np.float64).T ** self.cfg.TRAIN.power)
        y = processor.istft(D).astype(np.float32)
        return self.inv_preemphasis(y)

    def inv_spectrogram(self, spectrogram):
        '''Converts spectrogram to waveform using librosa'''
        S = self._db_to_amp(self._denormalize(spectrogram) + self.cfg.TRAIN.ref_level_db)  # Convert back to linear
        return self.inv_preemphasis(self._griffin_lim(S ** self.cfg.TRAIN.power))  # Reconstruct phase

    def _griffin_lim(self,S):
        '''librosa implementation of Griffin-Lim
        Based on https://github.com/librosa/librosa/issues/434
        '''
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for i in range(60):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    def _istft(self, y):
        # n_fft, hop_length, win_length = self._stft_parameters()
        return librosa.istft(y, hop_length=self.cfg.TRAIN.hop_size, win_length=self.cfg.TRAIN.win_length)

    def _stft(self, y):
        # n_fft, hop_length, win_length = self._stft_parameters()
        return librosa.stft(y=y, n_fft=self.cfg.TRAIN.fft_size, hop_length=self.cfg.TRAIN.hop_size, win_length=self.cfg.TRAIN.win_length)

    def _stft_parameters(self):
        n_fft = (self.cfg.TRAIN.num_freq - 1) * 2
        hop_length = int(self.cfg.TRAIN.frame_shift_ms / 1000 * self.cfg.TRAIN.sample_rate)
        win_length = int(self.cfg.TRAIN.frame_length_ms / 1000 * self.cfg.TRAIN.sample_rate)
        return n_fft, hop_length, win_length

    def melspectrogram(self, y):
        D = self._lws_processor().stft(self.preemphasis(y)).T
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.cfg.TRAIN.ref_level_db
        if not self.cfg.TRAIN.allow_clipping_in_normalization:
            assert S.max() <= 0 and S.min() - self.cfg.TRAIN.min_level_db >= 0
        return self._normalize(S)

    def _lws_processor(self, ):
        return lws.lws(self.cfg.TRAIN.fft_size, self.cfg.TRAIN.hop_size, mode="speech")

    # Conversions:

    _mel_basis = None

    def _linear_to_mel(self, spectrogram):
        global _mel_basis
        if _mel_basis is None:
            _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _build_mel_basis(self, ):
        if self.cfg.TRAIN.fmax is not None:
            assert self.cfg.TRAIN.fmax <= self.cfg.TRAIN.sample_rate // 2
        return librosa.filters.mel(self.cfg.TRAIN.sample_rate, self.cfg.TRAIN.fft_size,
                                   fmin=self.cfg.TRAIN.fmin, fmax=self.cfg.TRAIN.fmax,
                                   n_mels=self.cfg.TRAIN.num_mels)

    def _amp_to_db(self, x):
        min_level = np.exp(self.cfg.TRAIN.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def _normalize(self, S):
        return np.clip((S - self.cfg.TRAIN.min_level_db) / -self.cfg.TRAIN.min_level_db, 0, 1)

    def _denormalize(self, S):
        return (np.clip(S, 0, 1) * -self.cfg.TRAIN.min_level_db) + self.cfg.TRAIN.min_level_db
