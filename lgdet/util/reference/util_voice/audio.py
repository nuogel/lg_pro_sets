import librosa
import numpy as np
import copy
from scipy import signal
from scipy.io import wavfile


def get_spectrograms(fpath, cfg):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
 '''
    # Loading sound file
    y, sample_rate = librosa.load(fpath, sr=cfg.sample_rate)

    # Trimming
    y, _ = librosa.effects.trim(y, top_db=cfg.top_db)

    # Preemphasis
    y = np.append(y[0], y[1:] - cfg.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=cfg.n_fft,
                          hop_length=cfg.hop_length,
                          win_length=cfg.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(sample_rate, cfg.n_fft, cfg.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - cfg.ref_db + cfg.max_db) / cfg.max_db, 1e-8, 1)
    mag = np.clip((mag - cfg.ref_db + cfg.max_db) / cfg.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag


def melspectrogram2wav(mel, cfg):
    '''# Generate wave file from spectrogram'''
    # transpose
    mel = mel.T

    # de-noramlize
    mel = (np.clip(mel, 0, 1) * cfg.max_db) - cfg.max_db + cfg.ref_db

    # to amplitude
    mel = np.power(10.0, mel * 0.05)
    m = _mel_to_linear_matrix(cfg)
    mag = np.dot(m, mel)

    # wav reconstruction
    wav = griffin_lim(mag, cfg)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -cfg.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def magspectrogram2wav(mag, max_db, ref_db, preemphasis):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * max_db) - max_db + ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag, cfg)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def _mel_to_linear_matrix(cfg):
    m = librosa.filters.mel(cfg.sample_rate, cfg.n_fft, cfg.n_mels)
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))


def griffin_lim(spectrogram, cfg):
    '''Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(50):
        X_t = invert_spectrogram(X_best, cfg)
        est = librosa.stft(X_t, cfg.n_fft, cfg.hop_length, win_length=cfg.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best, cfg)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram, cfg):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, cfg.hop_length, win_length=cfg.win_length, window="hann")


def dc_notch_filter(wav):
    # code from speex
    notch_radius = 0.982
    den = notch_radius ** 2 + 0.7 * (1 - notch_radius) ** 2
    b = np.array([1, -2, 1]) * notch_radius
    a = np.array([1, -2 * notch_radius, den])
    return signal.lfilter(b, a, wav)


def save_wav(wav, path, cfg):
    wav = dc_notch_filter(wav)
    wav = wav / np.abs(wav).max() * 0.999
    f1 = 0.5 * 32767 / max(0.01, np.max(np.abs(wav)))
    f2 = np.sign(wav) * np.power(np.abs(wav), 0.95)
    wav = f1 * f2
    wavfile.write(path, cfg.sample_rate, wav.astype(np.int16))


if __name__ == '__main__':
    import cfg

    wav_path = '/home/lg/datasets/BZNSYP_kedaTTS/waves/0_000001.wav'
    mel, mag = get_spectrograms(wav_path, cfg)
    wav = melspectrogram2wav(mel, cfg)
    save_wav(wav, 'lg.wav', cfg)
