from lgdet.util.util_audio.util_audio import Util_Audio
from lgdet.util.util_audio.util_tacotron_audio import TacotronSTFT


class Score:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rate_all = 0.
        self.rate_batch = 0.
        self.batches = 0.
        self.audio = Util_Audio(cfg)
        self.stft = TacotronSTFT(cfg.TRAIN)

    def init_parameters(self):
        self.rate_all = 0.
        self.rate_batch = 0.
        self.batches = 0.

    def cal_score(self, predicted, dataset):
        _, mels, _, _, mel_lengths = predicted
        for i, mel in enumerate(mels):
            mel = mel.cpu()[..., :mel_lengths[i]]
            waveform = self.stft.in_mel_to_wav(mel)
            wav_path ='output/predicted/%d.wav' % (i)
            self.audio.write_wav(waveform, wav_path)

    def score_out(self):
        score = self.rate_all / (self.batches + 1e-6)
        return score, None
