from lgdet.util.util_audio import Util_Audio


class Score:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rate_all = 0.
        self.rate_batch = 0.
        self.batches = 0.
        self.util_audio = Util_Audio(cfg)

    def init_parameters(self):
        self.rate_all = 0.
        self.rate_batch = 0.
        self.batches = 0.

    def cal_score(self, predict, dataset):
        mel_outputs, linear_outputs, attn = predict
        self.rate_batch = 0.
        print('batch NO:', self.batches)
        self.batches += 1

        for batch_i in range(linear_outputs.shape[0]):
            self.rate_batch += 0

            linear_output = linear_outputs[batch_i].cpu().data.numpy()
            # Predicted audio signal
            signal = self.util_audio.inv_spectrogram(linear_output.T)
            self.util_audio.save_wav(signal, "step{}_predicted.wav".format(batch_i))


    def score_out(self):
        score = self.rate_all / self.batches
        return score, None
