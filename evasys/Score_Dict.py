from evasys.F1Score.f1score import F1Score
from evasys.ASR_WER.wer import ASR_SCORE
from evasys.Score_SR import SR_SCORE


Score = {'img': F1Score,
         'ASR': ASR_SCORE,
         'SR': SR_SCORE

         }
