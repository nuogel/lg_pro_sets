from evasys.Score_OBD_F1 import F1Score
from evasys.ASR_WER.wer import ASR_SCORE
from evasys.Score_SR_DN import SR_DN_SCORE


Score = {'OBD': F1Score,
         'ASR': ASR_SCORE,
         'SR_DN': SR_DN_SCORE

         }
