from net_works.model.ObdModel_yolov2 import ApolloYoloV2
from net_works.model.ObdModel_yolov3_net import YoloV3
from net_works.model.ObdModel_yolov3_tiny import YoloV3_Tiny
from net_works.model.ObdModel_yolov3_tiny_mobilenet import YoloV3_Tiny_MobileNet
from net_works.model.ObdModel_yolov3_mobilenet import YoloV3_MobileNet
from net_works.model.ObdModel_yolov3_tiny_squeezenet import YoloV3_Tiny_SqueezeNet
from net_works.model.ObdModel_yolov3_tiny_shufflenet import YoloV3_Tiny_ShuffleNet
from net_works.model.ObdModel_fcosnet import FCOS
from net_works.model.AsrModel_RNN import RNN
from net_works.model.AsrModel_CTC import MODEL_CTC
from net_works.model.AsrModel_Seq2Seq import Seq2Seq
from net_works.model.SrModel_EDSR import EDSR
from net_works.model.DeNoiseModel_CBDNet import CBDNet
from net_works.model.DeNoiseModel_DnCnn import DnCNN

from net_works.loss.ObdLoss_YOLO import YoloLoss
from net_works.loss.ObdLoss_FCOS import FCOSLOSS
from net_works.loss.AsrLoss_CTC import RnnLoss
from net_works.loss.AsrLoss_SEQ2SEQ import Seq2SeqLoss

from net_works.loss.Sr_Dn_Loss import SR_DN_Loss

ModelDict = {'yolov2': ApolloYoloV2,
             'yolov3': YoloV3,
             'yolov3_tiny': YoloV3_Tiny,
             'yolov3_tiny_squeezenet': YoloV3_Tiny_SqueezeNet,
             'yolov3_tiny_mobilenet': YoloV3_Tiny_MobileNet,
             'yolov3_tiny_shufflenet': YoloV3_Tiny_ShuffleNet,
             'yolov3_mobilenet': YoloV3_MobileNet,
             'fcos': FCOS,

             # ASR
             'rnn': RNN,
             'ctc': MODEL_CTC,
             'seq2seq': Seq2Seq,

             # SR
             'edsr': EDSR,
             'cbdnet': CBDNet,
             'dncnn': DnCNN,
             }

LossDict = {'yolov2': YoloLoss,
            'yolov3': YoloLoss,
            'yolov3_tiny': YoloLoss,
            'yolov3_tiny_mobilenet': YoloLoss,
            'yolov3_tiny_squeezenet': YoloLoss,
            'yolov3_tiny_shufflenet': YoloLoss,
            'yolov3_mobilenet': YoloLoss,
            'fcos': FCOSLOSS,

            # ASR
            'rnn': RnnLoss,
            'ctc': RnnLoss,
            'seq2seq': Seq2SeqLoss,

            # SR_DN
            'edsr': SR_DN_Loss,
            'cbdnet': SR_DN_Loss,
            'dncnn':SR_DN_Loss

            }
