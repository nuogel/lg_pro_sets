# from net_works.model.ObdModel_ssd_vgg import SSD
# from net_works.model.ObdModel_yolov2 import YoloV2
# from net_works.model.ObdModel_yolov3_net import YoloV3
# from net_works.model.ObdModel_yolov3_tiny import YoloV3_Tiny
# from net_works.model.ObdModel_yolov3_tiny_mobilenet import YoloV3_Tiny_MobileNet
# from net_works.model.ObdModel_yolov3_mobilenet import YoloV3_MobileNet
# from net_works.model.ObdModel_yolov3_tiny_squeezenet import YoloV3_Tiny_SqueezeNet
# from net_works.model.ObdModel_yolov3_tiny_shufflenet import YoloV3_Tiny_ShuffleNet
# from net_works.model.ObdModel_fcosnet import FCOS
# from net_works.model.ObdModel_refinedet import RefineDet
# from net_works.model.ObdModel_efficientdet import EfficientDet
# from net_works.model.AsrModel_RNN import RNN
# from net_works.model.AsrModel_CTC import CTC
# from net_works.model.AsrModel_SEQ2SEQ import SEQ2SEQ
#
# from net_works.model.SrModel_SRCNN import SRCNN
# from net_works.model.SrModel_FSRCNN import FSRCNN
# from net_works.model.SrModel_ESPCN import ESPCN
# from net_works.model.SrModel_VDSR import VDSR
# from net_works.model.SrModel_EDSR import EDSR
# from net_works.model.SrModel_RDN import RDN
# from net_works.model.SrModel_RCAN import RCAN
#
# from net_works.model.DnModel_CBDNet import CBDNet
# from net_works.model.DnModel_DNCNN import DnCNN
from NetWorks.loss.ObdLoss_FCOS import FCOSLOSS
from NetWorks.loss.AsrLoss_CTC import RnnLoss
from NetWorks.loss.AsrLoss_SEQ2SEQ import SEQ2SEQLOSS
from NetWorks.loss.SrDnLoss import SRDNLOSS
from NetWorks.loss.ObdLoss_REFINEDET import REFINEDETLOSS

# ModelDict = {
#     # OBD
#     'yolov2': YoloV2,
#     'yolov3': YoloV3,
#     'yolov3_tiny': YoloV3_Tiny,
#     'yolov3_tiny_squeezenet': YoloV3_Tiny_SqueezeNet,
#     'yolov3_tiny_mobilenet': YoloV3_Tiny_MobileNet,
#     'yolov3_tiny_shufflenet': YoloV3_Tiny_ShuffleNet,
#     'yolov3_mobilenet': YoloV3_MobileNet,
#     'fcos': FCOS,
#     'refinedet': RefineDet,
#     'efficientdet': EfficientDet,
#     'ssd': SSD,
#
#     # ASR
#     'rnn': RNN,
#     'ctc': CTC,
#     'seq2seq': SEQ2SEQ,
#
#     # SR_DN
#     'srcnn': SRCNN,
#     'fsrcnn': FSRCNN,
#     'espcn': ESPCN,
#     'edsr': EDSR,
#     'vdsr': VDSR,
#     'rdn': RDN,
#     'cbdnet': CBDNet,
#     'dncnn': DnCNN,
#     'rcan': RCAN,
# }

LossDict = {
    # OBD

    'fcos': FCOSLOSS,
    'refinedet': REFINEDETLOSS,

    # ASR
    'rnn': RnnLoss,
    'ctc': RnnLoss,
    'seq2seq': SEQ2SEQLOSS,

    # SR_DN
    'espcn': SRDNLOSS,
    'srcnn': SRDNLOSS,
    'fsrcnn': SRDNLOSS,
    'edsr': SRDNLOSS,
    'vdsr': SRDNLOSS,
    'rdn': SRDNLOSS,
    'cbdnet': SRDNLOSS,
    'dncnn': SRDNLOSS,
    'rcan': SRDNLOSS,

}


