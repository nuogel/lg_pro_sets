
def get_score_class(belongs):
    if belongs == 'VID': belongs = 'OBD'
    class_name = 'Score_' + str(belongs).upper()
    model_file = __import__('NetWorks.score.' + class_name, fromlist=[class_name])
    model_class = _get_sub_model(model_file, 'Score')
    return model_class


def get_model_class(belongs, modelname):
    belongs_uper = str(belongs).upper()
    modelname_uper = str(modelname).upper()
    model_file = belongs_uper[0] + belongs_uper[1:].lower() + 'Model_' + modelname_uper
    model_class = _get_sub_model(__import__('NetWorks.model.' + model_file, fromlist=[modelname_uper]), modelname_uper)
    return model_class


def get_loader_class(belongs):
    # if belongs == 'VID': belongs = 'OBD'
    class_name = 'Loader_' + str(belongs).upper()
    model_file = __import__('DataLoader.' + class_name, fromlist=[class_name])
    model_class = _get_sub_model(model_file, 'Loader')
    return model_class


def get_loss_class(belongs, modelname):
    if modelname[:4] == 'yolo':
        from NetWorks.loss.ObdLoss_YOLO import YoloLoss
        return YoloLoss
    elif modelname in ['ssdvgg', 'efficientdet']:
        from NetWorks.loss.ObdLoss_MULTIBOX import MULTIBOXLOSS
        return MULTIBOXLOSS
    elif belongs == 'SRDN':
        from NetWorks.loss.SrDnLoss import SRDNLOSS
        return SRDNLOSS
    else:
        from NetWorks.loss.ObdLoss_FCOS import FCOSLOSS
        from NetWorks.loss.AsrLoss_CTC import RnnLoss
        from NetWorks.loss.AsrLoss_SEQ2SEQ import SEQ2SEQLOSS
        from NetWorks.loss.ObdLoss_REFINEDET import REFINEDETLOSS
        from NetWorks.loss.OcrLoss_PAN import PANLoss
        from NetWorks.loss.FlowLoss import FlowLoss
        loss_dict = {
            'fcos': FCOSLOSS,
            'refinedet': REFINEDETLOSS,

            # ASR
            'rnn': RnnLoss,
            'ctc': RnnLoss,
            'seq2seq': SEQ2SEQLOSS,
            'PAN': PANLoss,
            'flow_fgfa': FlowLoss,
        }
        return loss_dict[modelname]


def _get_sub_model(model_file, class_name):
    if hasattr(model_file, class_name):
        class_model = getattr(model_file, class_name)
    else:
        class_model = None
        ImportError('no such a model')
    return class_model
