"""Parse the predictions."""
from lgdet.model.aid_models.fcos.head import DetectHead, ClipBoxes


class ParsePredict_fcos:
    def __init__(self, config):
        self.detection_head = DetectHead(config.score_threshold, config.nms_iou_threshold,
                                         config.max_detection_boxes_num, config.strides, config)
        self.clip_boxes = ClipBoxes()

    def parse_predict(self, predicts):
        out = self.detection_head(predicts)
        reshape_out = self.reshape_predict(out)
        return reshape_out

    def reshape_predict(self, out):
        scores, classes, boxes = out
        reshape_out = []
        B = len(scores)
        for i in range(B):
            img_box = []
            for j, score in enumerate(scores[i]):
                img_box.append(
                    [scores[i][j].item(), classes[i][j].item(), boxes[i][j][0].item(), boxes[i][j][1].item(), boxes[i][j][2].item(), boxes[i][j][3].item()])
            reshape_out.append(img_box)
        return reshape_out
