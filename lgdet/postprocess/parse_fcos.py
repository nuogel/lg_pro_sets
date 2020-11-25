"""Parse the predictions."""

from lgdet.model.head.fcos_head import DetectHead
from lgdet.util.util_NMS import NMS
from lgdet.util.util_iou import xyxy2xywh


class ParsePredict_fcos:
    def __init__(self, config):
        self.detection_head = DetectHead(config.score_threshold, config.nms_iou_threshold,
                                         config.max_detection_boxes_num, config.strides, config)
        self.NMS = NMS(config)

    def parse_predict(self, predicts):
        pre_score, pre_loc_xyxy = self.detection_head(predicts)
        pre_loc_xywh = xyxy2xywh(pre_loc_xyxy)
        labels_predict = self.NMS.forward(pre_score, pre_loc_xywh)
        return labels_predict

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

    def clip_boxes(self, size, batch_boxes):
        h, w = size
        batch_boxes = batch_boxes.clamp_(min=0)
        batch_boxes[..., [0, 2]] = batch_boxes[..., [0, 2]].clamp_(max=w - 1)
        batch_boxes[..., [1, 3]] = batch_boxes[..., [1, 3]].clamp_(max=h - 1)
        return batch_boxes
