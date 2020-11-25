import torch.nn as nn
import torch
import math
from lgdet.loss.ObdLoss_FCOS import coords_fmap2orig


class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, x):
        return torch.exp(x * self.scale)


class ClsCntRegHead(nn.Module):
    def __init__(self, in_channel, class_num, GN=True, cnt_on_reg=True, prior=0.01):
        '''
        Args  
        in_channel  
        class_num  
        GN  
        prior  
        '''
        super(ClsCntRegHead, self).__init__()
        self.prior = prior
        self.class_num = class_num
        self.cnt_on_reg = cnt_on_reg

        cls_branch = []
        reg_branch = []

        for i in range(4):
            cls_branch.append(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True))
            if GN:
                cls_branch.append(nn.GroupNorm(32, in_channel))
            cls_branch.append(nn.ReLU(True))

            reg_branch.append(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True))
            if GN:
                reg_branch.append(nn.GroupNorm(32, in_channel))
            reg_branch.append(nn.ReLU(True))

        self.cls_conv = nn.Sequential(*cls_branch)
        self.reg_conv = nn.Sequential(*reg_branch)

        self.cls_logits = nn.Conv2d(in_channel, class_num, kernel_size=3, padding=1)
        self.cnt_logits = nn.Conv2d(in_channel, 1, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(in_channel, 4, kernel_size=3, padding=1)

        self.apply(self.init_conv_RandomNormal)

        nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior) / prior))
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(5)])

    def init_conv_RandomNormal(self, module, std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs):
        '''inputs:[P3~P7]'''
        cls_logits = []
        cnt_logits = []
        reg_preds = []
        for index, P in enumerate(inputs):
            cls_conv_out = self.cls_conv(P)
            reg_conv_out = self.reg_conv(P)

            cls_logits.append(self.cls_logits(cls_conv_out))
            if not self.cnt_on_reg:
                cnt_logits.append(self.cnt_logits(cls_conv_out))
            else:
                cnt_logits.append(self.cnt_logits(reg_conv_out))
            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_conv_out)))
        return cls_logits, cnt_logits, reg_preds


class DetectHead(nn.Module):
    def __init__(self, score_threshold, nms_iou_threshold, max_detection_boxes_num, strides, config=None):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detection_boxes_num = max_detection_boxes_num
        self.strides = strides
        self.config = config
        self.device = self.config.TRAIN.DEVICE

    def forward(self, inputs):
        '''
        inputs  list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        cls_logits, coords = self._reshape_cat_out(inputs[0], self.strides)  # [batch_size,sum(_h*_w),class_num]
        cnt_logits, _ = self._reshape_cat_out(inputs[1], self.strides)  # [batch_size,sum(_h*_w),1]
        reg_preds, _ = self._reshape_cat_out(inputs[2], self.strides)  # [batch_size,sum(_h*_w),4]

        cls_preds = cls_logits.sigmoid_()
        cnt_preds = cnt_logits.sigmoid_()

        coords = coords.to(self.device)

        # cls_scores, cls_classes = torch.max(cls_preds, dim=-1)  # [batch_size,sum(_h*_w)]
        if self.config.add_centerness:
            cls_preds = torch.sqrt(cls_preds * cnt_preds)  # [batch_size,sum(_h*_w)]
        # cls_classes = cls_classes + 1  # [batch_size,sum(_h*_w)]

        boxes_x1y1x2y2 = self._coords2boxes(coords, reg_preds)  # [batch_size,sum(_h*_w),4]

        return cls_preds, boxes_x1y1x2y2
        # # select topk
        # max_num = min(self.max_detection_boxes_num, cls_scores.shape[-1])
        # topk_ind = torch.topk(cls_scores, max_num, dim=-1, largest=True, sorted=True)[1]  # [batch_size,max_num]
        # _cls_scores = []
        # _cls_classes = []
        # _boxes = []
        # for batch in range(cls_scores.shape[0]):
        #     _cls_scores.append(cls_scores[batch][topk_ind[batch]])  # [max_num]
        #     _cls_classes.append(cls_classes[batch][topk_ind[batch]])  # [max_num]
        #     _boxes.append(boxes[batch][topk_ind[batch]])  # [max_num,4]
        # cls_scores_topk = torch.stack(_cls_scores, dim=0)  # [batch_size,max_num]
        # cls_classes_topk = torch.stack(_cls_classes, dim=0)  # [batch_size,max_num]
        # boxes_topk = torch.stack(_boxes, dim=0)  # [batch_size,max_num,4]
        # assert boxes_topk.shape[-1] == 4
        #
        # return self._post_process([cls_scores_topk, cls_classes_topk, boxes_topk])

    def _post_process(self, preds_topk):
        '''
        cls_scores_topk [batch_size,max_num]
        cls_classes_topk [batch_size,max_num]
        boxes_topk [batch_size,max_num,4]
        '''
        _cls_scores_post = []
        _cls_classes_post = []
        _boxes_post = []
        cls_scores_topk, cls_classes_topk, boxes_topk = preds_topk
        for batch in range(cls_classes_topk.shape[0]):
            mask = cls_scores_topk[batch] >= self.score_threshold
            _cls_scores_b = cls_scores_topk[batch][mask]  # [?]
            _cls_classes_b = cls_classes_topk[batch][mask]  # [?]
            _boxes_b = boxes_topk[batch][mask]  # [?,4]
            nms_ind = self.batched_nms(_boxes_b, _cls_scores_b, _cls_classes_b, self.nms_iou_threshold)
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
        # scores, classes, boxes = torch.stack(_cls_scores_post, dim=0), torch.stack(_cls_classes_post, dim=0), torch.stack(_boxes_post, dim=0)

        return _cls_scores_post, _cls_classes_post, _boxes_post

    @staticmethod
    def box_nms(boxes, scores, thr):
        '''
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0] == 0:
            return torch.zeros(0, device=boxes.device).long()
        assert boxes.shape[-1] == 4
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()
                keep.append(i)

            xmin = x1[order[1:]].clamp(min=float(x1[i]))
            ymin = y1[order[1:]].clamp(min=float(y1[i]))
            xmax = x2[order[1:]].clamp(max=float(x2[i]))
            ymax = y2[order[1:]].clamp(max=float(y2[i]))
            inter = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            idx = (iou <= thr).nonzero().squeeze()
            if idx.numel() == 0:
                break
            order = order[idx + 1]
        return torch.LongTensor(keep)

    def batched_nms(self, boxes, scores, idxs, iou_threshold):

        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep

    def _coords2boxes(self, coords, offsets):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        '''
        x1y1 = coords[None, :, :] - offsets[..., :2]
        x2y2 = coords[None, :, :] + offsets[..., 2:]  # [batch_size,sum(_h*_w),2]
        boxes = torch.cat([x1y1, x2y2], dim=-1)  # [batch_size,sum(_h*_w),4]
        return boxes

    def _reshape_cat_out(self, inputs, strides):
        '''
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        batch_size = inputs[0].shape[0]
        c = inputs[0].shape[1]
        out = []
        coords = []
        for pred, stride in zip(inputs, strides):
            pred = pred.permute(0, 2, 3, 1)
            coord = coords_fmap2orig(pred, stride).to(device=pred.device)
            pred = torch.reshape(pred, [batch_size, -1, c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out, dim=1), torch.cat(coords, dim=0)



