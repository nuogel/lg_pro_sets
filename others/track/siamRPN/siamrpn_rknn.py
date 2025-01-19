import os
import sys
import numpy as np
import time
import cv2
import glob
import torch
import torch.nn.functional as F
import math
from rknn.api import RKNN

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import math
from collections import namedtuple

Corner = namedtuple('Corner', 'x1 y1 x2 y2')
# alias
BBox = Corner
Center = namedtuple('Center', 'x y w h')


def corner2center(corner):
    """ convert (x1, y1, x2, y2) to (cx, cy, w, h)
    Args:
        conrner: Corner or np.array (4*N)
    Return:
        Center or np.array (4 * N)
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


def center2corner(center):
    """ convert (cx, cy, w, h) to (x1, y1, x2, y2)
    Args:
        center: Center or np.array (4 * N)
    Return:
        center or np.array (4 * N)
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2


class Anchors:
    """
    This class generate anchors.
    """

    def __init__(self, stride, ratios, scales, image_center=0, size=0):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.image_center = image_center
        self.size = size

        self.anchor_num = len(self.scales) * len(self.ratios)

        self.anchors = None

        self.generate_anchors()

    def generate_anchors(self):
        """
        generate anchors based on predefined configuration
        """
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride * self.stride
        count = 0
        for r in self.ratios:
            ws = int(math.sqrt(size * 1. / r))
            hs = int(ws * r)

            for s in self.scales:
                w = ws * s
                h = hs * s
                self.anchors[count][:] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5][:]
                count += 1

    def generate_all_anchors(self, im_c, size):
        """
        im_c: image center
        size: image size
        """
        if self.image_center == im_c and self.size == size:
            return False
        self.image_center = im_c
        self.size = size

        a0x = im_c - size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori

        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]

        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1),
                             [x1, y1, x2, y2])
        cx, cy, w, h = corner2center([x1, y1, x2, y2])

        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride

        cx = cx + disp_x
        cy = cy + disp_y

        # broadcast
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.all_anchors = (np.stack([x1, y1, x2, y2]).astype(np.float32),
                            np.stack([cx, cy, w, h]).astype(np.float32))
        return True


def get_frames(video_name):
    """获取视频帧

    Args:
        video_name (_type_): _description_

    Yields:
        _type_: _description_
    """
    if not video_name:
        rtsp = "rtsp://%s:%s@%s:554/cam/realmonitor?channel=1&subtype=1" % ("admin", "123456", "192.168.1.108")
        cap = cv2.VideoCapture(rtsp) if rtsp else cv2.VideoCapture()

        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                # print('读取成功===>>>', frame.shape)
                yield cv2.resize(frame, (800, 600))
            else:
                break
    elif video_name.endswith('avi') or \
            video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = sorted(glob(os.path.join(video_name, 'img', '*.jp*')))
        for img in images:
            frame = cv2.imread(img)
            yield frame


class SiameseTracker:
    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite("2.png",im_patch)

        # im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        # im_patch = im_patch.astype(np.float32)
        # im_patch = torch.from_numpy(im_patch)
        return im_patch


class MFTrackerORT(SiameseTracker):
    def __init__(self) -> None:
        self.debug = True
        self.init_track_net()
        self.max_score_decay = 1.0
        self.search_factor = 4.5
        self.search_size = 255
        self.template_factor = 2.0
        self.template_size = 127
        self.update_interval = 200
        self.online_size = 1
        self.score_size = (255 - 127) // 8 + 1 + 8
        self.anchors = self.generate_anchor(self.score_size)
        self.anchor_num = 5
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)

    def generate_anchor(self, score_size):
        anchors = Anchors(8,
                          [0.33, 0.5, 1, 2, 3],
                          [8])
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def init_track_net(self):
        """使用设置的参数初始化tracker网络
        """
        self.rknn = RKNN(verbose=True)
        onnx_path = "/home/luogeng/code/lg_pro_sets/saved/checkpoint/siamrpn_onnx_ized.onnx"

        print('--> Config model')
        self.rknn.config(mean_values=[[0, 0, 0], [0, 0, 0]], std_values=[[1, 1, 1], [1, 1, 1]],
                         target_platform='rk3588')
        print('done')
        ret = self.rknn.load_onnx(model=onnx_path, input_size_list=[[1, 3, 255, 255], [1, 3, 127, 127]])

        # self.rknn.hybrid_quantization_step1(
        # dataset="/data1/wxf/rknn-toolkit2-v2.2.0-2024-09-18/rknn-toolkit2/examples/onnx/yolov5/dataset.txt",  # 表示模型量化所需要的数据集
        # rknn_batch_size=1,  # 表示自动调整模型输入batch数量
        # proposal=True,  # 设置为True，可以自动产生混合量化的配置建议，比较耗时
        # # proposal= True,  # 设置为True，可以自动产生混合量化的配置建议，比较耗时
        # proposal_dataset_size=1,  # 第三步骤所用的图片
        # )

        # time.sleep(10)

        # self.rknn.hybrid_quantization_step2(
        # model_input = "/data1/wxf/rknn-toolkit2-v2.2.0-2024-09-18/rknn-toolkit2/examples/onnx/yolov5/mixformer_v2.model",          # 表示第一步生成的模型文件
        # data_input= "/data1/wxf/rknn-toolkit2-v2.2.0-2024-09-18/rknn-toolkit2/examples/onnx/yolov5/mixformer_v2.data",             # 表示第一步生成的配置文件
        # model_quantization_cfg="/data1/wxf/rknn-toolkit2-v2.2.0-2024-09-18/rknn-toolkit2/examples/onnx/yolov5/mixformer_v2.quantization.cfg"  # 表示第一步生成的量化配置文件
        # )

        print('--> Building model')
        ret = self.rknn.build(do_quantization=True,
                              dataset="/home/luogeng/code/lg_pro_sets/others/track/siamRPN/dataset.txt")
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('done')

        print('--> Export rknn model')
        ret = self.rknn.export_rknn("mixformer.rknn")
        if ret != 0:
            print('Export rknn model failed!')
            exit(ret)
        print('done')

        # Init runtime environment
        print('--> Init runtime environment')
        ret = self.rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)
        print('done')

        # ret = self.rknn.codegen(output_path='./rknn_app_demo',
        # inputs=['/data1/wxf/rknn-toolkit2-v2.2.0-2024-09-18/rknn-toolkit2/examples/onnx/yolov5/1.jpg',"/data1/wxf/rknn-toolkit2-v2.2.0-2024-09-18/rknn-toolkit2/examples/onnx/yolov5/2.png"], overwrite=True)

    def track_init(self, frame, target_pos=None, target_sz=None):
        """使用第一帧进行初始化

        Args:
            frame (_type_): _description_
            target_pos (_type_, optional): _description_. Defaults to None.
            target_sz (_type_, optional): _description_. Defaults to None.
        """
        self.trace_list = []
        try:
            # [x, y, w, h]
            init_state = [target_pos[0], target_pos[1], target_sz[0], target_sz[1]]

            self.online_image = frame
            self.max_pred_score = -1.0
            self.center_pos = np.array([init_state[0] + (init_state[2] - 1) / 2,
                                        init_state[1] + (init_state[3] - 1) / 2])
            self.size = np.array([init_state[2], init_state[3]])

            # calculate z crop size
            w_z = self.size[0] + 0.5 * np.sum(self.size)
            h_z = self.size[1] + 0.5 * np.sum(self.size)
            s_z = round(np.sqrt(w_z * h_z))

            # calculate channle average
            self.channel_average = np.mean(frame, axis=(0, 1))

            # get crop
            self.z_crop = self.get_subwindow(frame, self.center_pos, 127,
                                             s_z, self.channel_average)

            # save states
            self.state = init_state
            self.frame_id = 0
            print(f"第一帧初始化完毕！")
        except:
            print(f"第一帧初始化异常！")
            exit()

    def _convert_score(self, score):
        score = torch.Tensor(score)

        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _convert_bbox(self, delta, anchor):
        delta = torch.Tensor(delta)
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def track(self, image, info: dict = None):
        w_z = self.size[0] + 0.5 * np.sum(self.size)
        h_z = self.size[1] + 0.5 * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = 127 / s_z
        H, W, _ = image.shape
        self.frame_id += 1
        s_x = s_z * (255 / 127)

        x_crop = self.get_subwindow(image, self.center_pos, 255,
                                    round(s_x), self.channel_average)

        # print("x_crop",x_crop)

        # print("self.z_crop",self.z_crop)

        # np.save('rknn_input_zcrop.npy', self.z_crop)
        # np.save('rknn_input_xcrop.npy', x_crop)
        outputs = self.rknn.inference(inputs=[self.z_crop, x_crop], data_format=['nhwc', 'nhwc'])
        # np.save('rknn_input_outpus0.npy', np.asarray(outputs[0]))
        # print('outputs', outputs)
        pred_bbox = self._convert_bbox(outputs[1], self.anchors)

        score = self._convert_score(outputs[0])
        # print("pred_bbox", pred_bbox)
        # print("pred_score", score)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * 0.04)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - 0.44) + \
                 self.window * 0.44
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * 0.4

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, image.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        if self.debug:
            x1, y1, w, h = bbox
            cv2.rectangle(image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
        return {
            'bbox': bbox,
            'best_score': best_score
        }

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def to_numpy(self, tensor):
        if self.fp16:
            return tensor.detach().cpu().half().numpy() if tensor.requires_grad else tensor.cpu().half().numpy()
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def sample_target(self, im, target_bb, search_area_factor, output_sz=None, mask=None):
        """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

        args:
            im - cv image
            target_bb - target box [x, y, w, h]
            search_area_factor - Ratio of crop size to target size
            output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

        returns:
            cv image - extracted crop
            float - the factor by which the crop has been resized to make the crop size equal output_size
        """
        if not isinstance(target_bb, list):
            x, y, w, h = target_bb.tolist()
        else:
            x, y, w, h = target_bb
        # Crop image
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

        if crop_sz < 1:
            raise Exception('Too small bounding box.')

        x1 = int(round(x + 0.5 * w - crop_sz * 0.5))
        x2 = int(x1 + crop_sz)

        y1 = int(round(y + 0.5 * h - crop_sz * 0.5))
        y2 = int(y1 + crop_sz)

        x1_pad = int(max(0, -x1))
        x2_pad = int(max(x2 - im.shape[1] + 1, 0))

        y1_pad = int(max(0, -y1))
        y2_pad = int(max(y2 - im.shape[0] + 1, 0))

        # Crop target
        im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
        if mask is not None:
            mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

        # Pad
        im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)
        # deal with attention mask
        H, W, _ = im_crop_padded.shape
        att_mask = np.ones((H, W))
        end_x, end_y = -x2_pad, -y2_pad
        if y2_pad == 0:
            end_y = None
        if x2_pad == 0:
            end_x = None
        att_mask[y1_pad:end_y, x1_pad:end_x] = 0
        if mask is not None:
            mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

        if output_sz is not None:
            resize_factor = output_sz / crop_sz
            im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
            att_mask = cv2.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
            if mask is None:
                return im_crop_padded, resize_factor, att_mask
            mask_crop_padded = \
                F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear',
                              align_corners=False)[0, 0]
            return im_crop_padded, resize_factor, att_mask, mask_crop_padded

        else:
            if mask is None:
                return im_crop_padded, att_mask.astype(np.bool_), 1.0
            return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded


if __name__ == '__main__':
    print("测试")
    Tracker = MFTrackerORT()
    first_frame = True
    Tracker.video_name = "/media/luogeng/ssd_datasets/code/uav_code/datasets/track_video.mp4"

    if Tracker.video_name:
        video_name = Tracker.video_name
    else:
        video_name = 'webcam'
    # cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    frame_id = 0
    total_time = 0
    for frame in get_frames(Tracker.video_name):
        # print(f"frame shape {frame.shape}")
        # frame = cv2.imread('/media/luogeng/ssd_datasets/code/uav_code/0.jpg')

        tic = cv2.getTickCount()
        if first_frame:
            # x, y, w, h = cv2.selectROI(video_name, frame, fromCenter=False)
            x, y, w, h = 1028, 623, 125, 87
            target_pos = [x, y]
            target_sz = [w, h]
            print('====================type=================', target_pos, type(target_pos), type(target_sz))
            Tracker.track_init(frame, target_pos, target_sz)
            first_frame = False
        else:
            state = Tracker.track(frame)
            frame_id += 1
            print(frame_id, ':update finished')
            print(state)
            # cv2.imwrite(f'/data1/wxf/datas/track/{frame_id}_rpn.jpg', frame)
            cv2.imshow('Tracking', frame)
            cv2.waitKey(1)

        toc = cv2.getTickCount() - tic
        toc = int(1 / (toc / cv2.getTickFrequency()))
        total_time += toc
        print('Video: {:12s} {:3.1f}fps'.format('tracking', toc))

    # print('video: average {:12s} {:3.1f} fps'.format('finale average tracking fps', total_time / (frame_id - 1)))
    # cv2.destroyAllWindows()
