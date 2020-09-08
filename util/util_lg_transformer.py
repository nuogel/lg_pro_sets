import cv2
from util.util_data_aug import Dataaug
import numpy as np
import torch


class LgTransformer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataaug = Dataaug(cfg)

    def data_aug(self, img, label):
        labels = 'None'
        try_tims = 0
        while labels is 'None':
            imgs, labels = self.dataaug.augmentation(aug_way_ids=([11, 20, 21, 22], [26]),
                                                     datas=([img], [label]))  # [11,20, 21, 22]
            try_tims += 1
        img_after = imgs[0]
        label_after = labels[0]
        return img_after, label_after

    def resize(self, img, label, size):
        img_after = cv2.resize(img, (size[1], size[0]))
        img_size = img.shape
        for lab in label:
            lab[1] /= (img_size[1] / size[1])
            lab[2] /= (img_size[0] / size[0])
            lab[3] /= (img_size[1] / size[1])
            lab[4] /= (img_size[0] / size[0])

        return img_after, label

    def transpose(self, img, label):
        img = np.asarray(img, dtype=np.float32)
        img = self.imnormalize(img, self.cfg.mean, self.cfg.std, to_rgb=True)
        if label:
            for lab in label:  # add 0 for label
                lab.insert(0, 0)
            label = torch.Tensor(label)
        return img, label

    def relative_label(self, img, label):
        # RELATIVE LABELS:
        img_size = img.shape
        if not label:
            label = None
        else:  # x1y1x2y2
            for lab in label:
                lab[1] /= img_size[1]
                lab[2] /= img_size[0]
                lab[3] /= img_size[1]
                lab[4] /= img_size[0]
        return img, label

    def rescale_size(self, old_size, scale, return_scale=False):
        """Calculate the new size to be rescaled to."""

        w, h = old_size

        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))

        new_size = self._scale_size((w, h), scale_factor)

        return new_size, scale_factor

    def pad_to_size(self, img, label, size, decode=False):
        h, w, c = img.shape
        ratio_w = size[1] / w
        ratio_h = size[0] / h
        # if ratio_w
        if ratio_w < ratio_h:
            ratio_min = ratio_w
        else:
            ratio_min = ratio_h

        # resize:

        img = cv2.resize(img, None, fx=ratio_min, fy=ratio_min)
        h, w, c = img.shape
        # Determine padding
        # pad =[left, right, top, bottom]
        if abs(w - size[1]) > abs(h - size[0]):
            dim_diff = abs(w - size[1])
            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            pad = (pad1, pad2, 0, 0)
        else:
            dim_diff = abs(h - size[0])
            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            pad = (0, 0, pad1, pad2)

        # Add padding
        img = cv2.copyMakeBorder(img, pad[2], pad[3], pad[0], pad[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if not decode:
            label_after = [[
                lab[0],
                lab[1] * ratio_min + pad[0],
                lab[2] * ratio_min + pad[2],
                lab[3] * ratio_min + pad[1],
                lab[4] * ratio_min + pad[3]
            ] for lab in label]
        else:
            label_after = [[
                lab[0],
                (lab[1] - pad[0]) / ratio_min,
                (lab[2] - pad[2]) / ratio_min,
                (lab[3] - pad[1]) / ratio_min,
                (lab[4] - pad[3]) / ratio_min,
            ] for lab in label]

        return img, label_after

    def _scale_size(self, size, scale):
        """Rescale a size by a ratio.

        Args:
            size (tuple[int]): (w, h).
            scale (float): Scaling factor.

        Returns:
            tuple[int]: scaled size.
        """
        w, h = size
        return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)

    def imnormalize(self, img, mean, std, to_rgb=True):
        """Inplace normalize an image with mean and std.

        Args:
            img (ndarray): Image to be normalized.
            mean (ndarray): The mean to be used for normalize.
            std (ndarray): The std to be used for normalize.
            to_rgb (bool): Whether to convert to rgb.

        Returns:
            ndarray: The normalized image.
        """
        # cv2 inplace normalization does not accept uint8
        assert img.dtype != np.uint8
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        return img

    def imdenormalize(self, img, mean, std, to_bgr=True):
        assert img.dtype != np.uint8
        mean = mean.reshape(1, -1).astype(np.float64)
        std = std.reshape(1, -1).astype(np.float64)
        img = cv2.multiply(img, std)  # make a copy
        cv2.add(img, mean, img)  # inplace
        if to_bgr:
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace
        img = img.astype(np.uint8)
        return img
