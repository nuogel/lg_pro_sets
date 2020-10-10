import cv2
from util.util_data_aug import Dataaug
import numpy as np
import torch
import random
import math


class LgTransformer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataaug = Dataaug(cfg)

    def aug_mosaic(self, imglabs):
        # loads images in a mosaic

        labels4 = []
        s = 640
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y

        for i, imglab in enumerate(imglabs):
            # Load image
            img, lab, info = imglab
            h0, w0, c = img.shape
            r = s / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
            h, w, c = img.shape
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels = lab.copy()
            if lab.size > 0:  # Normalized xywh to pixel xyxy format
                labels[:, 1] = r * lab[:, 1] + padw
                labels[:, 2] = r * lab[:, 2] + padh
                labels[:, 3] = r * lab[:, 3] + padw
                labels[:, 4] = r * lab[:, 4] + padh
            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine


        # Augment
        # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
        if self.cfg.TRAIN.AFFINE:
            img4, labels4 = self.random_affine(img4, labels4,
                                               degrees=1.98,
                                               translate=0.05,
                                               scale=0.05,
                                               shear=0.64,
                                               border=-s // 2)  # border to remove

        return img4, labels4

    def random_affine(self, img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
        # targets = [cls, xyxy]

        height = img.shape[0] + border * 2
        width = img.shape[1] + border * 2

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 + scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
        T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Combined rotation matrix
        M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
        if (border != 0) or (M != np.eye(3)).any():  # image changed
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

        # Transform label coordinates
        n = len(targets)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # # apply angle-based reduction of bounding boxes
            # radians = a * math.pi / 180
            # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            # x = (xy[:, 2] + xy[:, 0]) / 2
            # y = (xy[:, 3] + xy[:, 1]) / 2
            # w = (xy[:, 2] - xy[:, 0]) * reduction
            # h = (xy[:, 3] - xy[:, 1]) * reduction
            # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
            i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

            targets = targets[i]
            targets[:, 1:5] = xy[i]

        return img, targets

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
        if isinstance(label, list):
            for lab in label:  # add 0 for label
                lab.insert(0, 0)
            label = torch.Tensor(label)
        if isinstance(label, np.ndarray):
            label = np.insert(label, 0, 0, 1)
            label = torch.from_numpy(label)
        return img, label

    def relative_label(self, img, label):
        # RELATIVE LABELS:
        h, w, c = img.shape
        if isinstance(label, list):
            if not label:
                label = None
            else:  # x1y1x2y2
                for lab in label:
                    lab[1] /= w
                    lab[2] /= h
                    lab[3] /= w
                    lab[4] /= h
        if isinstance(label, np.ndarray):
            mask = np.asarray([1, w, h, w, h], np.float32)
            label = label / mask
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
