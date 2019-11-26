import torch
from torchvision.utils import save_image
import cv2
import numpy as np


class SR_SCORE:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rate_all = 0.
        self.rate_batch = 0.
        self.batches = 1

    def init_parameters(self):
        self.rate_all = 0.
        self.rate_batch = 0.
        self.batches = 1

    def cal_score(self, pre, dataset):
        input, target = dataset
        self.rate_batch = 0.
        print('batch NO:', self.batches)
        self.batches += 1
        save_name = 'pre'
        self._show_tensor_images(pre, time=0, save_name=save_name)
        print('saved img')

    def score_out(self):
        return self.rate_all / self.batches, None, None

    def _show_tensor_images(self, tensor_imgs, time=1000, save_name=None):
        for i in range(tensor_imgs.shape[0]):
            img = tensor_imgs[i]
            img = img.mul(1.0).clamp(0, 255).round().to('cpu', torch.uint8).numpy()
            if time:
                cv2.imshow('img', img)
                cv2.waitKey(time)
            if save_name:
                cv2.imwrite(save_name + '_{}.jpg'.format(self.rate_all), img)

    def denormalize(self, tensors):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        """ Denormalizes image tensors using mean and std """
        for c in range(3):
            tensors[:, c].mul_(std[c]).add_(mean[c])
        return torch.clamp(tensors, 0, 255)
