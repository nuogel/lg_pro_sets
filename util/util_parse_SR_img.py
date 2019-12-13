'''
Parse the prediction from SR model.
'''

import cv2
import torch


def parse_Tensor_img(tensor_imgs, pixcels_norm=[177.0, 1.], save_path=None, show_time=20000):
    for i in range(tensor_imgs.shape[0]):
        img = tensor_imgs[i]
        img = img.mul(pixcels_norm[1]).add(pixcels_norm[0]).clamp(0, 255).round().to('cpu', torch.uint8).numpy()
        if save_path:
            cv2.imwrite(save_path, img)
            print('     ---->', save_path)
        if show_time:
            cv2.imshow('img', img)
            cv2.waitKey(show_time)
