import math
import cv2
import numpy as np
import os


class PSNR:
    '''
    PSNR:XXXX
    '''

    def __init__(self, use_torch_mse=True):
        self.use_MSE = use_torch_mse

    def run_image(self, path_image1, path_image2):
        img1 = cv2.imread(path_image1).astype(np.float32) / 255.0
        img2 = cv2.imread(path_image2).astype(np.float32) / 255.0
        assert img1.shape == img2.shape, 'ERROR: shape is not the same.'
        if self.use_MSE:
            import torch
            self.mse = torch.nn.MSELoss()
            mse = self.mse(torch.FloatTensor(img1), torch.FloatTensor(img2))

        else:
            shape = img1.shape
            mse = np.sum((img1 - img2) ** 2 / (shape[0] * shape[1] * shape[2]))

        psnr = 10 * math.log10(1.0 / mse.item())
        print('PSNR: ', psnr)
        return psnr

    def video2images(self, video_path):
        cap = cv2.VideoCapture(video_path)
        i = 0
        ret = True
        while ret:
            ret, frame = cap.read()
            if not ret: break
            write_path = os.path.dirname(video_path)
            # cv2.imshow('img', frame)
            # cv2.waitKey()
            cv2.imwrite(os.path.join(write_path, str(i) + '.jpg'), frame)
            i += 1


if __name__ == '__main__':
    path_1 = 'E:/datasets/kitti/training/images/000001.png'
    path_2 = 'E:/datasets/kitti/training/images/000002.png'
    video_path = 'F:/SR_video/blur-64/raw_video/000_x5.mp4'
    images_raw_file = 'F:/SR_video/blur-64/raw_video/'
    images_rec_file = 'F:/SR_video/blur-64/recover_video/'

    psnr = PSNR(use_torch_mse=True)
    # psnr.video2images(video_path)
    out = 0
    for i in range(100):
        image1 = os.path.join(images_raw_file, str(i) + '.jpg')
        image2 = os.path.join(images_rec_file, str(i) + '.jpg')
        out += psnr.run_image(image1, image2)

    out = out / 100
    print('all: ', out)
