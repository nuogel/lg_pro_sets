# copy smale target with bounding boxes to background pics.
import json
import os
import cv2
import random
import numpy as np
import tqdm


class CopyLittleTarget:
    def __init__(self, imgs_path, labs_path, source_imgs_path, source_labs_path, copy_number, save_path):
        self.imgs_path = imgs_path
        self.labs_path = labs_path
        self.source_imgs_path = source_imgs_path
        self.source_labs_path = source_labs_path

        self.copy_number = copy_number
        self.img_list = os.listdir(self.imgs_path)
        self.source_imgs_list = os.listdir(self.source_imgs_path)

        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'labels'), exist_ok=True)

    def run(self):
        targets_dict = {'垃圾': [os.path.join(self.source_imgs_path, p) for p in self.source_imgs_list]}
        for ii, img_name in enumerate(self.img_list):
            print('deeling with pic:', ii, '-', img_name)
            dest_img, dest_mask = self._make_dest_images(img_name, targets_dict)
            #
            cv2.imshow('img', dest_img)
            cv2.waitKey()
            # cv2.imshow('img', dest_mask)
            # cv2.waitKey()

            dest_lab_path = os.path.join(self.save_path, 'labels', img_name.replace('.jpg', '.bmp'))
            dest_img_path = os.path.join(self.save_path, 'images', img_name)
            cv2.imwrite(dest_img_path, dest_img)
            cv2.imwrite(dest_lab_path, dest_mask)

    def _read_json_mask(self, file):
        f = open(file)
        source_bboxes = json.load(f)
        labels = []
        for annoinfo in source_bboxes['annotateInfo']:
            label = {}
            name = annoinfo['entityDetail']['name']
            positions = json.loads(annoinfo['positions'][0]['positions'])['meaningful']
            points = []
            for position in positions:
                points.append([int(position['x']), int(position['y'])])
            label['name'] = name
            label['points'] = points
            labels.append(label)
        return labels

    def _show_mask(self, img, label):
        for bblines in label:
            contours = np.asarray(bblines['points'], dtype=np.int)
            img = cv2.polylines(img, [contours], True, (0, 255, 255), 1)
        cv2.imshow('img', img)
        cv2.waitKey()

    def _read_source_targets(self, mask_img_path, mask_lab_path):  # little target with its bounding box as a pic size.
        mask_img = cv2.imread(mask_img_path)
        mask = np.zeros(mask_img.shape[:2], dtype=np.uint8)
        if mask_lab_path is not None:
            mask_points = self._read_json_mask(mask_lab_path)
            # self._show_mask(mask_img, mask_points)
            for mask_point in mask_points:
                if mask_point['name'] == '垃圾':
                    mask = cv2.fillPoly(mask, [np.asarray(mask_point['points'])], 255)
            # cv2.imshow('mask', mask)
            # cv2.waitKey()
        return mask_img, mask

    def _make_dest_images(self, dest_img, targets_dict):
        dest_img_path = os.path.join(self.imgs_path, dest_img)
        if self.labs_path:
            dest_lab_path = os.path.join(self.labs_path, dest_img.replace('.jpg', '.json'))
        else:
            dest_lab_path=self.labs_path
        dest_img, dest_mask = self._read_source_targets(dest_img_path, dest_lab_path)
        d_size = dest_img.shape
        for k, v in targets_dict.items():
            if len(v) >= self.copy_number:
                s_targets = random.sample(v, self.copy_number)
            else:
                s_targets = v
            for s_target_p in s_targets:
                s_img, s_mask = self._read_source_targets(s_target_p, os.path.join(self.source_labs_path,
                                                                                   os.path.basename(s_target_p).replace(
                                                                                       '.jpg', '.json')))
                max_hw = 60
                if s_img.shape[0] > max_hw or s_img.shape[1] > max_hw:
                    s_img = cv2.resize(s_img, (max_hw, max_hw))
                    s_mask = cv2.resize(s_mask, (max_hw, max_hw))
                s_size = s_img.shape
                x1 = random.randint(0, d_size[1] - s_size[1])
                y1 = random.randint(0, d_size[0] - s_size[0])

                d_box = dest_mask[y1:y1 + s_size[0], x1:x1 + s_size[1]]
                if d_box.sum() == 0:
                    s_mask_inv = cv2.bitwise_not(s_mask)
                    s_img = cv2.bitwise_and(s_img, s_img, mask=s_mask)

                    d_img = dest_img[y1:y1 + s_size[0], x1:x1 + s_size[1]]
                    d_img = cv2.bitwise_and(d_img, d_img, mask=s_mask_inv)

                    dst = cv2.add(s_img, d_img)  # 相加即可

                    dest_img[y1:y1 + s_size[0], x1:x1 + s_size[1]] = dst
                    dest_mask[y1:y1 + s_size[0], x1:x1 + s_size[1]] = s_mask

        return dest_img, dest_mask


if __name__ == '__main__':
    imgs_path = '/media/dell/data/garbage/合川城管_公安数据_公用背景训练/neg_image'
    labs_path = None#'/media/dell/data/garbage/道路垃圾-合川摆拍-雨天-训练集/labels'

    source_imgs_path = '/media/dell/data/garbage/合川城管-打包垃圾-网络下载-前景/images'#'/media/dell/data/garbage/河道垃圾_自动训练平台_训练集_前景垃圾_竞赛_修正/images'
    source_labs_path = '/media/dell/data/garbage/合川城管-打包垃圾-网络下载-前景/labels'#'/media/dell/data/garbage/河道垃圾_自动训练平台_训练集_前景垃圾_竞赛_修正/labels'

    save_path = '/media/dell/data/garbage/道路垃圾合成_打包垃圾'

    copy_number = 50
    clt = CopyLittleTarget(imgs_path, labs_path, source_imgs_path, source_labs_path, copy_number, save_path)
    clt.run()
