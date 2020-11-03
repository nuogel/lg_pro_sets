import cv2


def resize_save(img_path, save_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    img_resize = cv2.resize(img, (w // 4, h // 4))
    cv2.imwrite(save_path, img_resize)


if __name__ == '__main__':
    img_path = 'E://datasets//kitti//training//image_2//000000.png'
    save_path = '00000_1_4X.png'
    resize_save(img_path, save_path)
